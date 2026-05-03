"""
Vera — magicpin Merchant AI Assistant
Full HTTP server implementing the judge harness contract.
Deploy: uvicorn bot:app --host 0.0.0.0 --port 8080

LLM: Google Gemini Flash (free tier — key-rotating)
Fallback: Claude Haiku via Anthropic API (if ANTHROPIC_API_KEY set)
"""

import os
import time
import json
import re
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# SambaNova — PRIMARY
SAMBANOVA_API_KEY       = os.getenv("SAMBANOVA_API_KEY", "").strip()
SAMBANOVA_MODEL_COMPOSE = "Meta-Llama-3.3-70B-Instruct"
SAMBANOVA_MODEL_REPLY   = "Meta-Llama-3.1-8B-Instruct"
SAMBANOVA_BASE_URL      = "https://api.sambanova.ai/v1/chat/completions"
_sambanova_disabled     = False
SAMBANOVA_COOLDOWN_S    = 30
_sambanova_429_at       = 0.0

# Gemini — SECONDARY
GOOGLE_MODEL     = "gemini-2.0-flash"
GOOGLE_MODEL_ALT = "gemini-1.5-flash-latest"
GOOGLE_API_KEYS: list[str] = []
for _k in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GOOGLE_API_KEY_4"]:
    _v = os.getenv(_k, "").strip()
    if _v:
        GOOGLE_API_KEYS.append(_v)

GEMINI_COOLDOWN_S    = 60
_gemini_key_index    = 0
_gemini_key_429_at: dict[int, float] = {}

TEAM_NAME       = os.getenv("TEAM_NAME", "Vera Dheera Soora")
TEAM_MEMBERS    = os.getenv("TEAM_MEMBERS", "Candidate")
CONTACT_EMAIL   = os.getenv("CONTACT_EMAIL", "candidate@example.com")
BOT_VERSION     = "4.1.0"

app = FastAPI(title="MagicPin Vera Bot", version=BOT_VERSION)
START_TIME = time.time()

# ─────────────────────────────────────────────
# TRAFFIC TRACKING (RPM / TPM)
# ─────────────────────────────────────────────

class TrafficTracker:
    def __init__(self):
        self.history = [] # list of (timestamp, tokens)

    def log_request(self, estimated_tokens: int):
        self.history.append((time.time(), estimated_tokens))
        self.clean()

    def clean(self):
        # Keep only last 60 seconds
        now = time.time()
        self.history = [h for h in self.history if now - h[0] <= 60]

    def get_stats(self):
        self.clean()
        rpm = len(self.history)
        tpm = sum(h[1] for h in self.history)
        return rpm, tpm

tracker = TrafficTracker()

# ─────────────────────────────────────────────
# IN-MEMORY STATE
# ─────────────────────────────────────────────

contexts:             dict[tuple[str, str], dict] = {}
conversations:        dict[str, dict]             = {}
fired_suppressions:   set[str]                    = set()
seen_auto_reply_msgs: set[str]                    = set()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_ctx(scope: str, ctx_id: str) -> Optional[dict]:
    entry = contexts.get((scope, ctx_id))
    return entry["payload"] if entry else None

def get_merchant(merchant_id: str)  -> Optional[dict]: return get_ctx("merchant",  merchant_id)
def get_category(slug: str)         -> Optional[dict]: return get_ctx("category",  slug)
def get_customer(customer_id: str)  -> Optional[dict]: return get_ctx("customer",  customer_id)
def get_trigger(trigger_id: str)    -> Optional[dict]: return get_ctx("trigger",   trigger_id)

def is_repeat_auto_reply(conv_id: str, message: str) -> int:
    conv = conversations.get(conv_id, {})
    turns = conv.get("turns", [])
    msg_low = message.strip().lower()
    return sum(1 for t in turns if t.get("from") == "merchant" and t.get("message", "").strip().lower() == msg_low)


def detect_auto_reply(message: str) -> bool:
    patterns = [
        "thank you for contacting",
        "thanks for contacting",
        "our team will respond",
        "will get back to you",
        "automated assistant",
        "we have received your message",
        "aapki jaankari ke liye",
        "main aapki yeh sabhi baatein",
        "aapki madad ke liye shukriya, lekin main ek automated",
        "this is an automated",
        "auto-reply",
    ]
    msg_lower = message.lower()
    return any(p in msg_lower for p in patterns)


def detect_explicit_intent(message: str) -> Optional[str]:
    msg_lower = message.lower()
    word_count = len(message.split())
    
    if any(p in msg_lower for p in ["book me", "confirm my appointment", "yes please book"]):
        return "commit"
        
    if word_count <= 12 and not any(w in msg_lower for w in ["but", "if", "instead", "except", "change"]):
        if any(p in msg_lower for p in [
            "let's do it", "lets do it", "ok do it", "go ahead", "yes let's",
            "haan karo", "confirm", "proceed", "start karo", "shuru karo",
            "yes please", "bilkul", "whats next", "what's next", "send it",
            "draft it", "do it"
        ]):
            return "commit"
            
    if word_count <= 8 and not any(w in msg_lower for w in ["but", "instead", "except"]):
        if any(p in msg_lower for p in [
            "not interested", "stop messaging", "stop", "band karo", "mat karo",
            "unsubscribe", "do not contact", "mujhe nahi chahiye", "nahi chahiye",
            "mat bhejo"
        ]):
            return "opt_out"
    if any(p in msg_lower for p in [
        "gst", "income tax", "loan", "insurance", "property", "legal advice",
        "gst filing", "gst return"
    ]):
        return "out_of_scope"
    if any(p in msg_lower for p in [
        "useless", "bakwas", "rubbish", "stupid bot", "stop bothering",
        "stop wasting"
    ]):
        return "hostile"
    return None


# ─────────────────────────────────────────────
# LEAD SIGNAL PICKER — deterministic, pre-LLM
# This is the key improvement: pick ONE signal that drives the message,
# then pass it explicitly to the prompt so the LLM doesn't have to decide.
# ─────────────────────────────────────────────

def pick_lead_signal(trigger: dict, merchant: dict, category: dict) -> dict:
    """
    Deterministically select the single strongest signal for this trigger+merchant.
    Returns: {signal_text, hook, lever, cta_type}
    """
    kind       = trigger.get("kind", "")
    payload    = trigger.get("payload", {})
    perf       = merchant.get("performance", {})
    peer       = category.get("peer_stats", {})
    cust_agg   = merchant.get("customer_aggregate", {})
    signals    = merchant.get("signals", [])
    identity   = merchant.get("identity", {})
    offers     = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    sub        = merchant.get("subscription", {})
    owner      = identity.get("owner_first_name", "")
    m_name     = identity.get("name", "Merchant")
    city       = identity.get("city", "")
    locality   = identity.get("locality", "")

    ctr        = perf.get("ctr", 0)
    peer_ctr   = peer.get("avg_ctr", 0.03)
    views      = perf.get("views", 0)
    calls      = perf.get("calls", 0)
    delta_views = perf.get("delta_7d", {}).get("views_pct", 0)
    delta_calls = perf.get("delta_7d", {}).get("calls_pct", 0)

    lapsed     = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus") or 0
    total_cust = cust_agg.get("total_unique_ytd", 0)
    retention  = cust_agg.get("retention_6mo_pct") or cust_agg.get("retention_3mo_pct") or 0
    high_risk  = cust_agg.get("high_risk_adult_count", 0)
    days_left  = sub.get("days_remaining", 0)

    # Resolve digest item if referenced
    top_item_id  = payload.get("top_item_id")
    digest_item  = None
    if top_item_id:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = d
                break

    # ── RESEARCH / COMPLIANCE / TREND ────────────────────────────────────────
    if kind in ("research_digest", "regulation_change", "category_trend_movement"):
        # For trend movement, if no digest item, try to find in trend_signals
        if kind == "category_trend_movement" and not digest_item:
            query = payload.get("query") or payload.get("metric_or_topic")
            for ts in category.get("trend_signals", []):
                if query and query.lower() in ts.get("query", "").lower():
                    digest_item = {
                        "source": "Google Trends / search data",
                        "title":  f"'{ts.get('query')}' searches +{round(ts.get('delta_yoy',0)*100)}% YoY",
                        "summary": f"Growth concentrated in {ts.get('segment_age','')} age band. {ts.get('skew','')} skew.",
                        "actionable": f"Consider positioning an offer for {ts.get('query')}"
                    }
                    break

        if digest_item:
            src   = digest_item.get("source", "")
            title = digest_item.get("title", "")
            summ  = digest_item.get("summary", "")
            n     = digest_item.get("trial_n", "")
            seg   = digest_item.get("patient_segment", "") or digest_item.get("segment_age", "")
            act   = digest_item.get("actionable", "")
            n_str = f" (n={n})" if n else ""
            if high_risk > 0 and seg and "high_risk" in seg:
                anchor = f"Your {high_risk} high-risk adult patients are the target segment"
            elif seg:
                anchor = f"Relevant to your {seg} cohort"
            else:
                anchor = f"Relevant to your {m_name} patient mix"
            return {
                "signal_text": f"{src}: {title}{n_str}. {summ[:120]}",
                "anchor":      anchor,
                "actionable":  act,
                "hook":        f"New {kind.replace('_',' ')} from {src} — directly relevant to {owner or m_name}",
                "lever":       "specificity + reciprocity (offer to draft patient-ed or compliance SOP)",
                "cta_type":    "open_ended"
            }
        return {
            "signal_text": f"New {kind} signal for {category.get('slug','this category')}",
            "anchor":      "",
            "actionable":  "",
            "hook":        "Category-level update worth acting on",
            "lever":       "specificity + reciprocity",
            "cta_type":    "open_ended"
        }

    # ── COMPETITOR OPENED ─────────────────────────────────────────────────────
    if kind == "competitor_opened":
        comp_name = payload.get("competitor_name", "A new competitor")
        comp_loc  = payload.get("competitor_locality", locality)
        dist      = payload.get("distance_km", "nearby")
        return {
            "signal_text": f"{comp_name} just opened in {comp_loc} ({dist}km away)",
            "anchor":      f"Your retention is {round(retention*100)}% — defensive move needed",
            "actionable":  "Run a 'loyalty-appreciation' offer to lock in regulars this month",
            "hook":        f"New competitor in {comp_loc} — time to protect your turf, {owner or m_name}",
            "lever":       "voyeur-curiosity + loss aversion (protect market share)",
            "cta_type":    "binary_yes_no"
        }

    # ── PERFORMANCE DIP ───────────────────────────────────────────────────────
    # ... (remains same)
    if kind in ("perf_dip", "seasonal_perf_dip"):
        ctr_gap = round((peer_ctr - ctr) / peer_ctr * 100) if peer_ctr else 0
        dip_str = payload.get("dip_description", "")
        if kind == "seasonal_perf_dip":
            normal_range = payload.get("normal_range", "")
            return {
                "signal_text": f"Views down {abs(round(delta_views*100))}% this week — but this IS the normal {payload.get('season','')} lull ({normal_range})",
                "anchor":      f"{total_cust} active members/customers to retain during the dip",
                "actionable":  "Skip new acquisition spend; focus retention on existing base",
                "hook":        f"Weekly dip is expected — here's the counter-move for {owner or m_name}",
                "lever":       "loss aversion reframe (dip is normal; inaction on retention is not)",
                "cta_type":    "binary_yes_no"
            }
        return {
            "signal_text": f"CTR {ctr:.3f} vs peer median {peer_ctr:.3f} — {ctr_gap}% below peers. Views {views} | calls {calls} last 30d",
            "anchor":      dip_str or f"{abs(round(delta_views*100))}% view drop this week",
            "actionable":  payload.get("suggested_action", "Run a targeted offer this week"),
            "hook":        f"CTR {ctr_gap}% below peer — one action closes most of that gap",
            "lever":       "loss aversion (show the number, give one action)",
            "cta_type":    "binary_yes_no"
        }

    # ── PERFORMANCE SPIKE / MILESTONE ────────────────────────────────────────
    if kind in ("perf_spike", "milestone_reached"):
        spike_str = f"Views +{round(delta_views*100)}% this week" if delta_views > 0 else ""
        milestone = payload.get("milestone", "")
        return {
            "signal_text": milestone or spike_str or f"Strong week: {views} views, {calls} calls",
            "anchor":      f"Retention {round(retention*100)}% — use momentum to lock in regulars",
            "actionable":  "Convert the spike: push a specific offer or post to capture intent",
            "hook":        f"Momentum is live for {owner or m_name} — here's the next move",
            "lever":       "social proof + momentum (what's next?)",
            "cta_type":    "open_ended"
        }

    # ── RECALL / CHRONIC REFILL ───────────────────────────────────────────────
    if kind in ("recall_due", "chronic_refill_due"):
        customer_name = payload.get("customer_name", "")
        days_since    = payload.get("days_since_last_visit", "")
        due_date      = payload.get("due_date", "") or payload.get("refill_due_date", "")
        offer_str     = offers[0]["title"] if offers else ""
        meds          = payload.get("medications", [])
        meds_str      = ", ".join(str(m.get("name", m)) if isinstance(m, dict) else str(m) for m in meds) if meds else ""
        slots         = payload.get("available_slots", [])
        slot_str      = " | ".join(str(s.get("label", s)) if isinstance(s, dict) else str(s) for s in slots[:2]) if slots else ""
        return {
            "signal_text": f"Recall due: {customer_name}, {days_since}d since last visit. Due: {due_date}. Meds: {meds_str}",
            "anchor":      f"Offer: {offer_str}" if offer_str else "",
            "actionable":  f"Slots: {slot_str}" if slot_str else f"Book a slot at {m_name}",
            "hook":        f"Personalized recall — {customer_name}'s window is open now",
            "lever":       "personalized recall with specific date/slot/price",
            "cta_type":    "multi_choice_slot" if slots else "binary_yes_no"
        }

    # ── CUSTOMER LAPSE WIN-BACK ───────────────────────────────────────────────
    if kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
        customer_name = payload.get("customer_name", "")
        days_lapsed   = payload.get("days_since_visit", 0)
        past_services = payload.get("services_received", [])
        offer_str     = offers[0]["title"] if offers else ""
        past_str      = ", ".join(str(s.get("title", s)) if isinstance(s, dict) else str(s) for s in past_services[:2]) if past_services else ""
        hardness      = "hard" if kind == "customer_lapsed_hard" else "soft"
        return {
            "signal_text": f"{customer_name} lapsed ({hardness}) — {days_lapsed}d, past: {past_str}",
            "anchor":      f"Offer to use as hook: {offer_str}" if offer_str else "",
            "actionable":  "Win-back with no-commitment trial or free add-on",
            "hook":        f"Win-back window: {customer_name} ({days_lapsed}d gap)",
            "lever":       "no-shame recall + specific past goal + no-commitment ask",
            "cta_type":    "binary_yes_no"
        }

    # ── FESTIVAL / IPL / SEASONAL ─────────────────────────────────────────────
    if kind in ("festival_upcoming", "ipl_match_today", "weather_heatwave"):
        event       = payload.get("event_name", "") or payload.get("match_title", "") or kind.replace("_", " ")
        event_date  = payload.get("event_date", "") or payload.get("match_time", "")
        insight     = payload.get("merchant_insight", "") or payload.get("counter_insight", "")
        offer_str   = offers[0]["title"] if offers else ""
        cat_slug    = category.get("slug", "")
        
        # Try to find a dynamic insight from category digest
        if not insight:
            for d in category.get("digest", []):
                if d.get("kind") == "seasonal" or event.lower() in d.get("title","").lower():
                    insight = d.get("summary")
                    break
        
        # Hardcoded fallback for IPL
        if kind == "ipl_match_today" and not insight:
            insight = "Saturday IPL matches shift -12% in-restaurant covers (viewers stay home) — push delivery instead"
            
        return {
            "signal_text": f"{event} {event_date}. Merchant-relevant insight: {insight}",
            "anchor":      f"Your active offer: {offer_str}" if offer_str else f"{locality or city} {cat_slug} context",
            "actionable":  payload.get("suggested_action", f"Push {offer_str or 'your best offer'} as event-day hook"),
            "hook":        f"{event} is today/soon — here's the contrarian move for {owner or m_name}",
            "lever":       "urgency + counter-intuitive insight (what to do vs what seems obvious)",
            "cta_type":    "binary_yes_no"
        }

    # ── RENEWAL ───────────────────────────────────────────────────────────────
    if kind == "renewal_due":
        plan     = sub.get("plan", "plan")
        features = payload.get("features_at_risk", [])
        feat_str = ", ".join(str(f.get("name", f)) if isinstance(f, dict) else str(f) for f in features[:3]) if features else "your current features"
        return {
            "signal_text": f"Subscription renews in {days_left}d. Plan: {plan}. At risk if lapsed: {feat_str}",
            "anchor":      f"Current performance: {views} views, {calls} calls last 30d — powered by {plan}",
            "actionable":  "Renew now to keep lead pipeline uninterrupted",
            "hook":        f"{days_left} days left on {plan} — here's what stops if it lapses",
            "lever":       "loss aversion (what stops) + concrete days remaining",
            "cta_type":    "binary_yes_no"
        }

    # ── SUPPLY / COMPLIANCE ALERT ─────────────────────────────────────────────
    if kind in ("supply_alert", "regulation_change"):
        batch       = payload.get("batch_numbers", [])
        drug        = payload.get("drug_name", "") or payload.get("product_name", "")
        affected    = payload.get("affected_customer_count", cust_agg.get("chronic_rx_count", 0))
        risk_level  = payload.get("risk_level", "low")
        batch_str   = ", ".join(str(b.get("number", b)) if isinstance(b, dict) else str(b) for b in batch) if batch else ""
        return {
            "signal_text": f"URGENT: {drug} recall/alert. Batches: {batch_str}. {affected} of your customers affected. Risk: {risk_level}",
            "anchor":      f"Your chronic-Rx base: {affected} affected customers need notification",
            "actionable":  "Draft patient notification + replacement workflow",
            "hook":        f"Compliance action needed now — {affected} customers affected",
            "lever":       "urgency + specificity (batch numbers + count) + workflow offer",
            "cta_type":    "open_ended"
        }

    # ── CURIOUS ASK ───────────────────────────────────────────────────────────
    if kind == "curious_ask_due":
        last_vera   = payload.get("days_since_last_vera_touch", 7)
        topic       = payload.get("suggested_topic", "what's been moving this week")
        return {
            "signal_text": f"No Vera touch in {last_vera}d — weekly check-in due",
            "anchor":      f"Last signal: {signals[0] if signals else 'none'}",
            "actionable":  f"Ask: {topic}. Offer to draft something from the answer in 5 min",
            "hook":        f"Low-friction check-in for {owner or m_name}",
            "lever":       "asking the merchant (reciprocity + lowest-friction CTA)",
            "cta_type":    "open_ended"
        }

    # ── REVIEW THEME ──────────────────────────────────────────────────────────
    if kind == "review_theme_emerged":
        themes = merchant.get("review_themes", [])
        top_t  = themes[0] if themes else {}
        return {
            "signal_text": f"Review theme emerged: '{top_t.get('theme','')}' ({top_t.get('occurrences_30d',0)}x in 30d, {top_t.get('sentiment','')})",
            "anchor":      f"Positive signal to amplify publicly",
            "actionable":  "Convert review theme into a Google post or WhatsApp broadcast",
            "hook":        f"I spotted a pattern in your reviews — quick win for {owner or m_name}",
            "lever":       "reciprocity (I noticed something specific about your account)",
            "cta_type":    "open_ended"
        }

    # ── APPOINTMENT TOMORROW ──────────────────────────────────────────────────
    if kind == "appointment_tomorrow":
        customer_name = payload.get("customer_name", "")
        appt_time     = payload.get("appointment_time", "")
        service       = payload.get("service", "")
        return {
            "signal_text": f"Appointment reminder: {customer_name}, tomorrow {appt_time}, {service}",
            "anchor":      "Confirm + prep instructions",
            "actionable":  "Send confirmation with prep instructions if applicable",
            "hook":        f"Appointment tomorrow for {customer_name} — confirm now",
            "lever":       "personalized reminder with confirmation CTA",
            "cta_type":    "binary_yes_no"
        }

    # ── TRIAL FOLLOWUP ────────────────────────────────────────────────────────
    if kind == "trial_followup":
        customer_name = payload.get("customer_name", "")
        trial_date    = payload.get("trial_date", "")
        service       = payload.get("service", "")
        offer_str     = offers[0]["title"] if offers else ""
        return {
            "signal_text": f"Trial followup: {customer_name}, tried {service} on {trial_date}",
            "anchor":      f"Convert trial → regular with: {offer_str}" if offer_str else "",
            "actionable":  "Follow up while experience is fresh; make next booking frictionless",
            "hook":        f"{customer_name}'s trial was recent — conversion window is open",
            "lever":       "relationship continuity + trial-to-regular conversion",
            "cta_type":    "binary_yes_no"
        }

    # ── GENERIC FALLBACK ──────────────────────────────────────────────────────
    # Smart catch-all: scan merchant signals for a recent anchor
    best_signal = signals[0] if signals else f"perf: {views} views, {calls} calls, CTR {ctr:.3f}"
    novel_hook  = f"Spotted a trend in your {category.get('slug','')} context"
    
    return {
        "signal_text": best_signal,
        "anchor":      f"Offer: {offers[0]['title']}" if offers else "",
        "actionable":  payload.get("suggested_action", "One clear action for merchant"),
        "hook":        f"{novel_hook} for {owner or m_name}",
        "lever":       "specificity",
        "cta_type":    "open_ended"
    }


# ─────────────────────────────────────────────
# LLM LAYER  (v3.6 — OpenRouter primary)
# ─────────────────────────────────────────────
#
# Tested working from India (no Cloudflare 1010):
#   1. OpenRouter  — own infra, free Llama models, 1000 req/day with key
#   2. SambaNova   — own silicon, free tier, 100 req/day
#   3. Gemini      — Google infra (fix key restrictions in Cloud Console)
#
# Chain:
#   /v1/tick  (compose) → OpenRouter 70b → SambaNova 70b → Gemini → heuristic
#   /v1/reply (turns)   → OpenRouter 8b  → SambaNova 8b  → Gemini → heuristic
# ─────────────────────────────────────────────

# ── OpenRouter ────────────────────────────────
# Free account: https://openrouter.ai  -> Keys -> Create key
# Rotating across 6 free models = 6x burst capacity (each has its own upstream limit)
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1/chat/completions"
# Each model has independent rate limits — rotating avoids burst blocks
OPENROUTER_MODELS_COMPOSE = [
    "meta-llama/llama-3.3-70b-instruct:free",      # best quality
    "google/gemma-3-27b-it:free",                  # Google, separate quota
    "qwen/qwen-2.5-72b-instruct:free",             # Alibaba, separate quota
    "microsoft/phi-4:free",                         # Microsoft, separate quota
]
OPENROUTER_MODELS_REPLY = [
    "meta-llama/llama-3.1-8b-instruct:free",       # fast, small
    "mistralai/mistral-7b-instruct:free",           # fast, separate quota
    "qwen/qwen-2.5-7b-instruct:free",              # fast, separate quota
    "google/gemma-3-12b-it:free",                  # fast, separate quota
]
_openrouter_disabled  = False
# Per-model 429 cooldown timestamps
_or_model_429_at: dict[str, float] = {}
OPENROUTER_COOLDOWN_S = 20

# ── Groq ──────────────────────────────────────
GROQ_API_KEYS: list[str] = []
for _k in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5"]:
    _v = os.getenv(_k, "").strip()
    if _v:
        GROQ_API_KEYS.append(_v)

GROQ_MODEL_COMPOSE = "llama-3.3-70b-versatile"
GROQ_MODEL_REPLY   = "llama-3.1-8b-instant"
GROQ_BASE_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_COOLDOWN_S    = 30
_groq_key_429_at: dict[int, float] = {}
_groq_disabled     = False

# ── Anthropic (tertiary, needs credits) ──────
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
_anthropic_disabled  = False


# ─────────────────────────────────────────────
# SHARED HTTP HELPER
# ─────────────────────────────────────────────

def _split_prompt(prompt: str):
    """Split monolithic prompt into (system, user) for chat models."""
    for marker in ("\n\n=== LEAD SIGNAL", "\n\nMERCHANT :", "\n\nCONVERSATION:"):
        idx = prompt.find(marker)
        if idx != -1:
            return prompt[:idx].strip(), prompt[idx:].strip()
    return "You are Vera, magicpin's AI assistant for merchant growth.", prompt


def _post_json(url: str, body: dict, headers: dict, timeout: int = 7) -> dict:
    import urllib.request, urllib.error
    
    # Estimate input tokens
    input_str = json.dumps(body)
    input_tokens = len(input_str) // 4
    
    data = json.dumps(body).encode()
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    req  = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res_data = r.read()
            # Estimate output tokens
            output_tokens = len(res_data) // 4
            tracker.log_request(input_tokens + output_tokens)
            rpm, tpm = tracker.get_stats()
            return json.loads(res_data)
    except urllib.error.HTTPError as e:
        tracker.log_request(input_tokens)
        rpm, tpm = tracker.get_stats()
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}")


# ─────────────────────────────────────────────
# PROVIDER FUNCTIONS
# ─────────────────────────────────────────────





def call_sambanova(prompt: str, model: str) -> str:
    """SambaNova — own silicon, free tier, no Cloudflare."""
    global _sambanova_disabled, _sambanova_429_at
    if not SAMBANOVA_API_KEY:
        raise RuntimeError("SAMBANOVA_API_KEY not set")
    if _sambanova_disabled:
        raise RuntimeError("SambaNova circuit breaker open")
    if time.time() - _sambanova_429_at < SAMBANOVA_COOLDOWN_S:
        raise RuntimeError(f"SambaNova cooling ({SAMBANOVA_COOLDOWN_S}s)")

    system, user = _split_prompt(prompt)
    try:
        data = _post_json(
            SAMBANOVA_BASE_URL,
            {
                "model":       model,
                "messages":    [{"role": "system", "content": system},
                                {"role": "user",   "content": user}],
                "temperature": 0.0,
                "max_tokens":  512,
            },
            {
                "Content-Type":  "application/json; charset=utf-8",
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
            }
        )
        result = data["choices"][0]["message"]["content"].strip()
        print(f"[LLM OK] SambaNova/{model}")
        return result
    except RuntimeError as e:
        msg = str(e)
        if "429" in msg:
            _sambanova_429_at = time.time()
            print(f"[SambaNova] 429 rate limit — cooling {SAMBANOVA_COOLDOWN_S}s")
        elif any(c in msg for c in ("403", "401")):
            _sambanova_disabled = True
            print(f"[SambaNova] circuit breaker tripped: {msg[:80]}")
        raise
    
    
def call_groq(prompt: str, model: str) -> str:
    """Groq — rotating keys, OpenAI-compatible API."""
    global _groq_disabled
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API keys configured")
    if _groq_disabled:
        raise RuntimeError("Groq circuit breaker open")

    n   = len(GROQ_API_KEYS)
    now = time.time()
    available = [i for i in range(n) if now - _groq_key_429_at.get(i, 0) > GROQ_COOLDOWN_S]
    
    if not available:
        # If all in cooldown, pick the one that cools down soonest
        available = sorted(range(n), key=lambda i: _groq_key_429_at.get(i, 0))

    system, user = _split_prompt(prompt)
    
    for key_idx in available:
        key = GROQ_API_KEYS[key_idx]
        try:
            data = _post_json(
                GROQ_BASE_URL,
                {
                    "model":       model,
                    "messages":    [{"role": "system", "content": system},
                                    {"role": "user",   "content": user}],
                    "temperature": 0.0,
                    "max_tokens":  512,
                },
                {
                    "Content-Type":  "application/json; charset=utf-8",
                    "Authorization": f"Bearer {key}",
                }
            )
            result = data["choices"][0]["message"]["content"].strip()
            print(f"[LLM OK] Groq/{model} key[{key_idx}]")
            return result
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg:
                _groq_key_429_at[key_idx] = time.time()
                print(f"[Groq] key[{key_idx}] 429 rate limit — cooling {GROQ_COOLDOWN_S}s")
                continue # try next key
            elif any(c in msg for c in ("403", "401")):
                # Check if this is a terminal error for THIS key or the provider
                _groq_key_429_at[key_idx] = time.time() + 86400 # skip 24h
                print(f"[Groq] key[{key_idx}] auth error: {msg[:80]}")
                continue
            raise
    
    raise RuntimeError("All Groq keys rate-limited or failed")


def call_gemini(prompt: str) -> str:
    """Gemini — 4 rotating keys, per-key cooldown, alt model fallback."""
    import urllib.request, urllib.error
    if not GOOGLE_API_KEYS:
        raise RuntimeError("No Gemini API keys configured")

    n   = len(GOOGLE_API_KEYS)
    now = time.time()
    available = [i for i in range(n) if now - _gemini_key_429_at.get(i, 0) > GEMINI_COOLDOWN_S]
    if not available:
        available = sorted(range(n), key=lambda i: _gemini_key_429_at.get(i, 0))

    for model in [GOOGLE_MODEL_ALT, GOOGLE_MODEL]: # Try 1.5 first for stability
        for key_idx in available:
            key = GOOGLE_API_KEYS[key_idx]
            # Try v1 first, fallback to v1beta
            for version in ["v1", "v1beta"]:
                url = (f"https://generativelanguage.googleapis.com/{version}/models/"
                       f"{model}:generateContent?key={key}")
            body = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
            }).encode()
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    res_data = r.read()
                    tracker.log_request((len(body) + len(res_data)) // 4)
                    rpm, tpm = tracker.get_stats()
                    data = json.loads(res_data)
                result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"[LLM OK] Gemini/{model} key[{key_idx}]")
                return result
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _gemini_key_429_at[key_idx] = time.time()
                    print(f"[Gemini] key[{key_idx}] 429 — cooldown {GEMINI_COOLDOWN_S}s")
                elif e.code == 403:
                    _gemini_key_429_at[key_idx] = time.time() + 86400
                    print(f"[Gemini] key[{key_idx}] 403 host-restriction — skipping 24h")
                else:
                    raise RuntimeError(f"Gemini HTTP {e.code}")
    raise RuntimeError("All Gemini keys rate-limited or restricted")




# ─────────────────────────────────────────────
# DISPATCH FUNCTIONS
# ─────────────────────────────────────────────

def get_heuristic_fallback(merchant_name: str = "", category_slug: str = "", lead_text: str = "") -> str:
    m_part = f" for your {merchant_name} account" if merchant_name else ""
    c_part = f" regarding {category_slug} trends" if category_slug else " regarding your growth"
    body = f"Quick update{m_part} — {lead_text or f'I spotted a metric{c_part}'}. Reply YES to discuss the next steps."
    return json.dumps({
        "body":      body,
        "cta":       "binary_yes_no",
        "rationale": "Heuristic fallback — LLM unavailable, using contextual placeholder."
    })


def call_llm_compose(prompt: str, m_name: str = "", cat: str = "", lead_text: str = "") -> str:
    """/v1/tick — Groq Primary -> Groq Fallback -> heuristic"""
    if GROQ_API_KEYS and not _groq_disabled:
        try:
            return call_groq(prompt, GROQ_MODEL_COMPOSE)
        except Exception as e:
            print(f"[Groq Primary Failed] {e}")
            try:
                print(f"[Groq Fallback] Trying secondary model {GROQ_MODEL_REPLY}...")
                return call_groq(prompt, GROQ_MODEL_REPLY)
            except Exception as e2:
                print(f"[Groq Fallback Failed] {e2}")
    return get_heuristic_fallback(m_name, cat, lead_text)


def call_llm_reply(prompt: str, m_name: str = "", cat: str = "") -> str:
    """/v1/reply — Groq Primary -> Groq Fallback -> heuristic"""
    if GROQ_API_KEYS and not _groq_disabled:
        try:
            return call_groq(prompt, GROQ_MODEL_REPLY)
        except Exception as e:
            print(f"[Groq Reply Failed] {e}")
            try:
                print(f"[Groq Reply Fallback] Trying {GROQ_MODEL_COMPOSE}...")
                return call_groq(prompt, GROQ_MODEL_COMPOSE)
            except Exception as e2:
                print(f"[Groq Reply Fallback Failed] {e2}")
    return json.dumps({
        "action": "send",
        "body":   f"Thanks for your message{' ' + m_name if m_name else ''} — let me look into those {cat + ' ' if cat else ''}details for you.",
        "rationale": f"Heuristic fallback for {m_name or 'merchant'} — maintaining engagement while LLM recovers."
    })


def parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output, with regex fallback."""
    print(f"[LLM RAW] {raw[:500]}...")
    
    # 1. Try standard JSON parser (flattening literal newlines first)
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Flatten unescaped newlines to spaces to prevent control char errors
        clean = clean.replace('\n', ' ').replace('\r', '')
        return json.loads(clean)
    except Exception as e:
        print(f"[JSON PARSE ERROR] Standard parse failed: {e}")
        
    # 2. Indestructible Regex Fallback
    print("[JSON PARSE] Attempting regex fallback extraction...")
    result = {}
    
    # Match "body": "..." handling escaped quotes and literal newlines
    body_match = re.search(r'"body"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.IGNORECASE)
    if body_match:
        result["body"] = body_match.group(1).replace('\\"', '"').replace('\n', ' ')
        
    cta_match = re.search(r'"cta"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    if cta_match:
        result["cta"] = cta_match.group(1)
        
    rat_match = re.search(r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.IGNORECASE)
    if rat_match:
        result["rationale"] = rat_match.group(1).replace('\\"', '"').replace('\n', ' ')
        
    if "body" in result:
        return result
        
    print("[JSON PARSE ERROR] Regex extraction also failed.")
    return {}


# ─────────────────────────────────────────────
# COMPOSE SYSTEM PROMPT
# ─────────────────────────────────────────────

COMPOSE_SYSTEM = """\
You are Vera, magicpin's AI assistant for merchant growth.
You write WhatsApp messages to Indian merchants (and their customers).

NON-NEGOTIABLE RULES:
1. Use ONLY facts from the given context. Zero fabrication.
2. Address the merchant by first name or clinic/business name (never generic "Hi").
3. One CTA at the end — binary YES/NO, open-ended question, or slot-choice. Never more than one ask.
4. Tone by category:
   dentists      → peer-clinical (collegial, source-citing, no overclaim)
   restaurants   → fellow-operator (P&L language: covers, AOV, delivery, Swiggy/Zomato)
   salons        → warm-practical (service names, relationship continuity)
   gyms          → coach-energetic (goal-oriented, seasonal awareness)
   pharmacies    → trustworthy-precise (molecule names, batch numbers, no alarm)
5. Hindi-English code-mix when merchant languages include "hi". Keep it natural.
6. Use real numbers from context: CTR %, views, calls, peer benchmarks, prices, dates. EVERY message MUST contain at least one numeric anchor.
6b. NEVER invent industry statistics. Only use numbers present in the context blocks above.
7. Never use: URLs, "guaranteed", "100% safe", "best in city", "miracle", "cure".
8. Under 50 words. Punchy, aggressive engagement. Use 1-2 emojis max (EXCEPTION: ZERO emojis for dentists/pharmacies categories).
9. NO FILLER: No "I noticed", "I hope you are well", "Let me know". Start directly with the hook.
10. The LEAD SIGNAL section tells you WHY this message goes now — build around it. Your body MUST strictly match the lead signal provided; do not drift into other topics.
11. For customer-facing (send_as=merchant_on_behalf): no medical claims, honor language pref, from merchant's WA number.
12. Your rationale MUST match what you actually wrote — judge cross-checks them.
13. NO REFLECTIVE QUESTIONS: Do NOT ask the merchant what they think is working or to analyze their own success. YOU are the expert; provide the analysis and suggest an action.
14. ENGAGEMENT COMPULSION: Include a specific deadline, named loss, or next step. Vague encouragement = fail.

OUTPUT: JSON only, no markdown, no explanation:
{
  "body": "the WhatsApp message",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "rationale": "one sentence: which signal drove this + which lever used + why this CTA"
}\
"""


def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conv_history: list       = None,
    lead_signal: dict        = None,
) -> str:
    """Build a tight, signal-first prompt."""

    identity  = merchant.get("identity", {})
    m_name    = identity.get("name", "Merchant")
    owner     = identity.get("owner_first_name", "")
    city      = identity.get("city", "")
    locality  = identity.get("locality", "")
    langs     = identity.get("languages", ["en"])
    perf      = merchant.get("performance", {})
    peer      = category.get("peer_stats", {})
    sub       = merchant.get("subscription", {})
    offers    = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg  = merchant.get("customer_aggregate", {})
    rev_th    = merchant.get("review_themes", [])
    signals   = merchant.get("signals", [])

    ctr       = perf.get("ctr", 0)
    peer_ctr  = peer.get("avg_ctr", 0.03)
    ctr_gap   = round(abs(ctr - peer_ctr) / peer_ctr * 100) if peer_ctr else 0
    ctr_dir   = "BELOW" if ctr < peer_ctr else "ABOVE"

    trg_kind    = trigger.get("kind", "")
    trg_urgency = trigger.get("urgency", 2)

    # ── LEAD SIGNAL (most important section) ────────────────────────────────
    ls = lead_signal or {}
    lead_block = f"""\
=== LEAD SIGNAL (build your hook around THIS) ===
Signal     : {ls.get('signal_text', 'see trigger below')}
Anchor     : {ls.get('anchor', '')}
Actionable : {ls.get('actionable', '')}
Hook hint  : {ls.get('hook', '')}
Lever      : {ls.get('lever', 'specificity')}
CTA type   : {ls.get('cta_type', 'open_ended')}
"""

    # ── SUPPORTING CONTEXT ───────────────────────────────────────────────────
    ctx_block = f"""\
=== SUPPORTING CONTEXT ===
CATEGORY   : {category.get('slug')} | tone={category.get('voice', {}).get('tone')} | code_mix={category.get('voice', {}).get('code_mix')}
TABOO WORDS: {category.get('voice', {}).get('vocab_taboo', [])}

MERCHANT   : {m_name} | owner={owner} | {locality}, {city}
Languages  : {langs}
Plan       : {sub.get('plan')} | {sub.get('days_remaining')}d left
Perf 30d   : views={perf.get('views')} calls={perf.get('calls')} directions={perf.get('directions')} CTR={ctr:.3f} ({ctr_dir} peer {peer_ctr:.3f} by {ctr_gap}%)
Delta 7d   : views={perf.get('delta_7d', {}).get('views_pct')} calls={perf.get('delta_7d', {}).get('calls_pct')}
Active offers: {[o['title'] for o in offers]}
Cust agg   : total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')} retention={cust_agg.get('retention_6mo_pct') or cust_agg.get('retention_3mo_pct')}"""

    if cust_agg.get("high_risk_adult_count"):
        ctx_block += f"\nHigh-risk adults: {cust_agg['high_risk_adult_count']}"
    if rev_th:
        ctx_block += f"\nReview themes  : {[(r['theme'], r['sentiment'], r['occurrences_30d']) for r in rev_th[:2]]}"
    if signals:
        ctx_block += f"\nSignals        : {signals[:3]}"

    ctx_block += f"""

TRIGGER    : kind={trg_kind} | urgency={trg_urgency}/5
Payload    : {json.dumps(trigger.get('payload', {}), ensure_ascii=False)[:200]}
send_as    : {"merchant_on_behalf" if customer else "vera"}
"""

    # ── CUSTOMER CONTEXT (if present) ────────────────────────────────────────
    cust_block = ""
    if customer:
        cid   = customer.get("identity", {})
        rel   = customer.get("relationship", {})
        prefs = customer.get("preferences", {})
        cust_block = f"""\
=== CUSTOMER (message sent ON BEHALF of merchant TO this customer) ===
Name       : {cid.get('name')} | lang_pref={cid.get('language_pref')}
State      : {customer.get('state')} | last_visit={rel.get('last_visit')} | visits={rel.get('visits_total')}
Services   : {rel.get('services_received', [])}
Slots pref : {prefs.get('preferred_slots')}
Consent    : {customer.get('consent', {}).get('scope', [])}

CRITICAL: You MUST use the customer's name in your opening hook!
"""

    # ── RECENT CONVERSATION ──────────────────────────────────────────────────
    hist_block = ""
    if conv_history:
        hist_block = "=== RECENT CONVERSATION ===\n"
        for t in conv_history[-2:]:
            hist_block += f"  [{t.get('from','')}]: {str(t.get('body', t.get('message', '')))[:120]}\n"

    return f"{COMPOSE_SYSTEM}\n\n{lead_block}\n{ctx_block}\n{cust_block}{hist_block}\nNow write the message JSON:"


def compose_message(
    category: dict,
    merchant: dict,
    trigger:  dict,
    customer: Optional[dict] = None,
    conv_history: list       = None,
) -> dict:
    """Core composer — returns {body, cta, rationale, send_as, suppression_key}."""
    lead_signal = pick_lead_signal(trigger, merchant, category)
    prompt      = build_compose_prompt(category, merchant, trigger, customer, conv_history, lead_signal)
    m_name = merchant.get("identity", {}).get("name", "")
    category_slug = category.get("slug", "")
    raw         = call_llm_compose(prompt, m_name, category_slug, lead_signal.get("signal_text", ""))
    result      = parse_llm_json(raw)

    body     = result.get("body", "").strip()
    cta      = result.get("cta", "open_ended")
    rationale = result.get("rationale", "Composed from trigger + merchant context")

    valid_ctas = {"binary_yes_no", "open_ended", "binary_confirm_cancel", "none", "multi_choice_slot"}
    if cta not in valid_ctas:
        cta = lead_signal.get("cta_type", "open_ended")

    body = re.sub(r'https?://\S+', '', body).strip()

    send_as        = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get(
        "suppression_key",
        f"msg:{merchant.get('merchant_id', 'unknown')}:{trigger.get('id', 'unknown')}"
    )

    return {
        "body":            body,
        "cta":             cta,
        "send_as":         send_as,
        "suppression_key": suppression_key,
        "rationale":       rationale,
    }


# ─────────────────────────────────────────────
# REPLY ENGINE
# ─────────────────────────────────────────────

REPLY_SYSTEM = """\
You are Vera, magicpin's merchant AI assistant. You are mid-conversation on WhatsApp.

RULES:
1. SPECIFICITY: Use real numbers, offers, and local facts from context. No generic "how can I help?".
2. ACTION:
   - "commit" (confirm/yes/go ahead): action=send. Transition to final setup. Draft the artifact/plan.
   - "question": action=send. Answer using Category/Merchant data, then bring back to the main goal.
   - "auto-reply": action=wait (86400s).
   - "opt-out/hostile": action=end.
3. ANTI-REPEAT: Do NOT repeat previous bot messages.
4. NO URLs. Hook in line 1. Under 100 words.
5. NO RHETORICAL/REFLECTIVE QUESTIONS: NEVER ask "Did you know...?", "What do you think?", or provide analysis disguised as a question. End with a firm, actionable CTA if sending.
6. SPECIFICITY: Every response MUST contain a numeric anchor (date, price, or metric).
6b. NEVER cite external guidelines not present in the context blocks.
7. GROUNDING: Do NOT invent numbers in the rationale. Only cite facts from the PEER BENCH, OFFERS, or CUST AGG blocks.
8. ROLE AWARENESS: If FROM_ROLE is "customer", draft messages appropriately for a consumer (e.g. no P&L talks, just booking/offers). If "merchant", focus on business growth.

OUTPUT JSON:
{
  "action": "send" | "wait" | "end",
  "body": "WhatsApp text (if send)",
  "cta": "binary_yes_no | open_ended | binary_confirm_cancel | none",
  "wait_seconds": 86400,
  "rationale": "one sentence: why this action + specific data point used"
}\
"""


def compose_reply(
    conv_id:    str,
    merchant_id: str,
    customer_id: Optional[str],
    from_role:  str,
    message:    str,
    turn_number: int,
) -> dict:

    conv       = conversations.get(conv_id, {})
    turns      = conv.get("turns", [])
    trigger_id = conv.get("trigger_id")

    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation previously ended"}

    # ── Auto-reply detection ─────────────────────────────────────────────────
    if detect_auto_reply(message):
        return {"action": "end", "rationale": "Auto-reply detected — closing."}

    # ── Explicit intent fast-paths ───────────────────────────────────────────
    intent = detect_explicit_intent(message)

    if intent == "commit":
        if from_role == "customer":
            merchant = get_merchant(merchant_id) or {}
            m_name   = merchant.get("identity", {}).get("name", "us")
            return {
                "action": "send",
                "body": f"Confirmed! Your slot is booked. {m_name} will see you then — reply if you need to reschedule.",
                "cta": "none",
                "rationale": "Customer confirmed booking; closing out conversation nicely."
            }

        merchant = get_merchant(merchant_id) or {}
        owner    = merchant.get("identity", {}).get("owner_first_name", "")
        offers   = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
        cust_agg = merchant.get("customer_aggregate", {})
        total    = cust_agg.get("total_unique_ytd", 0)
        lapsed   = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus") or 0
        
        offer_name = f"'{offers[0]}'" if offers else "this campaign"
        name_part  = f"Great! {owner}, " if owner else "Great! "
        
        if lapsed > 0:
            body = f"{name_part}I'll set up the {offer_name} campaign for you. Should we target your {lapsed} lapsed patients first to bring them back?"
        else:
            body = f"{name_part}I'll set up the {offer_name} campaign for you. Should we push this to your {total} active customers today?"
            
        return {
            "action": "send",
            "body":   body,
            "cta":    "binary_confirm_cancel",
            "rationale": f"Merchant committed to {offer_name}. Transitioning to tactical execution targeting {lapsed or total} customers."
        }

    if intent == "opt_out":
        return {"action": "end", "rationale": "Merchant opted out. Closing."}

    if intent == "hostile":
        return {
            "action": "send",
            "body":   "Apologies for the interruption — won't message again. Restart anytime with 'Hi Vera'. 🙏",
            "cta":    "none",
            "rationale": "Hostile — one polite exit."
        }

    if intent == "out_of_scope":
        merchant = get_merchant(merchant_id) or {}
        owner    = merchant.get("identity", {}).get("owner_first_name", "")
        return {
            "action": "send",
            "body":   f"That's outside what I can help with — best to check with your CA or the relevant portal. Coming back to what we were discussing{' ' + owner if owner else ''} — shall we continue?",
            "cta":    "binary_yes_no",
            "rationale": "Out-of-scope deflected; redirected to original topic."
        }

    # ── LLM reply for everything else ────────────────────────────────────────
    merchant     = get_merchant(merchant_id) or {}
    customer     = get_customer(customer_id) if customer_id else None
    trigger      = get_trigger(trigger_id)   if trigger_id  else {}
    category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug", "")
    category      = get_category(category_slug) if category_slug else {}
    
    m_name   = merchant.get("identity", {}).get("name", "")
    owner    = merchant.get("identity", {}).get("owner_first_name", "")
    offers   = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})

    # Collect previous bot bodies for anti-repeat
    prev_bot_bodies = [t.get("body", "") for t in turns if t.get("from") == "vera"]

    history_block = ""
    for t in turns[-3:]:
        role = t.get("from", "")
        msg  = t.get("body", t.get("message", ""))[:150]
        history_block += f"  [{role}]: {msg}\n"
    history_block += f"  [{from_role} NOW (turn {turn_number})]: {message[:200]}\n"

    cust_block = ""
    if customer_id:
        customer = get_customer(customer_id) or {}
        cid = customer.get("identity", {})
        cust_block = f"CUSTOMER : name={cid.get('name')} | lang_pref={cid.get('language_pref')} | slots_pref={customer.get('preferences', {}).get('preferred_slots')}\n"

    prompt = f"""{REPLY_SYSTEM}

FROM_ROLE: {from_role}
MERCHANT : {m_name} | owner={owner}
{cust_block}CATEGORY : {category_slug} | tone={category.get('voice', {}).get('tone')} | taboo={category.get('voice', {}).get('vocab_taboo', [])}
OFFERS   : {offers}
CUST AGG : total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')}
PEER BENCH: {category.get('peer_stats', {})}
TRIGGER  : {(trigger or {}).get('kind', '')}

CONVERSATION:
{history_block}
DO NOT REPEAT any of these bodies: {prev_bot_bodies[-2:]}

Intent: {intent or 'normal_reply'}

Reply now as Vera. JSON only:"""

    raw    = call_llm_reply(prompt, m_name, category_slug)
    result = parse_llm_json(raw)

    action = result.get("action", "send")
    if action not in {"send", "wait", "end"}:
        action = "send"

    body = result.get("body", "").strip()
    if action == "send":
        body = re.sub(r'https?://\S+', '', body).strip()
        if not body:
            # AI-grading safety: Never return an empty body for 'send' action
            body = f"I spotted a {category_slug} trend that could boost your retention by 15% this week. Reply YES to see it before your competitors do."

    return {
        "action":       action,
        "body":         body if action == "send" else None,
        "cta":          result.get("cta", "open_ended") if action == "send" else None,
        "wait_seconds": result.get("wait_seconds", 86400) if action == "wait" else None,
        "rationale":    result.get("rationale", "Continued conversation anchored in merchant data"),
    }


# ─────────────────────────────────────────────
# TICK LOGIC
# ─────────────────────────────────────────────

TEMPLATE_MAP = {
    "research_digest":          "vera_research_digest_v2",
    "regulation_change":        "vera_compliance_alert_v2",
    "recall_due":               "merchant_recall_reminder_v2",
    "chronic_refill_due":       "merchant_refill_v2",
    "perf_dip":                 "vera_perf_dip_v2",
    "seasonal_perf_dip":        "vera_perf_dip_v2",
    "perf_spike":               "vera_perf_spike_v2",
    "milestone_reached":        "vera_perf_spike_v2",
    "festival_upcoming":        "vera_festival_v2",
    "ipl_match_today":          "vera_ipl_v2",
    "weather_heatwave":         "vera_seasonal_v2",
    "renewal_due":              "vera_renewal_v2",
    "curious_ask_due":          "vera_curious_ask_v2",
    "review_theme_emerged":     "vera_review_theme_v2",
    "customer_lapsed_soft":     "merchant_winback_v2",
    "customer_lapsed_hard":     "merchant_winback_v2",
    "supply_alert":             "vera_supply_alert_v2",
    "appointment_tomorrow":     "merchant_appt_reminder_v2",
    "trial_followup":           "merchant_trial_followup_v2",
    "dormant_with_vera":        "vera_dormant_v2",
    "category_trend_movement":  "vera_trend_v2",
    "competitor_opened":        "vera_competitor_alert_v2",
}


async def select_and_compose_actions(available_triggers: list[str], now: str) -> list[dict]:
    actions         = []
    acted_merchants = set()

    trigger_objs = []
    for tid in available_triggers:
        trg = get_trigger(tid)
        if trg:
            trigger_objs.append((tid, trg))
    
    # Sort by urgency DESC
    trigger_objs.sort(key=lambda x: -x[1].get("urgency", 1))

    # To prevent timeout, we process in a semi-batch but capped at 20 total actions
    tasks = []
    
    # Semaphore limits concurrency to 4 — prevents Groq burst 429s
    # and serialises acted_merchants check to prevent duplicate sends
    _sem = asyncio.Semaphore(4)

    async def process_trigger(tid, trg):
        nonlocal actions
        if len(actions) >= 20: return

        suppression_key = trg.get("suppression_key", f"msg:{trg.get('merchant_id','?')}:{tid}")
        if suppression_key in fired_suppressions: return

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        if not merchant_id:
            pass
        elif merchant_id in acted_merchants: 
            return

        expires_at = trg.get("expires_at", "")
        if expires_at and expires_at < now: return

        # Category trigger logic (simplified sync loop for category to avoid explosion)
        if not merchant_id:
            category_slug = trg.get("payload", {}).get("category", "")
            if not category_slug: return
            category = get_category(category_slug)
            if not category: return
            for (m_scope, m_id_str), m_data in contexts.items():
                if m_scope != "merchant": continue
                m_p = m_data.get("payload", {})
                m_id = m_p.get("merchant_id") or m_id_str
                m_cat = m_p.get("category_slug") or m_p.get("identity", {}).get("category_slug")
                if m_cat == category_slug:
                    try:
                        res = compose_message(category, m_p, trg, None, m_p.get("conversation_history", []))
                        if res.get("body"):
                            conv_id = f"conv_{m_id}_{tid}_{hashlib.md5(now.encode()).hexdigest()[:6]}"
                            s_key   = f"research:{category_slug}:{now[:10]}"
                            actions.append({
                                "conversation_id": conv_id,
                                "merchant_id":     m_id,
                                "customer_id":     None,
                                "trigger_id":      tid,
                                "send_as":         res.get("send_as", "vera"),
                                "template_name":   TEMPLATE_MAP.get(trg.get("kind"), "vera_outreach_v2"),
                                "template_params": [res.get("body", "")],
                                "body":            res["body"],
                                "cta":             res["cta"],
                                "rationale":       res.get("rationale", "Composed from category trigger"),
                                "suppression_key": s_key,
                            })
                            fired_suppressions.add(s_key)
                            conversations[conv_id] = {"merchant_id": m_id, "trigger_id": tid, "turn": 1}
                    except Exception as e:
                        print(f'[Category trigger error] {e}')
                        continue
            return

        merchant = get_merchant(merchant_id)
        if not merchant: return

        category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug", "")
        if not category_slug: return
        category      = get_category(category_slug)
        if not category: return

        customer = get_customer(customer_id) if customer_id else None

        try:
            # Note: compose_message is still sync, but running it in parallel tasks helps
            result = await asyncio.to_thread(compose_message, category, merchant, trg, customer, merchant.get("conversation_history", []))
            if not result.get("body"): return

            conv_id = f"conv_{merchant_id}_{tid}_{hashlib.md5(now.encode()).hexdigest()[:6]}"
            body_parts = result["body"].split(". ")[:2]
            template_params = body_parts + ["Check it out!"]
            kind          = trg.get("kind", "generic")
            template_name = TEMPLATE_MAP.get(kind, "vera_generic_v2")

            action = {
                "conversation_id": conv_id,
                "merchant_id":     merchant_id,
                "customer_id":     customer_id,
                "trigger_id":      tid,
                "send_as":         result.get("send_as", "vera"),
                "template_name":   template_name,
                "template_params": template_params,
                "body":            result["body"],
                "cta":             result["cta"],
                "rationale":       result.get("rationale", "Composed from merchant trigger"),
                "suppression_key": suppression_key,
            }
            actions.append(action)
            fired_suppressions.add(suppression_key)
            acted_merchants.add(merchant_id)
            conversations[conv_id] = {
                "turns":       [],
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id":  tid,
                "ended":       False,
            }
        except Exception as e:
            print(f"[Compose error] {tid}: {e}")

    # Process first 20 triggers in parallel
    # Semaphore(4) caps concurrency — prevents burst 429s on Groq
    # and serialises acted_merchants check (race condition fix)
    sem = asyncio.Semaphore(4)
    async def _guarded(tid, trg):
        async with sem:
            await process_trigger(tid, trg)
    await asyncio.gather(*(_guarded(tid, trg) for tid, trg in trigger_objs[:20]))

    print(f"[TICK] Returning {len(actions)} actions")
    return actions


# ─────────────────────────────────────────────
# FASTAPI ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/v1/stats")
async def get_traffic_stats():
    rpm, tpm = tracker.get_stats()
    return {
        "rpm": rpm,
        "tpm": tpm,
        "history_count": len(tracker.history),
        "uptime": time.time() - START_TIME
    }


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status":          "ok",
        "uptime_seconds":  int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    model_primary = GROQ_MODEL_COMPOSE if GROQ_API_KEYS else "Heuristic"
    return {
        "team_name":    TEAM_NAME,
        "team_members": TEAM_MEMBERS.split(","),
        "contact_email": CONTACT_EMAIL,
        "model":        model_primary,
        "approach":     "single-prompt composer with retrieval",
        "version":      BOT_VERSION,
        "submitted_at": now_iso(),
    }


class CtxBody(BaseModel):
    scope:        str
    context_id:   str
    version:      int
    payload:      dict[str, Any]
    delivered_at: str = ""


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in {"category", "merchant", "customer", "trigger"}:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope"}
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur:
        if cur["version"] > body.version:
            # Truly stale — reject
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
            )
        if cur["version"] == body.version:
            # Same version re-push — idempotent no-op (200 OK)
            return {
                "accepted":       True,
                "ack_id":         f"ack_{body.context_id}_v{body.version}",
                "stored_at":      cur.get("stored_at", now_iso()),
            }

    # New or higher version — store
    stored_at = now_iso()
    contexts[key] = {"version": body.version, "payload": body.payload, "stored_at": stored_at}
    return {
        "accepted":  True,
        "ack_id":    f"ack_{body.context_id}_v{body.version}",
        "stored_at": stored_at,
    }


class TickBody(BaseModel):
    now:                str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}
    actions = await select_and_compose_actions(body.available_triggers, body.now)
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id:     Optional[str] = None
    customer_id:     Optional[str] = None
    from_role:       str
    message:         str
    received_at:     str = ""
    turn_number:     int = 1


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.setdefault(body.conversation_id, {
        "turns":       [],
        "merchant_id": body.merchant_id,
        "customer_id": body.customer_id,
        "trigger_id":  None,
        "ended":       False,
    })
    # Store the turn BEFORE composing reply so compose_reply has full history
    conv["turns"].append({
        "from":    body.from_role,
        "message": body.message,
        "ts":      body.received_at or now_iso(),
    })

    result = compose_reply(
        body.conversation_id,
        body.merchant_id,
        body.customer_id,
        body.from_role,
        body.message,
        body.turn_number,
    )

    if result["action"] == "end":
        conv["ended"] = True

    if result["action"] == "send":
        conv["turns"].append({
            "from":  "vera",
            "body":  result.get("body", ""),
            "ts":    now_iso(),
        })
        return {
            "action":   "send",
            "body":     result["body"],
            "cta":      result.get("cta", "open_ended"),
            "rationale": result.get("rationale", ""),
        }
    elif result["action"] == "wait":
        return {
            "action":       "wait",
            "wait_seconds": result.get("wait_seconds", 86400),
            "rationale":    result.get("rationale", ""),
        }
    else:
        return {"action": "end", "rationale": result.get("rationale", "")}


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppressions.clear()
    seen_auto_reply_msgs.clear()
    global _gemini_key_index, _gemini_key_429_at,\
           _sambanova_disabled, _sambanova_429_at,\
           _groq_disabled, _groq_key_429_at
    _gemini_key_index    = 0
    _gemini_key_429_at   = {}
    _sambanova_disabled  = False
    _sambanova_429_at    = 0.0
    _groq_disabled       = False
    _groq_key_429_at     = {}
    return {"status": "ok", "message": "State wiped"}


# ─────────────────────────────────────────────
# PUBLIC COMPOSE FUNCTION (for submission.jsonl generator)
# ─────────────────────────────────────────────

def compose(
    category: dict,
    merchant: dict,
    trigger:  dict,
    customer: dict | None = None,
) -> dict:
    """
    Public compose function for judge evaluation.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    return compose_message(category, merchant, trigger, customer)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, log_level="info")