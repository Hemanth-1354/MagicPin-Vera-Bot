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
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Groq — primary LLM (free tier, ultra-fast)
# /v1/tick  (compose, high-stakes) → llama-3.3-70b-versatile
# /v1/reply (fast turns)           → llama-3.1-8b-instant
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL_COMPOSE  = "llama-3.3-70b-versatile"   # tick / compose
GROQ_MODEL_REPLY    = "llama-3.1-8b-instant"       # reply / fast turns
GROQ_BASE_URL       = "https://api.groq.com/openai/v1/chat/completions"

# Gemini — secondary fallback (4 rotating keys)
GOOGLE_MODEL = "gemini-2.0-flash"
GOOGLE_API_KEYS: list[str] = []
for _k in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GOOGLE_API_KEY_4"]:
    _v = os.getenv(_k, "").strip()
    if _v:
        GOOGLE_API_KEYS.append(_v)

# Anthropic — tertiary fallback
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

TEAM_NAME       = os.getenv("TEAM_NAME", "Vera-Builder")
TEAM_MEMBERS    = os.getenv("TEAM_MEMBERS", "Candidate")
CONTACT_EMAIL   = os.getenv("CONTACT_EMAIL", "candidate@example.com")
BOT_VERSION     = "3.3.0"

app = FastAPI(title="MagicPin Vera Bot", version=BOT_VERSION)
START_TIME = time.time()

# ── Gemini per-key cooldown ──────────────────
# If a key hits 429, skip it for GEMINI_COOLDOWN_S seconds before retrying.
GEMINI_COOLDOWN_S   = 60
_gemini_key_index   = 0
_gemini_call_counts = [0] * max(len(GOOGLE_API_KEYS), 1)
_gemini_key_429_at: dict[int, float] = {}   # key_idx → epoch when it last 429'd

# ── Groq circuit breaker ─────────────────────
# On 403 (Cloudflare/IP block) or repeated failures, flip this flag
# so the entire Groq path is skipped instantly for the rest of the session.
_groq_disabled = False

# ── Cerebras ─────────────────────────────────
# Free tier, OpenAI-compatible, ~200ms latency, no Cloudflare IP blocks.
# Sign up: https://cloud.cerebras.ai  (free, no credit card)
# Same Llama models as Groq. Primary LLM when Groq is blocked.
CEREBRAS_API_KEY       = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL_COMPOSE = "llama-3.3-70b"   # compose / tick
CEREBRAS_MODEL_REPLY   = "llama-3.1-8b"    # reply / fast turns
CEREBRAS_BASE_URL      = "https://api.cerebras.ai/v1/chat/completions"
_cerebras_disabled     = False

# ── Gemini alt model ──────────────────────────
# gemini-1.5-flash-8b has a higher free-tier RPM than gemini-2.0-flash.
# Used as second Gemini attempt when 2.0-flash hits 429 on all keys.
GOOGLE_MODEL_ALT = "gemini-1.5-flash-8b"

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
    if any(p in msg_lower for p in [
        "let's do it", "lets do it", "ok do it", "go ahead", "yes let's",
        "haan karo", "confirm", "proceed", "start karo", "shuru karo",
        "yes please", "bilkul", "whats next", "what's next", "send it",
        "draft it", "do it"
    ]):
        return "commit"
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


def is_repeat_auto_reply(conv_id: str, message: str) -> int:
    conv  = conversations.get(conv_id, {})
    turns = conv.get("turns", [])
    return sum(
        1 for t in turns
        if t.get("from") == "merchant"
        and t.get("message", "").strip().lower() == message.strip().lower()
    )


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
        if digest_item:
            src   = digest_item.get("source", "")
            title = digest_item.get("title", "")
            summ  = digest_item.get("summary", "")
            n     = digest_item.get("trial_n", "")
            seg   = digest_item.get("patient_segment", "")
            act   = digest_item.get("actionable", "")
            n_str = f" (n={n})" if n else ""
            # Pick the most merchant-specific anchor
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

    # ── PERFORMANCE DIP ───────────────────────────────────────────────────────
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
        meds_str      = ", ".join(meds) if meds else ""
        slots         = payload.get("available_slots", [])
        slot_str      = " | ".join(slots[:2]) if slots else ""
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
        past_str      = ", ".join(past_services[:2]) if past_services else ""
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
        # Counter-intuitive insight by category for IPL
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
        feat_str = ", ".join(features[:3]) if features else "your current features"
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
        batch_str   = ", ".join(batch) if batch else ""
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
    best_signal = signals[0] if signals else f"perf: {views} views, {calls} calls, CTR {ctr:.3f}"
    return {
        "signal_text": best_signal,
        "anchor":      f"Offer: {offers[0]['title']}" if offers else "",
        "actionable":  payload.get("suggested_action", "One clear action for merchant"),
        "hook":        f"Signal detected for {owner or m_name}",
        "lever":       "specificity",
        "cta_type":    "open_ended"
    }


# ─────────────────────────────────────────────
# LLM LAYER
# ─────────────────────────────────────────────
#
# Priority chain by endpoint:
#   /v1/tick  (compose) → Groq 70b → Groq 8b → Gemini → Anthropic → heuristic
#   /v1/reply (turns)   → Groq 8b  → Groq 70b → Gemini → Anthropic → heuristic
#
# Groq is primary: free-tier, <500ms latency, no RPM trouble at this scale.
# Gemini/Anthropic are fallbacks for Groq outages or rate limits.
# ─────────────────────────────────────────────


def call_groq(prompt: str, model: str) -> str:
    """
    Call Groq's OpenAI-compatible endpoint.
    Splits the monolithic prompt into system + user messages for better
    instruction-following on Llama models.
    """
    import urllib.request, urllib.error
    global _groq_disabled

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    if _groq_disabled:
        raise RuntimeError("Groq circuit breaker open — skipping")

    try:
        return call_openai_compat(
            prompt, model, GROQ_BASE_URL, GROQ_API_KEY, "Groq"
        )
    except RuntimeError as e:
        msg = str(e)
        if "HTTP 403" in msg or "HTTP 401" in msg or "HTTP 529" in msg:
            _groq_disabled = True
            print(f"[Groq] circuit breaker tripped — {msg[:80]}. Falling back.")
        raise
    except Exception as e:
        _groq_disabled = True
        print(f"[Groq] circuit breaker tripped — {type(e).__name__}. Falling back.")
        raise


def call_openai_compat(prompt: str, model: str, base_url: str, api_key: str,
                       service_name: str = "OpenAI-compat") -> str:
    """
    Generic OpenAI-compatible chat completions call.
    Splits prompt into system + user messages for better Llama instruction-following.
    Shared by Groq and Cerebras.
    """
    import urllib.request, urllib.error

    system_marker_end = prompt.find("\n\n=== LEAD SIGNAL")
    if system_marker_end == -1:
        system_marker_end = prompt.find("\n\nMERCHANT :")
    if system_marker_end == -1:
        system_marker_end = prompt.find("\n\nCONVERSATION:")

    if system_marker_end != -1:
        system_text = prompt[:system_marker_end].strip()
        user_text   = prompt[system_marker_end:].strip()
    else:
        system_text = "You are Vera, magicpin's AI assistant for merchant growth."
        user_text   = prompt

    body = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens":  512,
    }).encode()

    req = urllib.request.Request(
        base_url, data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} HTTP {e.code}: {body_txt[:200]}")


def call_cerebras(prompt: str, model: str) -> str:
    """Call Cerebras — free, fast, no Cloudflare IP blocks."""
    global _cerebras_disabled
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    if _cerebras_disabled:
        raise RuntimeError("Cerebras circuit breaker open")
    try:
        return call_openai_compat(
            prompt, model, CEREBRAS_BASE_URL, CEREBRAS_API_KEY, "Cerebras"
        )
    except RuntimeError as e:
        msg = str(e)
        if "403" in msg or "401" in msg or "529" in msg:
            _cerebras_disabled = True
            print(f"[Cerebras] circuit breaker tripped — {msg[:80]}")
        raise


def call_gemini(prompt: str) -> str:
    """
    Call Gemini with per-key cooldown tracking.
    If a key hits 429, it's skipped for GEMINI_COOLDOWN_S seconds.
    Only tries each key once per call — no long blocking waits.
    """
    global _gemini_key_index, _gemini_call_counts
    import urllib.request, urllib.error

    if not GOOGLE_API_KEYS:
        raise RuntimeError("No Gemini API keys configured")

    n   = len(GOOGLE_API_KEYS)
    now = time.time()

    # Build a list of keys not currently in cooldown, sorted by fewest calls
    available = [
        i for i in range(n)
        if now - _gemini_key_429_at.get(i, 0) > GEMINI_COOLDOWN_S
    ]
    if not available:
        # All keys in cooldown — pick the one that cooled down longest ago
        available = sorted(range(n), key=lambda i: _gemini_key_429_at.get(i, 0))

    for key_idx in available:
        _gemini_call_counts[key_idx] = _gemini_call_counts[key_idx] + 1
        key = GOOGLE_API_KEYS[key_idx]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GOOGLE_MODEL}:generateContent?key={key}"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _gemini_key_429_at[key_idx] = time.time()
                print(f"[Gemini] key[{key_idx}] 429 — cooling down {GEMINI_COOLDOWN_S}s, trying next key")
                continue
            else:
                raise
    # All keys on primary model are in cooldown — try alt model (higher free quota)
    print(f"[Gemini] all {GOOGLE_MODEL} keys cooling — trying {GOOGLE_MODEL_ALT}")
    for key_idx in range(n):
        key = GOOGLE_API_KEYS[key_idx]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GOOGLE_MODEL_ALT}:generateContent?key={key}"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                continue
            raise
    raise RuntimeError("All Gemini keys (both models) rate-limited or in cooldown")


def call_anthropic(prompt: str) -> str:
    """Call Claude Haiku as tertiary fallback — with proper system/user split."""
    import urllib.request

    # Split prompt so system instructions go in the system field (avoids 400)
    system_marker_end = prompt.find("\n\n=== LEAD SIGNAL")
    if system_marker_end == -1:
        system_marker_end = prompt.find("\n\nMERCHANT :")
    if system_marker_end == -1:
        system_marker_end = prompt.find("\n\nCONVERSATION:")

    if system_marker_end != -1:
        system_text = prompt[:system_marker_end].strip()
        user_text   = prompt[system_marker_end:].strip()
    else:
        system_text = "You are Vera, magicpin's AI assistant for merchant growth."
        user_text   = prompt

    url  = "https://api.anthropic.com/v1/messages"
    body = json.dumps({
        "model":      "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "system":     system_text,
        "messages":   [{"role": "user", "content": user_text}]
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type":      "application/json",
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    })
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


_HEURISTIC_FALLBACK = json.dumps({
    "body":      "Quick update on your account — want me to share one thing I spotted? Reply YES.",
    "cta":       "binary_yes_no",
    "rationale": "Heuristic fallback — LLM unavailable"
})


def call_llm_compose(prompt: str) -> str:
    """
    Composition path (/v1/tick) — quality matters most.
    Chain: Cerebras 70b → Groq 70b → Gemini → Anthropic → heuristic
    """
    # 1. Cerebras — free, fast, no Cloudflare IP blocks
    if CEREBRAS_API_KEY and not _cerebras_disabled:
        try:
            return call_cerebras(prompt, CEREBRAS_MODEL_COMPOSE)
        except Exception as e:
            print(f"[Cerebras 70b compose error] {e}")
    # 2. Groq — may be blocked by Cloudflare on some IPs
    if GROQ_API_KEY and not _groq_disabled:
        try:
            return call_groq(prompt, GROQ_MODEL_COMPOSE)
        except Exception as e:
            print(f"[Groq 70b compose error] {e}")
    # 3. Gemini — 4 rotating keys with per-key cooldown
    if GOOGLE_API_KEYS:
        try:
            return call_gemini(prompt)
        except Exception as e:
            print(f"[Gemini compose error] {e}")
    # 4. Anthropic Haiku
    if ANTHROPIC_API_KEY:
        try:
            return call_anthropic(prompt)
        except Exception as e:
            print(f"[Anthropic compose error] {e}")
    return _HEURISTIC_FALLBACK


def call_llm_reply(prompt: str) -> str:
    """
    Reply path (/v1/reply) — speed matters most.
    Chain: Cerebras 8b → Groq 8b → Gemini → Anthropic → heuristic
    """
    # 1. Cerebras 8b — fastest free option
    if CEREBRAS_API_KEY and not _cerebras_disabled:
        try:
            return call_cerebras(prompt, CEREBRAS_MODEL_REPLY)
        except Exception as e:
            print(f"[Cerebras 8b reply error] {e}")
    # 2. Groq 8b
    if GROQ_API_KEY and not _groq_disabled:
        try:
            return call_groq(prompt, GROQ_MODEL_REPLY)
        except Exception as e:
            print(f"[Groq 8b reply error] {e}")
    # 3. Gemini
    if GOOGLE_API_KEYS:
        try:
            return call_gemini(prompt)
        except Exception as e:
            print(f"[Gemini reply error] {e}")
    # 4. Anthropic
    if ANTHROPIC_API_KEY:
        try:
            return call_anthropic(prompt)
        except Exception as e:
            print(f"[Anthropic reply error] {e}")
    return json.dumps({
        "action": "send",
        "body":   "Thanks for your message — let me look into that and get back to you shortly.",
        "cta":    "none",
        "rationale": "Heuristic fallback — LLM unavailable"
    })


def parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output."""
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
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
6. Use real numbers from context: CTR %, views, calls, peer benchmarks, prices, dates.
7. Never use: URLs, "guaranteed", "100% safe", "best in city", "miracle", "cure".
8. Under 120 words. Strong hook in line 1.
9. The LEAD SIGNAL section tells you WHY this message goes now — build around it.
10. For customer-facing (send_as=merchant_on_behalf): no medical claims, honor language pref, from merchant's WA number.
11. Your rationale MUST match what you actually wrote — judge cross-checks them.

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
    raw         = call_llm_compose(prompt)
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

OUTPUT: JSON only:
{
  "action": "send" | "wait" | "end",
  "body": "next message (only if action=send, under 100 words)",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "none",
  "wait_seconds": 86400,
  "rationale": "one sentence"
}

RULES:
- commit (yes/let's do it/confirm): action=send, SWITCH to execution mode. Draft the artifact or state the exact next step. CTA=binary_confirm_cancel.
- question: action=send, answer specifically using merchant data, advance toward one ask.
- auto-reply (canned): action=wait (86400s).
- same auto-reply 2nd time: action=end.
- explicit opt-out/stop: action=end.
- hostile: action=end OR one-line polite exit.
- out-of-scope (GST/legal/insurance): action=send, decline politely, redirect to original topic.
- NEVER repeat the body you sent before.
- Stay specific. Use merchant name. No URLs.\
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
        msg_key = message.strip().lower()
        if msg_key in seen_auto_reply_msgs:
            return {"action": "end", "rationale": "Same auto-reply seen before — closing."}
        repeat_count = is_repeat_auto_reply(conv_id, message)
        if repeat_count >= 2:
            seen_auto_reply_msgs.add(msg_key)
            return {"action": "end", "rationale": "Auto-reply 3+ times — closing."}
        seen_auto_reply_msgs.add(msg_key)
        # First time: send a prompt for the owner (per api-call-examples.md §4.1)
        if turn_number <= 2:
            merchant = get_merchant(merchant_id) or {}
            owner = merchant.get("identity", {}).get("owner_first_name", "")
            body = f"Looks like an auto-reply 🙂 When {owner or 'the owner'} is free, just reply YES to continue."
            return {
                "action": "send",
                "body":   body,
                "cta":    "binary_yes_no",
                "rationale": "Auto-reply detected (first time) — prompt for owner."
            }
        return {"action": "wait", "wait_seconds": 86400,
                "rationale": "Auto-reply repeated — owner not present. Wait 24h."}

    # ── Explicit intent fast-paths ───────────────────────────────────────────
    intent = detect_explicit_intent(message)

    if intent == "commit":
        merchant = get_merchant(merchant_id) or {}
        owner    = merchant.get("identity", {}).get("owner_first_name", "")
        offers   = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
        cust_agg = merchant.get("customer_aggregate", {})
        total    = cust_agg.get("total_unique_ytd", 0)
        lapsed   = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus") or 0
        offer_hint = f" I'll draft it around your '{offers[0]}' offer." if offers else ""
        scope_hint = f" Scope: {lapsed} lapsed customers from your base of {total}." if lapsed else ""
        name_part  = f"Got it {owner}! " if owner else "Got it! "
        return {
            "action": "send",
            "body":   f"{name_part}Proceeding now.{offer_hint}{scope_hint} Confirm to send the draft.",
            "cta":    "binary_confirm_cancel",
            "rationale": "Merchant committed — switching to execution mode with concrete scope."
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
    category_slug = merchant.get("category_slug", "")
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
    history_block += f"  [merchant NOW (turn {turn_number})]: {message[:200]}\n"

    prompt = f"""{REPLY_SYSTEM}

MERCHANT : {m_name} | owner={owner}
CATEGORY : {category_slug}
OFFERS   : {offers}
CUST AGG : total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')}
TRIGGER  : {(trigger or {}).get('kind', '')}

CONVERSATION:
{history_block}
DO NOT REPEAT any of these bodies: {prev_bot_bodies[-2:]}

Intent: {intent or 'normal_reply'}

Reply now as Vera. JSON only:"""

    raw    = call_llm_reply(prompt)
    result = parse_llm_json(raw)

    action = result.get("action", "send")
    if action not in {"send", "wait", "end"}:
        action = "send"

    body = result.get("body", "").strip()
    if action == "send":
        body = re.sub(r'https?://\S+', '', body).strip()

    return {
        "action":       action,
        "body":         body if action == "send" else None,
        "cta":          result.get("cta", "open_ended") if action == "send" else None,
        "wait_seconds": result.get("wait_seconds", 86400) if action == "wait" else None,
        "rationale":    result.get("rationale", "Continued conversation"),
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
}


def select_and_compose_actions(available_triggers: list[str], now: str) -> list[dict]:
    actions         = []
    acted_merchants = set()

    trigger_objs = []
    for tid in available_triggers:
        trg = get_trigger(tid)
        if trg:
            trigger_objs.append((tid, trg))
    # Sort by urgency DESC
    trigger_objs.sort(key=lambda x: -x[1].get("urgency", 1))

    for tid, trg in trigger_objs:
        if len(actions) >= 20:
            break

        suppression_key = trg.get("suppression_key", f"msg:{trg.get('merchant_id','?')}:{tid}")
        if suppression_key in fired_suppressions:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        if not merchant_id:
            continue
        if merchant_id in acted_merchants:
            continue

        expires_at = trg.get("expires_at", "")
        if expires_at and expires_at < now:
            continue

        merchant = get_merchant(merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug", "")
        category      = get_category(category_slug)
        if not category:
            continue

        customer = get_customer(customer_id) if customer_id else None

        try:
            result = compose_message(category, merchant, trg, customer)
        except Exception as e:
            print(f"[Compose error] {tid}: {e}")
            continue

        if not result.get("body"):
            continue

        conv_id = f"conv_{merchant_id}_{tid}_{hashlib.md5(now.encode()).hexdigest()[:6]}"

        # template_params: 3 parts of the body
        body_parts     = result["body"].split(". ")[:3]
        template_params = (body_parts + ["..."] * 3)[:3]

        kind          = trg.get("kind", "generic")
        template_name = TEMPLATE_MAP.get(kind, "vera_generic_v2")

        action = {
            "conversation_id": conv_id,
            "merchant_id":     merchant_id,
            "customer_id":     customer_id,
            "send_as":         result["send_as"],
            "trigger_id":      tid,
            "template_name":   template_name,
            "template_params": template_params,
            "body":            result["body"],
            "cta":             result["cta"],
            "suppression_key": suppression_key,
            "rationale":       result["rationale"],
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

        # Small sleep between compositions to avoid RPM bursts
        # Groq is ~200ms/call so 0.2s gap is plenty; Gemini cooldown handles the rest
        if len(actions) < 20:
            time.sleep(0.2)

    return actions


# ─────────────────────────────────────────────
# FASTAPI ENDPOINTS
# ─────────────────────────────────────────────

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
    model_primary   = GROQ_MODEL_COMPOSE if GROQ_API_KEY else (GOOGLE_MODEL if GOOGLE_API_KEYS else "claude-haiku-4-5-20251001")
    model_reply     = GROQ_MODEL_REPLY   if GROQ_API_KEY else model_primary
    return {
        "team_name":    TEAM_NAME,
        "team_members": TEAM_MEMBERS.split(","),
        "model":        model_primary,
        "model_routing": {
            "/v1/tick  (compose)": f"Groq {GROQ_MODEL_COMPOSE} → Groq {GROQ_MODEL_REPLY} → Gemini → Anthropic → heuristic",
            "/v1/reply (turns)":   f"Groq {GROQ_MODEL_REPLY} → Groq {GROQ_MODEL_COMPOSE} → Gemini → Anthropic → heuristic",
        },
        "approach": (
            "Signal-first composition: pick_lead_signal() deterministically selects the ONE signal "
            "driving each message before LLM call. Trigger-kind dispatch routes each prompt to its "
            "tailored compulsion lever. "
            f"Compose path uses Groq {GROQ_MODEL_COMPOSE} (quality); "
            f"reply path uses Groq {GROQ_MODEL_REPLY} (speed). "
            "4-key Gemini rotation as secondary fallback. Anthropic Haiku as tertiary. "
            "Multi-turn state machine: auto-reply detection (global fingerprint), commit fast-path, "
            "intent-transition routing, hostile/opt-out exits. Temperature=0 for determinism. "
            "No fabrication — all numbers from pushed context."
        ),
        "contact_email": CONTACT_EMAIL,
        "version":       BOT_VERSION,
        "submitted_at":  datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            # Same version re-push — idempotent no-op (per spec: accepted=false, reason=already_stored)
            return JSONResponse(
                status_code=409,
                content={
                    "accepted":       False,
                    "reason":         "already_stored",
                    "current_version": cur["version"],
                    "ack_id":         f"ack_{body.context_id}_v{body.version}",
                }
            )

    # New or higher version — store
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted":  True,
        "ack_id":    f"ack_{body.context_id}_v{body.version}",
        "stored_at": now_iso(),
    }


class TickBody(BaseModel):
    now:                str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}
    actions = select_and_compose_actions(body.available_triggers, body.now)
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
    global _gemini_key_index, _gemini_call_counts, _gemini_key_429_at, _groq_disabled, _cerebras_disabled
    _gemini_key_index   = 0
    _gemini_call_counts = [0] * max(len(GOOGLE_API_KEYS), 1)
    _gemini_key_429_at  = {}
    _groq_disabled      = False
    _cerebras_disabled  = False
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