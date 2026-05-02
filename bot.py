"""
Vera — magicpin Merchant AI Assistant
Full HTTP server implementing the judge harness contract.
Deploy: uvicorn bot:app --host 0.0.0.0 --port 8080

LLM: Google Gemini Flash (free tier — generous quota, fast, low cost)
Fallback: Claude Haiku via Anthropic API (if ANTHROPIC_API_KEY set)
"""

import os
import time
import json
import re
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG — set via env vars before deployment
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")          # free Gemini Flash
GOOGLE_MODEL = os.getenv("GEMINI_MODEL")  # optional model selection
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")    # optional fallback
TEAM_NAME = os.getenv("TEAM_NAME", "Vera-Builder")
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Candidate")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "candidate@example.com")
BOT_VERSION = "2.0.0"

app = FastAPI(title="MagicPin Vera Bot", version=BOT_VERSION)
START_TIME = time.time()

# ─────────────────────────────────────────────
# IN-MEMORY STATE
# ─────────────────────────────────────────────
# (scope, context_id) → {version, payload}
contexts: dict[tuple[str, str], dict] = {}
# conv_id → {turns:[], merchant_id, customer_id, trigger_id, suppressed, ended}
conversations: dict[str, dict] = {}
# suppression_keys already sent
fired_suppressions: set[str] = set()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_ctx(scope: str, ctx_id: str) -> Optional[dict]:
    entry = contexts.get((scope, ctx_id))
    return entry["payload"] if entry else None


def get_merchant(merchant_id: str) -> Optional[dict]:
    return get_ctx("merchant", merchant_id)


def get_category(slug: str) -> Optional[dict]:
    return get_ctx("category", slug)


def get_customer(customer_id: str) -> Optional[dict]:
    return get_ctx("customer", customer_id)


def get_trigger(trigger_id: str) -> Optional[dict]:
    return get_ctx("trigger", trigger_id)


def detect_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    patterns = [
        "thank you for contacting",
        "thanks for contacting",
        "our team will respond",
        "will get back to you",
        "automated assistant",
        "we have received your message",
        "aapki jaankari ke liye bahut-bahut shukriya",
        "main aapki yeh sabhi baatein",
        "aapki madad ke liye shukriya, lekin main ek automated",
        "this is an automated",
        "auto-reply",
    ]
    msg_lower = message.lower()
    return any(p in msg_lower for p in patterns)


def detect_explicit_intent(message: str) -> Optional[str]:
    """Detect clear merchant intent transitions."""
    msg_lower = message.lower()
    # Positive/action intent
    if any(p in msg_lower for p in ["let's do it", "lets do it", "ok do it", "go ahead", "yes let's", "haan karo",
                                    "confirm", "proceed", "start karo", "shuru karo", "yes please", "bilkul"]):
        return "commit"
    # Negative/opt-out
    if any(p in msg_lower for p in ["not interested", "stop messaging", "stop", "band karo", "mat karo",
                                    "unsubscribe", "do not contact", "mujhe nahi chahiye", "nahi chahiye"]):
        return "opt_out"
    # Out of scope
    if any(p in msg_lower for p in ["gst", "income tax", "loan", "insurance", "property", "legal advice"]):
        return "out_of_scope"
    return None


def is_repeat_auto_reply(conv_id: str, message: str) -> int:
    """Count how many times this exact (or near-identical) auto-reply appeared."""
    conv = conversations.get(conv_id, {})
    turns = conv.get("turns", [])
    count = sum(1 for t in turns if t.get("from") == "merchant" and
                t.get("message", "").strip().lower() == message.strip().lower())
    return count


# ─────────────────────────────────────────────
# LLM COMPOSER
# ─────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """Call Google Gemini Flash (free tier) for composition."""
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
                                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()










def call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude Haiku as fallback."""
    import urllib.request
    url = "https://api.anthropic.com/v1/messages"
    body = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    })
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def call_llm(prompt: str) -> str:
    """Call LLM with fallback chain."""
    if GOOGLE_API_KEY:
        try:
            return call_gemini(prompt)
        except Exception as e:
            print(f"Gemini error: {e}")
    if ANTHROPIC_API_KEY:
        try:
            return call_anthropic(prompt)
        except Exception as e:
            print(f"Anthropic error: {e}")
    return fallback_compose_heuristic(prompt)


def fallback_compose_heuristic(prompt: str) -> str:
    """Pure-logic fallback if no LLM key — uses signal extraction."""
    return json.dumps({
        "body": "Hi — quick update on your account. Want me to share a specific opportunity I spotted? Reply YES.",
        "cta": "binary_yes_no",
        "rationale": "Heuristic fallback — LLM unavailable"
    })


# ─────────────────────────────────────────────
# COMPOSITION ENGINE
# ─────────────────────────────────────────────

COMPOSE_SYSTEM = """You are Vera, magicpin's AI assistant for merchant growth. You write WhatsApp messages to Indian merchants.

RULES (non-negotiable):
1. Write ONE specific, grounded message using ONLY facts from the given contexts. No invention.
2. Use the merchant's first name or clinic/business name.
3. One clear CTA at the end — binary YES/NO or open-ended question. Never multi-choice except for booking slots.
4. Tone by category: dentists=peer-clinical, restaurants=fellow-operator, salons=warm-practical, gyms=coach-energetic, pharmacies=trustworthy-precise.
5. Hindi-English code-mix is preferred when merchant languages include "hi". Keep it natural.
6. Use real numbers from the data: CTR %, views, calls, peer benchmarks, prices.
7. No URLs. No promotional hype. No "guaranteed". No "best in city".
8. Keep under 120 words. Strong hook in line 1.
9. The reason for messaging NOW must be clear (the trigger).
10. For customer-facing messages: no medical claims, honor language preference, use merchant's name as sender.

OUTPUT: Return ONLY a JSON object with these fields:
{
  "body": "the WhatsApp message text",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "none",
  "rationale": "one sentence: which signal drove this + which compulsion lever used"
}

No markdown, no explanation, just the JSON."""


def build_compose_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
                         conv_history: list = None) -> str:
    """Build a tight, token-efficient prompt for composition."""

    # Extract key signals
    m_name = merchant.get("identity", {}).get("name", "Merchant")
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    city = merchant.get("identity", {}).get("city", "")
    locality = merchant.get("identity", {}).get("locality", "")
    langs = merchant.get("identity", {}).get("languages", ["en"])
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    signals = merchant.get("signals", [])
    active_offers = [o for o in merchant.get(
        "offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    # Trigger details
    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 2)

    # Find relevant digest item
    top_item_id = trg_payload.get("top_item_id")
    digest_item = None
    if top_item_id:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = d
                break

    # Seasonal / trend signals relevant to trigger
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    # Compute peer comparison
    ctr = perf.get("ctr", 0)
    peer_ctr = peer.get("avg_ctr", 0.03)
    ctr_vs_peer = f"{ctr:.3f} vs peer {peer_ctr:.3f} ({'BELOW' if ctr < peer_ctr else 'ABOVE'} peer)"

    # Build the slim context block
    ctx_block = f"""CATEGORY: {category.get('slug')} | tone={category.get('voice', {}).get('tone')} | code_mix={category.get('voice', {}).get('code_mix')}
taboo_words={category.get('voice', {}).get('vocab_taboo', [])}

MERCHANT:
  name={m_name} | owner={owner} | city={city}/{locality}
  languages={langs}
  plan={merchant.get('subscription', {}).get('status')} {merchant.get('subscription', {}).get('plan')} {merchant.get('subscription', {}).get('days_remaining')}d remaining
  perf_30d: views={perf.get('views')} calls={perf.get('calls')} directions={perf.get('directions')} ctr={ctr_vs_peer}
  delta_7d: views={perf.get('delta_7d', {}).get('views_pct')} calls={perf.get('delta_7d', {}).get('calls_pct')}
  active_offers={[o['title'] for o in active_offers]}
  customer_agg: total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')} retention={cust_agg.get('retention_6mo_pct') or cust_agg.get('retention_3mo_pct')}
  signals={signals}
  review_themes={[(r['theme'], r['sentiment'], r['occurrences_30d']) for r in review_themes]}
"""

    if cust_agg.get("high_risk_adult_count"):
        ctx_block += f"  high_risk_adults={cust_agg['high_risk_adult_count']}\n"

    if digest_item:
        ctx_block += f"""
DIGEST ITEM (this is WHY we're messaging):
  title={digest_item.get('title')}
  source={digest_item.get('source')}
  trial_n={digest_item.get('trial_n', '')}
  patient_segment={digest_item.get('patient_segment', '')}
  summary={digest_item.get('summary', '')}
  actionable={digest_item.get('actionable', '')}
"""

    ctx_block += f"""
TRIGGER:
  kind={trg_kind} | urgency={trg_urgency}/5
  payload={json.dumps(trg_payload, ensure_ascii=False)[:300]}
"""

    if customer:
        cust_id = customer.get("identity", {})
        rel = customer.get("relationship", {})
        ctx_block += f"""
CUSTOMER (message is sent on behalf of merchant TO this customer):
  name={cust_id.get('name')} | lang_pref={cust_id.get('language_pref')}
  state={customer.get('state')} | last_visit={rel.get('last_visit')} | visits={rel.get('visits_total')}
  services={rel.get('services_received', [])}
  preferred_slots={customer.get('preferences', {}).get('preferred_slots')}
  consent_scope={customer.get('consent', {}).get('scope', [])}
  send_as=merchant_on_behalf (from merchant's WA number, drafted by Vera)
"""

    if conv_history:
        last_turns = conv_history[-3:]
        ctx_block += "\nRECENT CONVERSATION:\n"
        for t in last_turns:
            ctx_block += f"  [{t.get('from', '')}]: {str(t.get('body', t.get('message', '')))[:120]}\n"

    send_as = "merchant_on_behalf" if customer else "vera"

    ctx_block += f"\nsend_as={send_as}\n"

    # Peer catalog hint for this trigger kind
    if trg_kind in ["research_digest", "regulation_change", "category_trend_movement"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: specificity + reciprocity (offer to draft something). CTA: open_ended.\n"
    elif trg_kind in ["perf_dip", "seasonal_perf_dip"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: loss aversion reframe. Show the number. Give one action. CTA: binary_yes_no.\n"
    elif trg_kind in ["perf_spike", "milestone_reached"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: social proof + momentum. What's next? CTA: open_ended.\n"
    elif trg_kind in ["recall_due", "chronic_refill_due", "customer_lapsed_soft", "customer_lapsed_hard"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: personalized recall with specific date/slot/price. CTA: multi_choice_slot or binary_yes_no.\n"
    elif trg_kind in ["festival_upcoming", "ipl_match_today", "weather_heatwave"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: urgency + counter-intuitive insight. What should merchant do NOW? CTA: binary_yes_no.\n"
    elif trg_kind == "renewal_due":
        ctx_block += "\nCOMPULSION LEVER TO USE: loss aversion (what stops if lapsed). Specific days left. CTA: binary_yes_no.\n"
    elif trg_kind == "curious_ask_due":
        ctx_block += "\nCOMPULSION LEVER TO USE: asking the merchant (reciprocity + low friction). CTA: open_ended.\n"
    elif trg_kind in ["review_theme_emerged", "dormant_with_vera"]:
        ctx_block += "\nCOMPULSION LEVER TO USE: reciprocity (you noticed something specific). CTA: open_ended.\n"

    return f"{COMPOSE_SYSTEM}\n\n{ctx_block}\n\nNow write the message JSON:"


def compose_message(category: dict, merchant: dict, trigger: dict,
                    customer: Optional[dict] = None,
                    conv_history: list = None) -> dict:
    """Core composer — returns {body, cta, rationale, send_as, suppression_key}."""

    prompt = build_compose_prompt(
        category, merchant, trigger, customer, conv_history)

    raw = call_llm(prompt)

    # Parse JSON from LLM
    result = {}
    try:
        # Strip markdown fences if any
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
    except Exception:
        # Try to extract JSON object
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except Exception:
                pass

    body = result.get("body", "").strip()
    cta = result.get("cta", "open_ended")
    rationale = result.get(
        "rationale", "Composed from trigger + merchant context")

    # Validate CTA
    valid_ctas = {"binary_yes_no", "open_ended",
                  "binary_confirm_cancel", "none", "multi_choice_slot"}
    if cta not in valid_ctas:
        cta = "open_ended"

    # Validate no URLs in body
    body = re.sub(r'https?://\S+', '[link]', body)

    send_as = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get(
        "suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger.get('id')}")

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale
    }


REPLY_SYSTEM = """You are Vera, magicpin's merchant AI assistant. You're mid-conversation with a merchant on WhatsApp.

OUTPUT: Return ONLY a JSON object:
{
  "action": "send" | "wait" | "end",
  "body": "next message (only if action=send)",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "none",
  "wait_seconds": 3600,  (only if action=wait)
  "rationale": "one sentence"
}

Rules:
- If merchant committed (yes/let's do it/confirm): action=send, switch to execution mode, offer a concrete next step
- If merchant asked a question: action=send, answer specifically, advance the conversation
- If auto-reply detected (canned thank-you/team-will-respond): action=wait (14400s) with rationale
- If same auto-reply 2+ times: action=end
- If explicit opt-out/stop/not interested: action=end
- If hostile: action=end or one-line polite acknowledgment then end
- If out-of-scope (GST/insurance etc): action=send, politely decline, redirect to original topic
- NEVER repeat body text you already sent in this conversation
- Stay on mission. Be specific. Use merchant name if you know it."""


def compose_reply(conv_id: str, merchant_id: str, customer_id: Optional[str],
                  from_role: str, message: str, turn_number: int) -> dict:
    """Compose a reply to a merchant/customer message."""

    conv = conversations.get(conv_id, {})
    turns = conv.get("turns", [])
    trigger_id = conv.get("trigger_id")

    # 1. Check if conversation is suppressed/ended
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation previously ended"}

    # 2. Detect auto-reply
    if detect_auto_reply(message):
        repeat_count = is_repeat_auto_reply(conv_id, message)
        if repeat_count >= 2:
            return {"action": "end", "rationale": "Auto-reply detected 3+ times. Closing conversation to avoid spam."}
        elif repeat_count >= 1:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Same auto-reply twice — owner not at phone. Waiting 24h."}
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 🙂 When the owner sees this, just reply YES to continue.",
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected — one prompt to flag for owner."
            }

    # 3. Detect explicit intent
    intent = detect_explicit_intent(message)
    if intent == "opt_out":
        return {"action": "end", "rationale": "Merchant opted out explicitly. Closing."}
    if intent == "out_of_scope":
        merchant = get_merchant(merchant_id)
        m_name = merchant.get("identity", {}).get(
            "owner_first_name", "") if merchant else ""
        return {
            "action": "send",
            "body": f"That's outside what I can help with directly — best to check with your CA or the relevant portal. Coming back to what we were discussing{' ' + m_name if m_name else ''} — want me to proceed?",
            "cta": "binary_yes_no",
            "rationale": "Out-of-scope deflected politely; redirected to original topic."
        }

    # 4. Build reply prompt with full context
    merchant = get_merchant(merchant_id) if merchant_id else {}
    customer = get_customer(customer_id) if customer_id else None
    trigger = get_trigger(trigger_id) if trigger_id else {}
    category_slug = (merchant or {}).get("category_slug", "")
    category = get_category(category_slug) if category_slug else {}

    # Gather conversation history for context
    history_block = ""
    for t in turns[-4:]:
        role = t.get("from", "")
        msg = t.get("body", t.get("message", ""))[:150]
        history_block += f"  [{role}]: {msg}\n"
    history_block += f"  [merchant JUST NOW (turn {turn_number})]: {message[:200]}\n"

    merchant_name = (merchant or {}).get("identity", {}).get("name", "")
    owner = (merchant or {}).get("identity", {}).get("owner_first_name", "")
    active_offers = [o["title"] for o in (merchant or {}).get(
        "offers", []) if o.get("status") == "active"]
    cust_agg = (merchant or {}).get("customer_aggregate", {})

    prompt = f"""{REPLY_SYSTEM}

MERCHANT: {merchant_name} | owner={owner}
CATEGORY: {category_slug}
ACTIVE_OFFERS: {active_offers}
CUSTOMER_AGG: total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')}
TRIGGER_KIND: {(trigger or {}).get('kind', '')}

CONVERSATION SO FAR:
{history_block}

Merchant's intent from this message: {intent or 'normal_reply'}

Reply now as Vera. JSON only:"""

    raw = call_llm(prompt)

    result = {}
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except Exception:
                pass

    action = result.get("action", "send")
    if action not in {"send", "wait", "end"}:
        action = "send"

    body = result.get("body", "").strip()
    if action == "send":
        body = re.sub(r'https?://\S+', '[link]', body)

    return {
        "action": action,
        "body": body if action == "send" else None,
        "cta": result.get("cta", "open_ended") if action == "send" else None,
        "wait_seconds": result.get("wait_seconds", 3600) if action == "wait" else None,
        "rationale": result.get("rationale", "Continued conversation")
    }


# ─────────────────────────────────────────────
# TICK LOGIC — which triggers to act on?
# ─────────────────────────────────────────────

def select_and_compose_actions(available_triggers: list[str], now: str) -> list[dict]:
    """Select triggers worth acting on and compose messages for them."""
    actions = []
    acted_merchants = set()  # One action per merchant per tick

    # Sort by urgency (highest first)
    trigger_objs = []
    for tid in available_triggers:
        trg = get_trigger(tid)
        if trg:
            trigger_objs.append((tid, trg))
    trigger_objs.sort(key=lambda x: -x[1].get("urgency", 1))

    for tid, trg in trigger_objs:
        if len(actions) >= 20:
            break

        suppression_key = trg.get("suppression_key", "")
        if suppression_key in fired_suppressions:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        if not merchant_id:
            continue
        if merchant_id in acted_merchants:
            continue

        # Skip expired triggers
        expires_at = trg.get("expires_at", "")
        if expires_at and expires_at < now:
            continue

        merchant = get_merchant(merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug", "")
        category = get_category(category_slug)
        if not category:
            continue

        customer = get_customer(customer_id) if customer_id else None

        # Compose the message
        try:
            result = compose_message(category, merchant, trg, customer)
        except Exception as e:
            print(f"Compose error for {tid}: {e}")
            continue

        if not result.get("body"):
            continue

        conv_id = f"conv_{merchant_id}_{tid}_{hashlib.md5(now.encode()).hexdigest()[:6]}"
        send_as = result["send_as"]

        # Build template params (first 3 meaningful parts of the body)
        body_parts = result["body"].split(". ")[:3]
        template_params = body_parts[:3] if len(
            body_parts) >= 3 else body_parts + ["..."] * (3 - len(body_parts))

        # Determine template name
        kind = trg.get("kind", "generic")
        template_map = {
            "research_digest": "vera_research_digest_v2",
            "regulation_change": "vera_compliance_alert_v2",
            "recall_due": "merchant_recall_reminder_v2",
            "perf_dip": "vera_perf_dip_v2",
            "perf_spike": "vera_perf_spike_v2",
            "festival_upcoming": "vera_festival_v2",
            "ipl_match_today": "vera_ipl_v2",
            "renewal_due": "vera_renewal_v2",
            "curious_ask_due": "vera_curious_ask_v2",
            "review_theme_emerged": "vera_review_theme_v2",
            "customer_lapsed_soft": "merchant_winback_v2",
            "customer_lapsed_hard": "merchant_winback_v2",
            "chronic_refill_due": "merchant_refill_v2",
            "supply_alert": "vera_supply_alert_v2",
        }
        template_name = template_map.get(kind, "vera_generic_v2")

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": tid,
            "template_name": template_name,
            "template_params": template_params,
            "body": result["body"],
            "cta": result["cta"],
            "suppression_key": suppression_key,
            "rationale": result["rationale"]
        }
        actions.append(action)

        # Track state
        fired_suppressions.add(suppression_key)
        acted_merchants.add(merchant_id)
        conversations[conv_id] = {
            "turns": [],
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": tid,
            "ended": False
        }

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
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS.split(","),
        "model": "gemini-2.0-flash" if GOOGLE_API_KEY else "claude-haiku-4-5-20251001",
        "approach": (
            "Trigger-dispatch composer: each trigger kind routes to a tailored prompt variant. "
            "Signals ranked by urgency, one action per merchant per tick. "
            "Auto-reply detection, intent-transition routing, graceful exit logic built in. "
            "Temperature=0 for determinism. Uses real merchant numbers — no fabrication."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in {"category", "merchant", "customer", "trigger"}:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope"})
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version", "current_version": cur["version"]
        })
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": now_iso()}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}
    actions = select_and_compose_actions(body.available_triggers, body.now)
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 1


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store the incoming turn
    conv = conversations.setdefault(body.conversation_id, {
        "turns": [], "merchant_id": body.merchant_id,
        "customer_id": body.customer_id, "trigger_id": None, "ended": False
    })
    conv["turns"].append({
        "from": body.from_role,
        "message": body.message,
        "ts": body.received_at or now_iso()
    })

    result = compose_reply(
        body.conversation_id, body.merchant_id,
        body.customer_id, body.from_role,
        body.message, body.turn_number
    )

    if result["action"] == "end":
        conv["ended"] = True

    if result["action"] == "send":
        # Store our reply turn too
        conv["turns"].append({
            "from": "vera",
            "body": result.get("body", ""),
            "ts": now_iso()
        })
        return {
            "action": "send",
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", "")
        }
    elif result["action"] == "wait":
        return {
            "action": "wait",
            "wait_seconds": result.get("wait_seconds", 3600),
            "rationale": result.get("rationale", "")
        }
    else:
        return {"action": "end", "rationale": result.get("rationale", "")}


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppressions.clear()
    return {"status": "ok", "message": "State wiped"}


# ─────────────────────────────────────────────
# STATIC COMPOSE FUNCTION (for submission.jsonl)
# ─────────────────────────────────────────────

def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    Public compose function for judge evaluation.
    Inputs: raw dicts from dataset JSON.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    return compose_message(category, merchant, trigger, customer)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, log_level="info")
