"""
conversation_handlers.py — Multi-turn conversation state machine for Vera.

Implements the optional `respond` function for the tiebreaker round.
Also exposes ConversationState for use in bot.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import re
import json


# ─────────────────────────────────────────────
# STATE MODEL
# ─────────────────────────────────────────────

ConversationPhase = Literal[
    "opening",        # First message sent, awaiting reply
    "qualifying",     # Gathering merchant intent / preference
    "executing",      # Merchant committed — now do the work
    "awaiting_owner", # Auto-reply detected, waiting for real owner
    "cooling_off",    # Merchant asked for time / not now
    "closing"         # Graceful exit in progress
]

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    trigger_id: Optional[str]
    trigger_kind: str = ""
    phase: ConversationPhase = "opening"
    turns: list[dict] = field(default_factory=list)
    auto_reply_count: int = 0
    merchant_committed: bool = False
    merchant_opted_out: bool = False
    last_bot_body: str = ""
    meta: dict = field(default_factory=dict)  # for arbitrary trigger-specific state

    def add_turn(self, from_role: str, message: str):
        self.turns.append({"from": from_role, "message": message})

    def last_merchant_messages(self, n: int = 3) -> list[str]:
        return [t["message"] for t in self.turns if t["from"] == "merchant"][-n:]

    def all_bot_bodies(self) -> list[str]:
        return [t["message"] for t in self.turns if t["from"] == "vera"]


# ─────────────────────────────────────────────
# INTENT CLASSIFIERS
# ─────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank.{0,10}you for contacting",
    r"our team will (get back|respond)",
    r"automated (message|assistant|reply)",
    r"we (have received|received) your (message|query)",
    r"aapki jaankari ke liye",
    r"madad ke liye shukriya.*automated",
    r"main ek automated",
]

COMMIT_PATTERNS = [
    r"\blet'?s? do it\b",
    r"\bok (do it|go|proceed|chalte hain)\b",
    r"\b(yes|haan|ha|bilkul) (please|karo|proceed|let'?s?)\b",
    r"\bconfirm\b",
    r"\bgo ahead\b",
    r"\bshuru karo\b",
    r"\bsend it\b",
    r"\bdraft it\b",
]

OPT_OUT_PATTERNS = [
    r"\b(not interested|no thanks|nahi chahiye)\b",
    r"\bstop (messaging|sending|contacting)\b",
    r"\bband karo\b",
    r"\bmat bhejo\b",
    r"\bunsubscribe\b",
    r"\bdo not (message|contact|call)\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bgst (filing|return)\b",
    r"\bincome tax\b",
    r"\bproperty (dispute|loan)\b",
    r"\blegal advice\b",
    r"\binsurance claim\b",
]

HOSTILE_PATTERNS = [
    r"\b(useless|bakwas|rubbish|stupid bot|idiot|ch[u\*]+tiya)\b",
    r"\bwhy (are you|keep) (bothering|messaging)\b",
    r"\bstop (wasting|bothering)\b",
]


def classify_message(message: str) -> str:
    """Returns intent category."""
    msg_lower = message.lower()

    for p in AUTO_REPLY_PATTERNS:
        if re.search(p, msg_lower):
            return "auto_reply"

    for p in OPT_OUT_PATTERNS:
        if re.search(p, msg_lower):
            return "opt_out"

    for p in HOSTILE_PATTERNS:
        if re.search(p, msg_lower):
            return "hostile"

    for p in OUT_OF_SCOPE_PATTERNS:
        if re.search(p, msg_lower):
            return "out_of_scope"

    for p in COMMIT_PATTERNS:
        if re.search(p, msg_lower):
            return "commit"

    # Question detection
    if "?" in message or any(w in msg_lower for w in ["kya", "how", "when", "where", "what", "kaisa", "kitna", "kab"]):
        return "question"

    return "engaged"


# ─────────────────────────────────────────────
# PHASE TRANSITIONS
# ─────────────────────────────────────────────

def transition_phase(state: ConversationState, intent: str) -> ConversationPhase:
    """Determine next phase based on current phase + intent."""
    current = state.phase

    if intent == "auto_reply":
        state.auto_reply_count += 1
        if state.auto_reply_count >= 3:
            return "closing"
        return "awaiting_owner"

    if intent == "opt_out":
        state.merchant_opted_out = True
        return "closing"

    if intent == "hostile":
        return "closing"

    if intent == "commit":
        state.merchant_committed = True
        return "executing"

    if intent == "question" and current == "opening":
        return "qualifying"

    if intent in ("engaged", "question") and current in ("qualifying", "opening"):
        return "qualifying"

    if current == "executing":
        return "executing"

    return current


# ─────────────────────────────────────────────
# RESPONSE TEMPLATES BY PHASE + INTENT
# ─────────────────────────────────────────────

def build_reply_prompt(state: ConversationState, merchant_message: str,
                       merchant: dict, category: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Build a tight reply prompt based on conversation state."""

    intent = classify_message(merchant_message)
    new_phase = transition_phase(state, intent)
    state.phase = new_phase

    owner = merchant.get("identity", {}).get("owner_first_name", "")
    m_name = merchant.get("identity", {}).get("name", "")
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})
    langs = merchant.get("identity", {}).get("languages", ["en"])
    use_hindi = "hi" in langs or "hi-en mix" in str(merchant.get("identity", {}).get("owner_first_name", ""))

    # History (last 4 turns)
    history = "\n".join(f"  [{t['from']}]: {t['message'][:120]}" for t in state.turns[-4:])
    history += f"\n  [merchant NOW (turn {len(state.turns)+1})]: {merchant_message[:200]}"

    phase_instructions = {
        "executing": (
            "Merchant committed. Switch to ACTION mode immediately. "
            "Draft the specific artifact they asked for (post / message / template / plan). "
            "Name the concrete next step. Specific numbers. One binary CTA."
        ),
        "awaiting_owner": (
            "Auto-reply detected. "
            f"Count: {state.auto_reply_count}. "
            "If count=1: send one friendly prompt for the owner. "
            "If count=2: action=wait (86400s). "
            "If count>=3: action=end."
        ),
        "closing": (
            "Merchant opted out / hostile / 3+ auto-replies. "
            "action=end (or one-line polite exit then end). Do not pitch anything."
        ),
        "qualifying": (
            "Merchant is engaged but hasn't committed. "
            "Answer their question specifically. "
            "Advance toward one clear ask or commitment. "
            "Use merchant-specific data, not generic advice."
        ),
        "opening": (
            "First reply. Acknowledge and add value. Move toward action."
        )
    }

    system_block = f"""You are Vera, magicpin's AI assistant. Mid-conversation. Phase={new_phase}.
Intent detected: {intent}

PHASE INSTRUCTION: {phase_instructions.get(new_phase, 'Continue helpfully.')}

MERCHANT: {m_name} | owner={owner} | langs={langs}
ACTIVE_OFFERS: {active_offers}
CUSTOMER_AGG: total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')}
TRIGGER_KIND: {state.trigger_kind}

CONVERSATION:
{history}

RULES:
- Do NOT repeat: {state.last_bot_body[:80]}
- No URLs. One CTA. Under 100 words.
- Hindi-English mix if merchant uses Hindi.
- action must be one of: send / wait / end

OUTPUT JSON only:
{{"action": "send"|"wait"|"end", "body": "...", "cta": "binary_yes_no"|"open_ended"|"none", "wait_seconds": 3600, "rationale": "..."}}"""

    return system_block


def respond(state: ConversationState, merchant_message: str,
            merchant: dict = None, category: dict = None,
            trigger: dict = None, customer: dict = None) -> dict:
    """
    Multi-turn reply handler.
    Given current state + merchant message, returns the next Vera response.
    """
    # Import LLM from bot (avoid circular — inline here)
    from bot import call_llm

    state.add_turn("merchant", merchant_message)

    intent = classify_message(merchant_message)

    # Fast-path exits (no LLM needed)
    if intent == "opt_out":
        state.phase = "closing"
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Conversation closed."
        }

    if intent == "auto_reply":
        state.auto_reply_count += 1
        if state.auto_reply_count >= 3:
            state.phase = "closing"
            return {"action": "end", "rationale": "3 consecutive auto-replies. Closing."}
        elif state.auto_reply_count == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Second auto-reply — owner not present. Waiting 24h."}
        else:
            resp_body = "Looks like an auto-reply 🙂 When the owner's free, just reply YES to continue."
            state.add_turn("vera", resp_body)
            state.last_bot_body = resp_body
            return {
                "action": "send", "body": resp_body,
                "cta": "binary_yes_no",
                "rationale": "Auto-reply detected (first time). Prompt for owner."
            }

    if intent == "hostile":
        state.phase = "closing"
        return {
            "action": "send",
            "body": "Apologies for the interruption — won't message again. Restart anytime with 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Hostile message — one polite exit then close."
        }

    if intent == "out_of_scope":
        owner = (merchant or {}).get("identity", {}).get("owner_first_name", "")
        body = f"That's outside my scope — for GST/legal, your CA or portal is the right move. Back to what we were working on{' ' + owner if owner else ''} — shall we continue?"
        state.add_turn("vera", body)
        state.last_bot_body = body
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Out-of-scope deflected; redirected."}

    # LLM-powered reply
    prompt = build_reply_prompt(state, merchant_message, merchant or {}, category or {}, trigger or {}, customer)

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
        if body:
            state.add_turn("vera", body)
            state.last_bot_body = body

    if action == "end":
        state.phase = "closing"

    response = {"action": action, "rationale": result.get("rationale", "")}
    if action == "send":
        response["body"] = body
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 3600)

    return response
