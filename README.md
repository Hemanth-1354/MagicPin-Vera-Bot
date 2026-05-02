# Vera — magicpin AI Merchant Growth Engine

**Team:** Vera Dheera Soora · **Version:** 4.1.0 · **Model:** Llama-3.3-70B (Groq)

---

## What this is

Vera is a signal-driven WhatsApp AI assistant for magicpin merchants. It implements the full judge harness contract — five HTTP endpoints that receive merchant context, fire proactive messages, and handle multi-turn conversations — with a deliberate architecture that keeps every generated message anchored to real business data before the LLM is ever called.

---

## Architecture

### Signal-First Engine

The most important design decision is `pick_lead_signal()` — a deterministic function that runs *before* the LLM. Given a trigger kind (`perf_dip`, `recall_due`, `research_digest`, etc.) and the merchant's live data, it selects one specific business signal and packages it as a structured hook:

```
signal_text  →  what's happening (with real numbers)
anchor       →  the merchant-specific fact that makes it urgent
actionable   →  what to do right now
hook         →  how to open the message
lever        →  the psychological driver (loss aversion, reciprocity, urgency)
cta_type     →  which kind of close to use
```

This prevents the LLM from drifting. The prompt tells the model exactly what signal to build around, so the generated `body` and `rationale` always match.

### Trigger coverage

Every trigger kind maps to a specific signal strategy:

| Trigger kind | Lever | CTA |
|---|---|---|
| `perf_dip` / `seasonal_perf_dip` | Loss aversion — show the gap vs peers | binary yes/no |
| `perf_spike` / `milestone_reached` | Momentum — what's the next move | open ended |
| `research_digest` / `regulation_change` | Benchmarking + reciprocity | open ended |
| `competitor_opened` | Loss aversion — protect retention | binary yes/no |
| `recall_due` / `chronic_refill_due` | Personalised recall with slot/date | slot choice |
| `customer_lapsed_soft` / `_hard` | No-shame win-back + specific past service | binary yes/no |
| `festival_upcoming` / `ipl_match_today` | Contrarian insight + event urgency | binary yes/no |
| `renewal_due` | Loss aversion — what stops if it lapses | binary yes/no |
| `supply_alert` | Urgency + batch numbers + affected count | open ended |
| `review_theme_emerged` | Reciprocity — I spotted something specific | open ended |
| `appointment_tomorrow` | Personalised confirmation | binary yes/no |
| `trial_followup` | Conversion window while experience is fresh | binary yes/no |
| `curious_ask_due` | Lowest-friction check-in | open ended |
| Novel/unknown kind | Scans merchant `signals[]` for best available anchor | open ended |

### Compose prompt rules (enforced in the system prompt)

- Every message must contain at least one numeric anchor (CTR %, views, calls, price, date)
- Address merchant by first name or business name — never generic "Hi"
- One CTA only, at the end
- Zero emojis for dentists and pharmacies; max 2 for others
- No URLs, no "guaranteed", no "best in city", no "cure"
- Under 50 words
- No filler phrases ("I noticed", "I hope you are well")
- No reflective questions — Vera provides the analysis, not the merchant
- Rationale must match what was actually written

### Category voice

| Category | Tone |
|---|---|
| dentists | Peer-clinical — collegial, source-citing, no overclaim |
| restaurants | Fellow-operator — P&L language, covers, AOV, Swiggy/Zomato |
| salons | Warm-practical — service names, relationship continuity |
| gyms | Coach-energetic — goal-oriented, seasonal awareness |
| pharmacies | Trustworthy-precise — molecule names, batch numbers, no alarm |

Hindi-English code-mix fires automatically when the merchant's `languages` field includes `"hi"`.

### Multi-turn reply engine

`/v1/reply` runs a fast-path state machine before touching the LLM:

1. **Auto-reply detection** — pattern-matches WhatsApp Business canned replies. First occurrence: prompt for the real owner. Second+ occurrence: `action=wait` (24h). Same auto-reply seen before in this conversation: `action=end`.
2. **Commit intent** — "confirm", "let's do it", "go ahead" → immediate `action=send` with execution-mode response drafting the artifact, scope, and next step. No re-qualifying.
3. **Opt-out** — `action=end`, graceful close.
4. **Hostile** — one polite exit message, then end.
5. **Out-of-scope** (GST, loans, insurance) — deflect to appropriate resource, bridge back to original topic.
6. **Everything else** — LLM reply with full conversation history, anti-repeat guard on previous bot bodies, and numeric anchor requirement.

### LLM infrastructure

Primary: **Groq** — 5 rotating API keys (`GROQ_API_KEY_1` through `GROQ_API_KEY_5`), per-key 429 cooldown tracking, automatic rotation to the next available key on rate-limit.

- Compose calls (`/v1/tick`): `llama-3.3-70b-versatile` → fallback to `llama-3.1-8b-instant`
- Reply calls (`/v1/reply`): `llama-3.1-8b-instant` → fallback to `llama-3.3-70b-versatile`

Supporting providers (implemented, available as extension):

- **SambaNova** — own silicon, free tier, OpenAI-compatible
- **Gemini** — 4 rotating keys with per-key cooldown and alt-model fallback

JSON parsing is robust: standard parse first, then regex extraction of `body`/`cta`/`rationale` fields as fallback. Empty `send` bodies are caught and replaced with a category-anchored safety string.

### Concurrency

`/v1/tick` processes up to 20 triggers per call using `asyncio.gather` with `asyncio.Semaphore(4)`. Triggers are sorted by urgency descending before processing. One action per merchant per tick (deduped by `acted_merchants`). Suppression keys prevent re-firing the same message across ticks.

---

## Endpoints

### `GET /v1/healthz`
Liveness probe. Returns `status: ok`, uptime, and context counts by scope.

### `GET /v1/metadata`
Team identity for the leaderboard.

### `POST /v1/context`
Receives category, merchant, customer, or trigger context. Idempotent by `(context_id, version)` — same version is a no-op, lower version returns 409, higher version replaces atomically.

### `POST /v1/tick`
Periodic wake-up. Takes `now` and `available_triggers`, returns up to 20 composed actions. Each action includes `conversation_id`, `merchant_id`, `trigger_id`, `body`, `cta`, `rationale`, `suppression_key`, `template_name`, and `template_params`.

### `POST /v1/reply`
Receives a merchant or customer reply to a previous bot message. Returns `send` / `wait` / `end` within 30 seconds.

### `POST /v1/teardown`
Wipes all in-memory state between judge test runs: contexts, conversations, suppression keys, auto-reply cache, and all LLM provider rate-limit state.

---

## Running locally

```bash
pip install -r requirements.txt

# .env
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...
TEAM_NAME=Vera Dheera Soora
TEAM_MEMBERS=Hemanth
BOT_VERSION=4.1.0

uvicorn bot:app --host 0.0.0.0 --port 8080
```

Verify:
```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

## Deployment

Configured for Render via `render.yaml`. Set env vars in the Render dashboard — do not commit `.env` to the repo.

Live URL: `https://magicpin-vera-bot-uvt8.onrender.com`