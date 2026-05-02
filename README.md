# Vera Bot — magicpin AI Challenge Submission

## What this builds

A complete HTTP bot implementing Vera's message engine: `/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`.

---

## Approach

**Signal-first composition, not template-first.**

Every message starts with the trigger, not a template. The composer asks: *what one signal justifies this message right now?* Then it builds around that signal using real merchant numbers.

### Architecture

```
Judge → /v1/context  ─→  In-memory context store (category/merchant/customer/trigger)
Judge → /v1/tick     ─→  Trigger ranker → Prompt builder → LLM → JSON validator → actions[]
Judge → /v1/reply    ─→  Intent classifier → Phase router → LLM → response
```

### Trigger dispatch

Each `trigger.kind` routes to a tailored prompt variant with the specific compulsion lever pre-selected:

| Trigger kind | Lever |
|---|---|
| `research_digest` | Specificity + reciprocity (offer to draft) |
| `perf_dip` | Loss aversion (number + one action) |
| `perf_spike` | Social proof + momentum |
| `recall_due` | Personalized recall with specific slot + price |
| `festival_upcoming` | Urgency + counter-intuitive insight |
| `renewal_due` | Loss aversion (what stops if lapsed) |
| `curious_ask_due` | Asking the merchant (lowest-friction) |
| `review_theme_emerged` | Reciprocity (I noticed X about your account) |

### Multi-turn handling

State machine with phases: `opening → qualifying → executing → closing`

- **Auto-reply detection**: Pattern match on canned WA Business replies. First: prompt for owner. Second: wait 24h. Third: end.
- **Intent transitions**: Commit detected → switch to `executing` phase immediately. No re-qualifying.
- **Hostile/opt-out**: Fast-path exit (no LLM call).
- **Out-of-scope**: Polite deflection + redirect.

### Grounding rules (enforced post-LLM)

- No URLs stripped if LLM includes them
- Same body can't repeat in a conversation (anti-repetition check)
- Suppression keys tracked across ticks
- One action per merchant per tick

---

## Model choice

**Primary: Google Gemini 2.0 Flash** (free tier, fast, sufficient for composition)  
**Fallback: Claude Haiku 4.5** (set `ANTHROPIC_API_KEY`)  
**Temperature: 0** for determinism

---

## Tradeoffs

1. **In-memory state** — simple, survives the 60-min test window, doesn't survive process restarts. Fine for this context.
2. **Single LLM call per message** — no retrieval or tool use. Faster, cheaper; the prompt context is rich enough.
3. **Heuristic fallback** — if both APIs fail, returns a generic but structurally correct response. Better than a timeout.
4. **Trigger urgency sorting** — one action per merchant per tick sorted by urgency. May miss a lower-urgency item this tick; catches it next tick.

---

## What would have helped most

- Real slot availability data (we infer from merchant history)
- A merchant's WhatsApp conversation history beyond last 3 turns
- Peer benchmark data broken down by sub-locality (not just metro)
- Historical trigger → engagement correlation data to calibrate urgency thresholds

---

## Deploy

```bash
# Option 1: Render / Railway / Fly.io
# Push this repo, set env vars:
GOOGLE_API_KEY=your_free_gemini_key
TEAM_NAME=YourName
CONTACT_EMAIL=you@example.com

# Option 2: Docker
docker build -t vera-bot .
docker run -p 8080:8080 -e GOOGLE_API_KEY=xxx vera-bot

# Option 3: Local test
pip install -r requirements.txt
GOOGLE_API_KEY=xxx uvicorn bot:app --host 0.0.0.0 --port 8080
```

**Get free Gemini API key:** https://aistudio.google.com/apikey (no credit card, generous quota)

---

## Generate submission.jsonl

```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
GOOGLE_API_KEY=xxx python3 generate_submission.py --expanded expanded --out submission.jsonl
```

---

## Pre-flight checklist

- [x] All 5 endpoints implemented with correct schemas
- [x] `/v1/context` idempotent on `(scope, context_id, version)`
- [x] `/v1/tick` returns ≤20 actions, one per merchant, within 30s
- [x] `/v1/reply` handles auto-reply / commit / opt-out / out-of-scope
- [x] No URLs in message bodies
- [x] Anti-repetition check on body text
- [x] Suppression keys tracked
- [x] Temperature=0 for determinism
- [x] `conversation_handlers.py` with `respond()` for replay test
