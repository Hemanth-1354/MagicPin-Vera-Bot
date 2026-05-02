"""
local_test.py — Self-test all 5 endpoints before submitting.
Run: python3 local_test.py (with bot.py running on :8080)
"""

import json
import sys
import time

try:
    import urllib.request
    import urllib.error
except ImportError:
    print("urllib not available")
    sys.exit(1)

BOT_URL = "http://localhost:8080"

PASS = "✅"
FAIL = "❌"

results = []


def req(method, path, body=None):
    url = BOT_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status} {name}" + (f": {detail}" if detail else ""))


print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Vera Bot — Local Pre-flight Test")
print(f"  Target: {BOT_URL}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# 1. Healthz
print("1. GET /v1/healthz")
code, body = req("GET", "/v1/healthz")
check("status=200", code == 200)
check("status field = ok", body.get("status") == "ok")
check("uptime_seconds present", "uptime_seconds" in body)
check("contexts_loaded present", "contexts_loaded" in body)

# 2. Metadata
print("\n2. GET /v1/metadata")
code, body = req("GET", "/v1/metadata")
check("status=200", code == 200)
check("team_name present", bool(body.get("team_name")))
check("model present", bool(body.get("model")))
check("version present", bool(body.get("version")))

# 3. Context push
print("\n3. POST /v1/context — category")
cat_payload = {
    "scope": "category",
    "context_id": "dentists",
    "version": 1,
    "delivered_at": "2026-04-26T09:45:00Z",
    "payload": {
        "slug": "dentists",
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
        "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ ₹299", "value": "299", "audience": "new_user", "type": "service_at_price"}],
        "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030},
        "digest": [{"id": "d_2026W17_jida_fluoride", "kind": "research", "title": "3-month fluoride recall cuts caries 38% better", "source": "JIDA Oct 2026, p.14", "trial_n": 2100, "patient_segment": "high_risk_adults"}],
        "patient_content_library": [],
        "seasonal_beats": [],
        "trend_signals": []
    }
}
code, body = req("POST", "/v1/context", cat_payload)
check("accepted=True", body.get("accepted") == True, str(body))
check("ack_id present", bool(body.get("ack_id")))

# Idempotency check (same version)
code2, body2 = req("POST", "/v1/context", cat_payload)
check("409 on re-push same version", code2 == 409, str(body2))

# Push merchant context
print("\n4. POST /v1/context — merchant")
merchant_payload = {
    "scope": "merchant",
    "context_id": "m_001_drmeera_dentist_delhi",
    "version": 1,
    "delivered_at": "2026-04-26T09:45:30Z",
    "payload": {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "category_slug": "dentists",
        "identity": {"name": "Dr. Meera's Dental Clinic", "city": "Delhi", "locality": "Lajpat Nagar",
                     "verified": True, "languages": ["en", "hi"], "owner_first_name": "Meera"},
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {"window_days": 30, "views": 2410, "calls": 18, "directions": 45,
                        "ctr": 0.021, "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05}},
        "offers": [{"id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active"}],
        "conversation_history": [],
        "customer_aggregate": {"total_unique_ytd": 540, "lapsed_180d_plus": 78,
                               "retention_6mo_pct": 0.38, "high_risk_adult_count": 124},
        "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"]
    }
}
code, body = req("POST", "/v1/context", merchant_payload)
check("merchant accepted", body.get("accepted") == True)

# Push trigger
print("\n5. POST /v1/context — trigger")
trigger_payload = {
    "scope": "trigger",
    "context_id": "trg_001_research_digest_dentists",
    "version": 1,
    "delivered_at": "2026-04-26T10:32:00Z",
    "payload": {
        "id": "trg_001_research_digest_dentists",
        "scope": "merchant",
        "kind": "research_digest",
        "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
        "urgency": 2,
        "suppression_key": "research:dentists:2026-W17",
        "expires_at": "2026-05-03T00:00:00Z"
    }
}
code, body = req("POST", "/v1/context", trigger_payload)
check("trigger accepted", body.get("accepted") == True)

# 6. Tick
print("\n6. POST /v1/tick")
tick_payload = {
    "now": "2026-04-26T10:35:00Z",
    "available_triggers": ["trg_001_research_digest_dentists"]
}
t0 = time.time()
code, body = req("POST", "/v1/tick", tick_payload)
elapsed = time.time() - t0
check("status=200", code == 200, str(code))
check("actions is list", isinstance(body.get("actions"), list))
check("responded in <30s", elapsed < 30, f"{elapsed:.1f}s")

actions = body.get("actions", [])
if actions:
    a = actions[0]
    check("action has body", bool(a.get("body")))
    check("action has cta", bool(a.get("cta")))
    check("action has rationale", bool(a.get("rationale")))
    check("action has conversation_id", bool(a.get("conversation_id")))
    check("action has suppression_key", bool(a.get("suppression_key")))
    check("no URLs in body", "http://" not in a.get("body", "")
          and "https://" not in a.get("body", ""))
    print(f"\n  📩 Sample message:\n  {a.get('body', '')[:200]}")

    # Tick again — should NOT re-send same suppression_key (idempotency)
    code2, body2 = req("POST", "/v1/tick", tick_payload)
    actions2 = body2.get("actions", [])
    check("suppression works (no dup action)", len(
        actions2) == 0, f"got {len(actions2)} actions")

    # 7. Reply
    if actions:
        conv_id = actions[0].get("conversation_id", "conv_test_001")
        print(f"\n7. POST /v1/reply (engaged)")
        reply_payload = {
            "conversation_id": conv_id,
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Yes please send the abstract",
            "received_at": "2026-04-26T10:42:00Z",
            "turn_number": 2
        }
        t0 = time.time()
        code, body = req("POST", "/v1/reply", reply_payload)
        elapsed = time.time() - t0
        check("reply status=200", code == 200)
        check("reply has action", body.get(
            "action") in {"send", "wait", "end"})
        check("reply within 30s", elapsed < 30, f"{elapsed:.1f}s")
        if body.get("action") == "send":
            check("reply body not empty", bool(body.get("body")))
            print(f"\n  💬 Reply:\n  {body.get('body', '')[:200]}")

        # Auto-reply test
        print("\n8. POST /v1/reply (auto-reply detection)")
        ar_payload = {**reply_payload,
                      "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
                      "turn_number": 3}
        code, body = req("POST", "/v1/reply", ar_payload)
        check("auto-reply handled", body.get("action")
              in {"send", "wait", "end"}, body.get("action"))

        # Opt-out test
        print("\n9. POST /v1/reply (opt-out)")
        opt_payload = {
            **reply_payload, "message": "Not interested. Stop messaging me.", "turn_number": 4}
        code, body = req("POST", "/v1/reply", opt_payload)
        check("opt-out → end", body.get("action") == "end", body.get("action"))

else:
    print("  ℹ️  No actions in tick (LLM may not be configured)")
    print("     Set GOOGLE_API_KEY or ANTHROPIC_API_KEY to test composition")

# Summary
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  🎉 All checks passed — ready to submit!")
else:
    failed = [(n, d) for n, ok, d in results if not ok]
    print(f"  ⚠️  Failed: {[n for n, _ in failed]}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
