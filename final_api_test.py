import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:8080"

def test_endpoint(name, method, path, body=None):
    print(f"Testing {name} ({method} {path})...", end=" ", flush=True)
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req) as res:
            print("[PASS] 200 OK")
            return json.loads(res.read())
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {e.code}")
        try: return json.loads(e.read())
        except: return {}

# 1. Push Category Context
cat_ctx = {
    "scope": "category",
    "context_id": "dentists",
    "version": 1,
    "payload": { "slug": "dentists", "voice": {"tone": "peer_clinical"} }
}
test_endpoint("Push Category", "POST", "/v1/context", cat_ctx)

# 2. Push Merchant Context (Linked to 'dentists')
merchant_ctx = {
    "scope": "merchant",
    "context_id": "m_001_drmeera",
    "version": 3,
    "payload": { 
        "merchant_id": "m_001_drmeera",
        "category_slug": "dentists", 
        "identity": {"name": "Dr. Meera", "owner_first_name": "Meera"}, 
        "performance": {"ctr": 0.021, "views": 1000}, 
        "offers": [{"title": "Dental Cleaning @ 299", "status": "active"}] 
    },
    "delivered_at": "2026-04-29T10:00:00Z"
}
test_endpoint("Push Merchant", "POST", "/v1/context", merchant_ctx)

# 3. Push Trigger Context
trigger_ctx = {
    "scope": "trigger",
    "context_id": "trg_research_digest_dentists",
    "version": 1,
    "payload": {
        "id": "trg_research_digest_dentists",
        "kind": "research_digest",
        "merchant_id": "m_001_drmeera",
        "payload": {"top_item_id": "d_fluoride_2026"}
    }
}
test_endpoint("Push Trigger", "POST", "/v1/context", trigger_ctx)

# 4. Tick
tick_data = { "now": "2026-04-29T10:30:00Z", "available_triggers": ["trg_research_digest_dentists"] }
res_tick = test_endpoint("Tick", "POST", "/v1/tick", tick_data)

# 5. Reply
reply_data = { "conversation_id": "conv_001", "from_role": "merchant", "message": "Yes, send me the abstract", "turn_number": 2 }
res_reply = test_endpoint("Reply", "POST", "/v1/reply", reply_data)

# 6. Metadata & Healthz
test_endpoint("Healthz", "GET", "/v1/healthz")
res_meta = test_endpoint("Metadata", "GET", "/v1/metadata")

print("\n--- Final Results ---")
actions = res_tick.get('actions', [])
print(f"Tick Actions Count: {len(actions)}")
if actions:
    print(f"Sample Action Body: {actions[0].get('body')[:100]}...")
