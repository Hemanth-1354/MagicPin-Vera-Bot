import os
import json
import urllib.request
import time

RENDER_URL = "https://magicpin-vera-bot-uvt8.onrender.com"
EXPANDED_DIR = "expanded"

def push(scope, cid, version, payload):
    url = f"{RENDER_URL}/v1/context"
    body = json.dumps({
        "scope": scope,
        "context_id": cid,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-05-02T22:50:00Z"
    }).encode()
    
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  FAIL {scope}/{cid}: {e}")
        return None

def main():
    print(f"Starting context push to {RENDER_URL}...")
    
    # 1. Categories
    cat_dir = os.path.join(EXPANDED_DIR, "categories")
    for f in os.listdir(cat_dir):
        if f.endswith(".json"):
            data = json.load(open(os.path.join(cat_dir, f)))
            print(f"Pushing category: {data['slug']}")
            push("category", data["slug"], 1, data)

    # 2. Merchants
    merch_dir = os.path.join(EXPANDED_DIR, "merchants")
    for f in os.listdir(merch_dir):
        if f.endswith(".json"):
            data = json.load(open(os.path.join(merch_dir, f)))
            print(f"Pushing merchant: {data['merchant_id']}")
            push("merchant", data["merchant_id"], 1, data)

    # 3. Triggers
    trg_dir = os.path.join(EXPANDED_DIR, "triggers")
    for f in os.listdir(trg_dir):
        if f.endswith(".json"):
            data = json.load(open(os.path.join(trg_dir, f)))
            print(f"Pushing trigger: {data['id']}")
            push("trigger", data["id"], 1, data)

    print("\nDONE! Your Render bot is now fully populated and ready for evaluation.")

if __name__ == "__main__":
    main()
