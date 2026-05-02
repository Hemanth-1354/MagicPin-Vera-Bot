"""
generate_submission.py — Generate submission.jsonl for the 30 canonical test pairs.

Run this AFTER generating the expanded dataset:
  python3 dataset/generate_dataset.py --seed-dir dataset --out expanded

Then:
  GOOGLE_API_KEY=xxx python3 generate_submission.py --expanded expanded --out submission.jsonl

Or with Anthropic:
  ANTHROPIC_API_KEY=xxx python3 generate_submission.py --expanded expanded --out submission.jsonl
"""

import sys
import os
import json
import argparse
import time

# Ensure bot.py is importable
sys.path.insert(0, os.path.dirname(__file__))
from bot import compose

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", default="expanded", help="Path to expanded/ directory")
    parser.add_argument("--out", default="submission.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    expanded = args.expanded

    # Load all contexts
    categories = {}
    for cat_file in os.listdir(os.path.join(expanded, "categories")):
        if cat_file.endswith(".json"):
            data = load_json(os.path.join(expanded, "categories", cat_file))
            categories[data["slug"]] = data

    merchants = {}
    for m_file in os.listdir(os.path.join(expanded, "merchants")):
        if m_file.endswith(".json"):
            data = load_json(os.path.join(expanded, "merchants", m_file))
            merchants[data["merchant_id"]] = data

    customers = {}
    cust_dir = os.path.join(expanded, "customers")
    if os.path.exists(cust_dir):
        for c_file in os.listdir(cust_dir):
            if c_file.endswith(".json"):
                data = load_json(os.path.join(expanded, "customers", c_file))
                customers[data["customer_id"]] = data

    triggers = {}
    trg_dir = os.path.join(expanded, "triggers")
    if os.path.exists(trg_dir):
        for t_file in os.listdir(trg_dir):
            if t_file.endswith(".json"):
                data = load_json(os.path.join(expanded, "triggers", t_file))
                triggers[data["id"]] = data

    # Load test pairs
    test_pairs_path = os.path.join(expanded, "test_pairs.jsonl")
    if not os.path.exists(test_pairs_path):
        print(f"ERROR: {test_pairs_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    test_pairs = load_jsonl(test_pairs_path)
    print(f"Found {len(test_pairs)} test pairs")

    results = []
    for i, pair in enumerate(test_pairs):
        test_id = pair.get("test_id", f"T{i+1:02d}")
        merchant_id = pair.get("merchant_id")
        trigger_id = pair.get("trigger_id")
        customer_id = pair.get("customer_id")  # may be None

        merchant = merchants.get(merchant_id)
        trigger = triggers.get(trigger_id)
        customer = customers.get(customer_id) if customer_id else None

        if not merchant:
            print(f"  WARNING: merchant {merchant_id} not found for {test_id}")
            continue
        if not trigger:
            print(f"  WARNING: trigger {trigger_id} not found for {test_id}")
            continue

        category_slug = merchant.get("category_slug", "")
        category = categories.get(category_slug)
        if not category:
            print(f"  WARNING: category {category_slug} not found for {test_id}")
            continue

        print(f"  [{i+1:02d}/{len(test_pairs)}] {test_id}: {merchant_id} + {trigger_id}", end=" ")

        try:
            result = compose(category, merchant, trigger, customer)
            output = {
                "test_id": test_id,
                "merchant_id": merchant_id,
                "trigger_id": trigger_id,
                "customer_id": customer_id,
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", "")
            }
            results.append(output)
            print(f"✓ ({len(result.get('body',''))} chars)")
        except Exception as e:
            print(f"✗ ERROR: {e}")

        time.sleep(4)  # Rate limit buffer

    # Write output
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Written {len(results)}/{len(test_pairs)} results to {args.out}")

if __name__ == "__main__":
    main()
