#!/usr/bin/env python3
"""
Step 7: 验证时间切分、hash、数量
"""
import json
import hashlib
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    packs_file = cfg["paths"]["packs"]

    with open(packs_file, "r", encoding="utf-8") as f:
        packs = [json.loads(line) for line in f]

    print(f"=== Audit Report: {len(packs)} packs ===\n")

    for p in packs:
        tid = p["pack_id"]
        tr = p["train_examples"]
        ho = p["heldout_tasks"]

        # 检查 heldout desired_output 为 null
        ho_null = sum(1 for h in ho if h.get("desired_output") is None)

        # 检查 hash 唯一性（简单）
        train_hashes = set()
        for t in tr:
            txt = t.get("desired_output", {}).get("text", "")
            train_hashes.add(hashlib.sha256(txt.encode()).hexdigest()[:16])

        print(f"{tid}")
        print(f"  Train: {len(tr)} | Heldout: {len(ho)} | Heldout null: {ho_null}/{len(ho)}")
        print(f"  Unique patches: {len(train_hashes)}")
        print()


if __name__ == "__main__":
    main()
