#!/usr/bin/env python3
"""
Step 5: 应用 inclusion/exclusion 规则过滤
"""
import json
import os
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def should_keep(thread):
    """返回 True 表示保留"""
    # Exclusion 规则
    signal = thread.get("review_signal", "")
    patch = thread.get("accepted_change", "")

    # 1. 依赖隐藏 CI/内部数据的（简单判断：提到 CI/benchmark/private）
    lower = signal.lower()
    if any(x in lower for x in ["ci passed", "benchmark result", "internal", "private"]):
        return False

    # 2. 社交性评论
    if any(x in lower for x in ["thanks", "thank you", "great job", "lgtm", "nice work"]):
        return False

    # 3. 没有 actionable 内容
    if len(signal.strip()) < 10:
        return False

    # 4. patch 太大（>200 行变化）
    if patch and patch.count("\n") > 200:
        return False

    # 5. 没有 patch 内容
    if not patch or not patch.strip():
        return False

    # Inclusion：必须有 review signal 和 patch
    return True


def main():
    cfg = load_config()
    in_file = os.path.join(cfg["paths"]["cleaned"], "classified_threads.jsonl")
    out_file = os.path.join(cfg["paths"]["cleaned"], "filtered_threads.jsonl")

    with open(in_file, "r", encoding="utf-8") as f:
        threads = [json.loads(line) for line in f]

    kept = [t for t in threads if should_keep(t)]

    with open(out_file, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Filtered: {len(threads)} -> {len(kept)} threads kept -> {out_file}")


if __name__ == "__main__":
    main()
