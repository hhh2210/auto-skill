#!/usr/bin/env python3
"""
Step 4: 对 thread 分类（thread_kind, path_cluster）
如果 config.llm.enabled=true，则调用 LLM；否则用规则分类
"""
import json
import os
from openai import OpenAI
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_by_rules(thread):
    """基于规则的快速分类"""
    path = thread.get("path", "")
    signal = thread.get("review_signal", "").lower()

    # Path cluster
    if "test" in path or path.endswith("_test.py"):
        cluster = "tests"
    elif "docs" in path or path.endswith(".md"):
        cluster = "docs"
    elif "model" in path:
        cluster = "models"
    elif "integration" in path:
        cluster = "integrations"
    else:
        cluster = "general"

    # Thread kind（简单启发式）
    if "suggestion" in signal or "```suggestion" in signal:
        kind = "suggestion_block"
    elif "rename" in signal or "format" in signal or "import" in signal:
        kind = "mechanical_followup"
    elif "?" in signal or len(signal.split()) > 30:
        kind = "discussion_resolution"
    else:
        kind = "code_owner_review"

    return kind, cluster


def classify_by_llm(thread, client, model):
    """调用 LLM 分类"""
    prompt = f"""You are a code review classifier. Given a GitHub review comment and file path, classify:

File path: {thread['path']}
Review comment: {thread['review_signal']}

Respond ONLY in JSON:
{{
  "thread_kind": "suggestion_block|code_owner_review|discussion_resolution|mechanical_followup",
  "path_cluster": "tests|docs|models|integrations|generation|examples|general",
  "confidence": 0.0-1.0
}}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content
        # 简单提取 JSON
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"LLM error: {e}")

    return None


def main():
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})

    client = None
    if llm_cfg.get("enabled"):
        client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
        print("LLM classification enabled")

    in_file = os.path.join(cfg["paths"]["cleaned"], "all_thread_slices.jsonl")
    out_file = os.path.join(cfg["paths"]["cleaned"], "classified_threads.jsonl")

    with open(in_file, "r", encoding="utf-8") as f:
        threads = [json.loads(line) for line in f]

    results = []
    for t in threads:
        if client:
            llm_res = classify_by_llm(t, client, llm_cfg["model"])
            if llm_res:
                t["thread_kind"] = llm_res.get("thread_kind", "unknown")
                t["path_cluster"] = llm_res.get("path_cluster", "general")
                t["confidence"] = llm_res.get("confidence", 0.5)
            else:
                t["thread_kind"], t["path_cluster"] = classify_by_rules(t)
                t["confidence"] = 0.3
        else:
            kind, cluster = classify_by_rules(t)
            t["thread_kind"] = kind
            t["path_cluster"] = cluster
            t["confidence"] = 0.5

        results.append(t)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Classified {len(results)} threads -> {out_file}")


if __name__ == "__main__":
    main()
