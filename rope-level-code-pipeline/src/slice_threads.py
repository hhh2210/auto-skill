#!/usr/bin/env python3
"""
Step 3: 将 review comment 与对应的 diff hunk / patch 匹配
输出：thread 切片（一条 review + 对应代码上下文 + 修改结果）
"""
import json
import os
import re
from tqdm import tqdm
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_changed_lines(patch):
    """从 patch 提取修改后的行号范围"""
    if not patch:
        return None, None
    # 简单解析 @@ -old,old_len +new,new_len @@
    m = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch)
    if m:
        start = int(m.group(1))
        length = int(m.group(2)) if m.group(2) else 1
        return start, start + length
    return None, None


def match_comment_to_file(comment, files_changed):
    """将 review comment 匹配到对应的 file patch"""
    target_path = comment["path"]
    for f in files_changed:
        if f["filename"] == target_path or f.get("previous_filename") == target_path:
            return f
    return None


def build_thread_slices(pr_detail):
    """从单个 PR 的详情中提取 thread slices"""
    threads = []
    files = pr_detail["files_changed"]

    for rc in pr_detail.get("review_comments", []):
        matched_file = match_comment_to_file(rc, files)
        if not matched_file:
            continue

        start_line, end_line = extract_changed_lines(matched_file.get("patch", ""))

        # 构造 before_context：从 diff_hunk 提取
        diff_hunk = rc.get("diff_hunk", "")

        thread = {
            "repo": pr_detail["repo"],
            "pr_number": pr_detail["pr_number"],
            "merged_at": pr_detail["merged_at"],
            "thread_id": f"{pr_detail['repo']}#{pr_detail['pr_number']}/review_{rc['id']}",
            "path": rc["path"],
            "review_signal": rc["body"],
            "diff_hunk": diff_hunk,
            "file_patch": matched_file.get("patch", ""),
            "before_context": diff_hunk,  # 简化：直接用 diff_hunk 作为上下文
            "accepted_change": matched_file.get("patch", ""),  # 简化：整个 file patch
            "line_start": start_line,
            "line_end": end_line,
            "user": rc.get("user"),
        }
        threads.append(thread)

    return threads


def main():
    cfg = load_config()
    os.makedirs(cfg["paths"]["cleaned"], exist_ok=True)

    all_threads = []

    for repo_cfg in cfg["repos"]:
        name = repo_cfg["name"]
        repo_dir = os.path.join(cfg["paths"]["raw_threads"], name.replace("/", "_"))
        if not os.path.exists(repo_dir):
            continue

        files = [f for f in os.listdir(repo_dir) if f.startswith("pr_")]
        print(f"\n>>> Slicing {len(files)} PRs for {name}")

        for fname in tqdm(files):
            with open(os.path.join(repo_dir, fname), "r", encoding="utf-8") as f:
                detail = json.load(f)
            threads = build_thread_slices(detail)
            all_threads.extend(threads)

    # 保存所有 thread slices
    out = os.path.join(cfg["paths"]["cleaned"], "all_thread_slices.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t in all_threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nTotal thread slices: {len(all_threads)} -> {out}")


if __name__ == "__main__":
    main()
