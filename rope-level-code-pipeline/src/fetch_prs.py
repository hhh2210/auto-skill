#!/usr/bin/env python3
"""
Step 1: 抓取 merged PR 元数据
"""
import json
import os
from github import Github
from tqdm import tqdm
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_merged_prs(repo_name, start_date, end_date, token, max_results):
    g = Github(token, per_page=100)
    repo = g.get_repo(repo_name)

    query = f"repo:{repo_name} is:pr is:merged merged:{start_date}..{end_date}"
    issues = g.search_issues(query, sort="created", order="asc")

    prs = []
    for issue in tqdm(issues, total=min(max_results, 1000), desc=f"Fetch PRs {repo_name}"):
        if len(prs) >= max_results:
            break
        try:
            pr = repo.get_pull(issue.number)
            if not pr.merged:
                continue
            prs.append(
                {
                    "repo": repo_name,
                    "pr_number": pr.number,
                    "title": pr.title,
                    "body": pr.body or "",
                    "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                    "created_at": pr.created_at.isoformat(),
                    "base_commit": pr.base.sha,
                    "merge_commit": pr.merge_commit_sha,
                    "user": pr.user.login if pr.user else None,
                    "changed_files": pr.changed_files,
                    "additions": pr.additions,
                    "deletions": pr.deletions,
                }
            )
        except Exception as e:
            print(f"  [Skip PR #{issue.number}] {e}")
            continue

    return prs


def main():
    cfg = load_config()
    token = cfg["github"]["token"]
    start = cfg["time_range"]["start"]
    end = cfg["time_range"]["end"]

    os.makedirs(cfg["paths"]["raw_prs"], exist_ok=True)

    for repo_cfg in cfg["repos"]:
        name = repo_cfg["name"]
        key = name.replace("/", "_").replace("-", "_")
        max_fetch = cfg["quotas"].get(f"{key}_candidates", 50)
        if "transformers" in name:
            max_fetch = cfg["quotas"]["transformers_candidates"]
        elif "vllm" in name:
            max_fetch = cfg["quotas"]["vllm_candidates"]

        print(f"\n>>> Fetching {name} (max {max_fetch})")
        prs = fetch_merged_prs(name, start, end, token, max_fetch)

        out = os.path.join(cfg["paths"]["raw_prs"], f"{name.replace('/', '_')}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for p in prs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(prs)} PRs -> {out}")


if __name__ == "__main__":
    main()
