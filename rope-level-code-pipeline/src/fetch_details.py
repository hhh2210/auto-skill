#!/usr/bin/env python3
"""
Step 2: 对每个 PR 抓取 review comments 和 files changed
"""
import json
import os
from github import Github
from tqdm import tqdm
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_details(repo_name, pr_number, token):
    g = Github(token, per_page=100)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Review comments（代码行级别）
    review_comments = []
    for c in pr.get_review_comments():
        review_comments.append(
            {
                "id": c.id,
                "path": c.path,
                "diff_hunk": c.diff_hunk,
                "body": c.body,
                "line": c.line,
                "original_line": c.original_line,
                "commit_id": c.commit_id,
                "original_commit_id": c.original_commit_id,
                "user": c.user.login if c.user else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "pull_request_review_id": c.pull_request_review_id,
            }
        )

    # Files changed
    files = []
    for f in pr.get_files():
        files.append(
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
                "previous_filename": f.previous_filename,
            }
        )

    # Issue comments（PR 页面上的普通评论）
    issue_comments = []
    for c in pr.get_issue_comments():
        issue_comments.append(
            {
                "id": c.id,
                "body": c.body,
                "user": c.user.login if c.user else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
        )

    return {
        "repo": repo_name,
        "pr_number": pr_number,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "review_comments": review_comments,
        "issue_comments": issue_comments,
        "files_changed": files,
    }


def main():
    cfg = load_config()
    token = cfg["github"]["token"]

    os.makedirs(cfg["paths"]["raw_threads"], exist_ok=True)

    for repo_cfg in cfg["repos"]:
        name = repo_cfg["name"]
        pr_file = os.path.join(cfg["paths"]["raw_prs"], f"{name.replace('/', '_')}.jsonl")
        if not os.path.exists(pr_file):
            continue

        with open(pr_file, "r", encoding="utf-8") as f:
            prs = [json.loads(line) for line in f]

        print(f"\n>>> Fetching details for {len(prs)} PRs in {name}")
        for meta in tqdm(prs):
            num = meta["pr_number"]
            try:
                detail = fetch_details(name, num, token)
                repo_dir = os.path.join(cfg["paths"]["raw_threads"], name.replace("/", "_"))
                os.makedirs(repo_dir, exist_ok=True)
                with open(os.path.join(repo_dir, f"pr_{num}.json"), "w", encoding="utf-8") as f:
                    json.dump(detail, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  [Error PR #{num}] {e}")


if __name__ == "__main__":
    main()
