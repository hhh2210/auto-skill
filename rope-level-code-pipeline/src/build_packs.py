#!/usr/bin/env python3
"""
Step 6: 将清洗后的 threads 组织成 example-pack/v1 格式
"""
import json
import os
import hashlib
import yaml


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_example(thread, repo, is_heldout=False):
    """单条 example 格式"""
    ex_id = f"{repo.replace('/', '_')}::{'heldout' if is_heldout else 'train'}::{thread['thread_id']}"

    task_input = f"""Repository: {repo}
File: {thread['path']}
Review Comment:
{thread['review_signal']}

Original Code Context:
{thread.get('before_context', '')}
"""

    ex = {
        "schema_version": "example-pack/v1",
        "example_id": ex_id,
        "source": "CodeBench",
        "source_task_id": f"{repo}#{thread['pr_number']}",
        "domain": {
            "primary": "code",
            "secondary": thread.get("path_cluster", "general"),
            "language": "python",
        },
        "task_input": task_input,
        "materials": [thread.get("review_signal", "")],
        "expected_artifacts": ["code_patch"],
    }

    if not is_heldout:
        patch_text = thread.get("accepted_change", "")
        ex["desired_output"] = {
            "schema_version": "extracted-patch/v1",
            "status": "extracted",
            "text": patch_text,
            "patch_sha256": hashlib.sha256(patch_text.encode()).hexdigest()[:16],
        }
    else:
        ex["desired_output"] = None

    return ex


def build_pack(threads, repo, path_cluster, train_cutoff):
    """构建一个 pack"""
    train = [t for t in threads if t.get("merged_at", "") < train_cutoff]
    heldout = [t for t in threads if t.get("merged_at", "") >= train_cutoff]

    pack = {
        "schema_version": "example-pack/v1",
        "builder_version": "code-pack-builder/0.1",
        "pack_id": f"codebench_{repo.replace('/', '_')}_{path_cluster}",
        "split_id": f"codebench::{repo.split('/')[-1]}::{path_cluster}",
        "source": "CodeBench",
        "learning_problem": "few_shot_skill_induction",
        "domain": {
            "primary": "code",
            "secondary": path_cluster,
            "language": "python",
        },
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.materials",
                "train_examples.desired_output.text",
            ],
            "must_not_use_for_induction": [
                "private rubrics",
                "private judge prompts",
                "heldout tasks",
                "heldout scores",
                "CI logs",
                "maintainer-only notes",
            ],
        },
        "train_examples": [build_example(t, repo, False) for t in train],
        "heldout_tasks": [build_example(t, repo, True) for t in heldout],
        "example_pack_status": {
            "desired_outputs_applied": len(train),
            "desired_outputs_missing": 0,
            "desired_outputs_rejected": 0,
            "rejections": [],
        },
    }
    return pack


def main():
    cfg = load_config()
    cutoff = cfg["train_cutoff"]

    in_file = os.path.join(cfg["paths"]["cleaned"], "filtered_threads.jsonl")
    out_file = cfg["paths"]["packs"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as f:
        threads = [json.loads(line) for line in f]

    # 按 repo + path_cluster 分组
    groups = {}
    for t in threads:
        key = (t["repo"], t.get("path_cluster", "general"))
        groups.setdefault(key, []).append(t)

    packs = []
    for (repo, cluster), items in groups.items():
        if len(items) < 3:  # 太少的跳过
            continue
        pack = build_pack(items, repo, cluster, cutoff)
        packs.append(pack)

    with open(out_file, "w", encoding="utf-8") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Built {len(packs)} packs -> {out_file}")
    for p in packs:
        tr = len(p["train_examples"])
        ho = len(p["heldout_tasks"])
        print(f"  {p['pack_id']}: train={tr}, heldout={ho}")


if __name__ == "__main__":
    main()
