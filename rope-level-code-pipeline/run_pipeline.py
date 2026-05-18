#!/usr/bin/env python3
"""
一键运行完整 pipeline
"""
import subprocess
import sys

steps = [
    ("Step 1: Fetch PRs", "python -m src.fetch_prs"),
    ("Step 2: Fetch Details", "python -m src.fetch_details"),
    ("Step 3: Slice Threads", "python -m src.slice_threads"),
    ("Step 4: Classify", "python -m src.classify_threads"),
    ("Step 5: Filter", "python -m src.filter_threads"),
    ("Step 6: Build Packs", "python -m src.build_packs"),
    ("Step 7: Audit", "python -m src.audit"),
]

for name, cmd in steps:
    print(f"\n{'=' * 60}")
    print(f">>> {name}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {name} failed!")
        sys.exit(1)

print("\n>>> All steps completed!")
