"""Build JSONL artifacts for author-style T2 probes."""

from __future__ import annotations

import argparse
from typing import Any

from auto_skill.probes.author_style.negatives import (
    NegativePool,
    build_negative_pool,
    select_negatives,
)
from auto_skill.probes.author_style.outputs import manifest_row
from auto_skill.probes.author_style.pairwise_jobs import add_cross_pairwise, add_within_pairwise
from auto_skill.probes.author_style.records import negative_row, private_eval_row, public_pack
from auto_skill.probes.author_style.sampling import (
    VARIANTS,
    bucket,
    bucket_names,
    choose_posts,
    select_authors,
    stable_key,
    stable_order,
)


def build_probe(
    *,
    blog_posts: list[dict[str, Any]],
    reddit_posts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    selected = select_probe_authors(blog_posts=blog_posts, reddit_posts=reddit_posts, args=args)
    selected_posts = choose_probe_posts(selected, args=args)
    style_ids = style_id_map(selected_posts, seed=args.seed)
    pools = {
        "blog": build_negative_pool(blog_posts, seed=args.seed),
        "reddit": build_negative_pool(reddit_posts, seed=args.seed),
    }
    packs: list[dict[str, Any]] = []
    private_eval: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    pairwise_jobs: list[dict[str, Any]] = []
    add_author_oracles(
        packs,
        private_eval,
        negatives,
        pairwise_jobs,
        selected_posts,
        pools,
        style_ids,
        args,
    )
    add_length_bucket_oracles(packs, private_eval, negatives, selected, pools["blog"], args)
    add_cross_pairwise(
        pairwise_jobs,
        selected_posts,
        seed=args.seed,
        cross_pairs_per_corpus=args.cross_pairs_per_corpus,
    )
    manifest = manifest_row(
        selected_posts,
        packs,
        negatives,
        pairwise_jobs,
        args,
        style_ids=style_ids,
    )
    private_manifest = manifest_row(
        selected_posts,
        packs,
        negatives,
        pairwise_jobs,
        args,
        style_ids=style_ids,
        include_author_hash=True,
    )
    return packs, private_eval, negatives, pairwise_jobs, manifest, private_manifest


def select_probe_authors(
    *,
    blog_posts: list[dict[str, Any]],
    reddit_posts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        "blog": select_authors(
            blog_posts,
            corpus="blog",
            count=args.blog_authors,
            posts_per_author=args.posts_per_author,
            seed=args.seed,
            blog_min_buckets=args.blog_min_buckets,
        ),
        "reddit": select_authors(
            reddit_posts,
            corpus="reddit",
            count=args.reddit_authors,
            posts_per_author=args.posts_per_author,
            seed=args.seed,
            blog_min_buckets=args.blog_min_buckets,
        ),
    }


def choose_probe_posts(
    selected: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    args: argparse.Namespace,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        corpus: {
            author: choose_posts(author, posts, n=args.posts_per_author, seed=args.seed)
            for author, posts in grouped.items()
        }
        for corpus, grouped in selected.items()
    }


def style_id_map(
    selected_posts: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    seed: int,
) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    for corpus, grouped in selected_posts.items():
        authors = sorted(grouped, key=lambda author: stable_key(seed, "style-id", corpus, author))
        aliases[corpus] = {
            author: f"{corpus}_{index:03d}" for index, author in enumerate(authors, start=1)
        }
    return aliases


def add_author_oracles(
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pairwise_jobs: list[dict[str, Any]],
    selected_posts: dict[str, dict[str, list[dict[str, Any]]]],
    pools: dict[str, NegativePool],
    style_ids: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> None:
    for corpus, grouped in selected_posts.items():
        for author, posts in grouped.items():
            for index in range(min(args.triples_per_author, len(posts) - 1)):
                for variant in VARIANTS:
                    pack_id = f"probe_{style_ids[corpus][author]}_{index:02d}_{variant}"
                    add_oracle_pack(
                        packs,
                        private_eval,
                        negatives,
                        pack_id=pack_id,
                        corpus=corpus,
                        target=posts[index],
                        reference=posts[(index + 1) % len(posts)],
                        negs=select_negatives(
                            posts[index],
                            pools[corpus],
                            variant=variant,
                            count=args.negatives_per_triple,
                            seed=args.seed,
                            salt=pack_id,
                        ),
                        variant=variant,
                    )
            add_within_pairwise(pairwise_jobs, corpus=corpus, author=author, posts=posts)


def add_oracle_pack(
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    pack_id: str,
    corpus: str,
    target: dict[str, Any],
    reference: dict[str, Any],
    negs: list[dict[str, Any]],
    variant: str,
) -> None:
    task_id = f"{pack_id}::heldout::0"
    packs.append(public_pack(pack_id, corpus, task_id, target, reference, variant))
    private_eval.append(private_eval_row(pack_id, task_id, target))
    for rank, neg in enumerate(negs, start=1):
        negatives.append(negative_row(pack_id, task_id, rank, target, neg, variant))


def add_length_bucket_oracles(
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    selected: dict[str, dict[str, list[dict[str, Any]]]],
    blog_pool: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    for name in bucket_names():
        eligible = bucket_eligible_authors(selected["blog"], name)
        if not eligible:
            continue
        authors = sorted(eligible, key=lambda a: stable_key(args.seed, "len", name, a))
        for index in range(args.triples_per_bucket):
            author = authors[index % len(authors)]
            posts = stable_order(eligible[author], args.seed, "len", name, author)
            pack_id = f"probe_blog_length_{name}_{index:04d}"
            add_oracle_pack(
                packs,
                private_eval,
                negatives,
                pack_id=pack_id,
                corpus="blog",
                target=posts[index % len(posts)],
                reference=posts[(index + 1) % len(posts)],
                negs=select_negatives(
                    posts[index % len(posts)],
                    blog_pool,
                    variant="length",
                    count=args.negatives_per_triple,
                    seed=args.seed,
                    salt=pack_id,
                ),
                variant="length_bucket",
            )


def bucket_eligible_authors(
    blog_authors: dict[str, list[dict[str, Any]]],
    bucket_name: str,
) -> dict[str, list[dict[str, Any]]]:
    eligible = {
        author: [post for post in posts if bucket(int(post["word_count"])) == bucket_name]
        for author, posts in blog_authors.items()
    }
    return {author: posts for author, posts in eligible.items() if len(posts) >= 2}
