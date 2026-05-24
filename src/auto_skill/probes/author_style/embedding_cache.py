"""Cached sentence embeddings for author-style probe construction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def model_slug(model_id: str) -> str:
    return safe_slug(model_id)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip("-")


def truncate_for_embedding(text: str, max_chars: int) -> str:
    return text if max_chars <= 0 or len(text) <= max_chars else text[:max_chars]


def embedding_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_cache_slug(*, backend: str, model_id: str, dim: int | None = None) -> str:
    suffix = f"--dim{dim}" if dim else "--dim_unknown"
    return f"{safe_slug(backend)}--{model_slug(model_id)}{suffix}"


def embedding_path(cache_root: Path, slug: str, sha: str) -> Path:
    return cache_root / slug / sha[:2] / f"{sha}.npy"


def package_versions() -> dict[str, str | None]:
    return {
        name: version_or_none(name)
        for name in ("sentence-transformers", "transformers", "torch", "numpy", "openai", "httpx")
    }


def version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_sentence_transformer(model_id: str, *, device: str, max_seq_length: int) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id,
        device=device,
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = max_seq_length
    return model


def encode_batch(model: Any, texts: list[str], *, batch_size: int) -> Any:
    kwargs = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    if hasattr(model, "encode_document"):
        return model.encode_document(texts, **kwargs)
    return model.encode(texts, prompt_name="document", **kwargs)


def embed_with_cache(
    texts: list[str],
    *,
    cache_root: Path,
    model_id: str,
    backend: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    max_chars: int,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    expected_dim: int | None = None,
    max_retries: int = 3,
    workers: int = 1,
    retry_sleep: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    slug = embedding_cache_slug(backend=backend, model_id=model_id, dim=expected_dim)
    truncated = [truncate_for_embedding(text, max_chars) for text in texts]
    shas = [embedding_sha(text) for text in truncated]
    vectors: dict[str, Any] = {}
    missing: dict[str, str] = {}
    cache_hits = 0
    for sha, text in zip(shas, truncated, strict=True):
        path = embedding_path(cache_root, slug, sha)
        if sha in vectors:
            continue
        if path.exists():
            vectors[sha] = np.load(path)
            cache_hits += 1
        else:
            missing[sha] = text
    cache_misses = len(missing)
    if missing:
        items = list(missing.items())
        texts_to_encode = [text for _, text in items]
        if backend == "openai":
            encoded = encode_openai(
                texts_to_encode,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                batch_size=batch_size,
                expected_dim=expected_dim,
                max_retries=max_retries,
                workers=workers,
                retry_sleep=retry_sleep,
            )
        elif backend == "sentence-transformers":
            model = load_sentence_transformer(model_id, device=device, max_seq_length=max_seq_length)
            encoded = encode_batch(model, texts_to_encode, batch_size=batch_size)
        else:
            raise ValueError(f"unsupported embedding backend: {backend}")
        if expected_dim is not None and int(encoded.shape[1]) != expected_dim:
            raise ValueError(f"embedding dim mismatch: expected {expected_dim}, got {encoded.shape[1]}")
        assert_unit_norm(encoded)
        for (sha, _), vector in zip(items, encoded, strict=True):
            path = embedding_path(cache_root, slug, sha)
            path.parent.mkdir(parents=True, exist_ok=True)
            vector = vector.astype("float32", copy=False)
            np.save(path, vector)
            vectors[sha] = vector
    matrix = np.vstack([vectors[sha] for sha in shas]).astype("float32", copy=False)
    if expected_dim is not None and int(matrix.shape[1]) != expected_dim:
        raise ValueError(f"cached embedding dim mismatch: expected {expected_dim}, got {matrix.shape[1]}")
    assert_unit_norm(matrix)
    device_value = None if backend == "openai" else device
    max_seq_length_value = None if backend == "openai" else max_seq_length
    summary = {
        "model": model_id,
        "model_slug": slug,
        "cache_namespace_policy": "backend+model_id+dim",
        "backend": backend,
        "base_url": base_url,
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "device": device_value,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length_value,
        "truncate_chars": max_chars,
        "cache_root": str(cache_root),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "expected_dim": expected_dim,
        "max_retries": max_retries,
        "workers": workers,
        "package_versions": package_versions(),
    }
    return matrix, summary


def assert_unit_norm(matrix: Any, *, atol: float = 1e-3) -> None:
    import numpy as np

    if not getattr(matrix, "size", 0):
        return
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError(
            "embedding vectors must be L2-normalized for dot-product cosine; "
            f"observed norm range [{float(norms.min()):.6f}, {float(norms.max()):.6f}]"
        )


def encode_openai(
    texts: list[str],
    *,
    model_id: str,
    base_url: str | None,
    api_key: str,
    batch_size: int,
    expected_dim: int | None,
    max_retries: int,
    workers: int,
    retry_sleep: float,
) -> Any:
    import httpx
    import numpy as np
    from openai import OpenAI

    if not base_url:
        raise ValueError("base_url is required for openai embedding backend")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(trust_env=False, timeout=120.0),
    )
    batches = [(start, texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)]
    results: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(create_embeddings_with_retry, client, model_id=model_id, batch=batch, max_retries=max_retries, retry_sleep=retry_sleep): start
            for start, batch in batches
        }
        for future in as_completed(futures):
            start = futures[future]
            response = future.result()
            results[start] = [row.embedding for row in sorted(response.data, key=lambda row: row.index)]
    vectors = [vector for start, _ in batches for vector in results[start]]
    if expected_dim is not None:
        for vector in vectors:
            if len(vector) != expected_dim:
                raise ValueError(f"embedding dim mismatch: expected {expected_dim}, got {len(vector)}")
    return np.asarray(vectors, dtype="float32")


def create_embeddings_with_retry(client: Any, *, model_id: str, batch: list[str], max_retries: int, retry_sleep: float) -> Any:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.embeddings.create(model=model_id, input=batch)
        except Exception as exc:  # noqa: BLE001 - retry transport/API flakes, fail loud after budget.
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt))
    raise RuntimeError(f"embedding request failed after {max_retries + 1} attempts: {last_exc}") from last_exc
