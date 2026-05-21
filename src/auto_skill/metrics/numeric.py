"""Numeric summary helpers for experiment metrics."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": _safe_mean(clean),
        "min": min(clean) if clean else None,
        "p50": _percentile(clean, 0.5),
        "max": max(clean) if clean else None,
    }
