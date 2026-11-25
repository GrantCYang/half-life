"""Endpoint probability metrics with explicit row, take and group weighting."""

from __future__ import annotations

from collections import defaultdict
import random
import numpy as np


def _validated(p, y):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if p.ndim != 1 or y.shape != p.shape or not len(p):
        raise ValueError("p and y must be equal nonempty one-dimensional arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("predictions must be finite probabilities")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be 0 or 1; exclude masked targets first")
    return p, y


def _ids(values, count, name):
    values = np.asarray(values)
    if values.shape != (count,):
        raise ValueError(f"{name} must have one id per row")
    return values


def brier(p, y) -> float:
    """Pooled row-average squared probability error (lower is better)."""
    p, y = _validated(p, y)
    return float(np.mean((p - y) ** 2))


def take_macro_brier(p, y, takes, mask=None) -> float:
    """Average each take's row Brier, then give each included take equal weight."""
    p, y = _validated(p, y)
    takes = _ids(takes, len(p), "takes")
    mask = np.ones(len(p), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != p.shape or not mask.any():
        raise ValueError("mask must select at least one row")
    return float(
        np.mean(
            [
                np.mean((p[mask & (takes == t)] - y[mask & (takes == t)]) ** 2)
                for t in sorted(set(takes[mask]))
            ]
        )
    )


def pooled_auc(p, y) -> float | None:
    """Pooled ROC AUC: ranking accuracy, half credit for ties; None for one class."""
    p, y = _validated(p, y)
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(p, kind="stable")
    scores, labels = p[order], y[order]
    rank_sum = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        rank_sum += float(labels[start:end].sum()) * (start + 1 + end) / 2
        start = end
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def calibration_bins(p, y, n_bins: int = 10) -> list[dict]:
    """Equal-width pooled bins, with p=1 included in the final bin."""
    p, y = _validated(p, y)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    indices = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = []
    for index in range(n_bins):
        selected = indices == index
        if selected.any():
            bins.append(
                {
                    "bin": index,
                    "count": int(selected.sum()),
                    "weight": float(selected.mean()),
                    "mean_probability": float(p[selected].mean()),
                    "alive_fraction": float(y[selected].mean()),
                }
            )
    return bins


def ece(p, y, n_bins: int = 10) -> float:
    """Pooled expected calibration error; it does not measure discrimination."""
    return sum(
        b["weight"] * abs(b["mean_probability"] - b["alive_fraction"])
        for b in calibration_bins(p, y, n_bins)
    )


def murphy(p, y, n_bins: int = 10) -> dict:
    """Binned pooled Murphy terms; reconstruction is approximate for varying p.

    REL-RES+UNC is the Brier after replacing each p with its bin's mean.
    The exact row Brier is reported separately; neither is take-macro Brier.
    """
    p, y = _validated(p, y)
    bins = calibration_bins(p, y, n_bins)
    mean = float(y.mean())
    reliability = sum(
        b["weight"] * (b["mean_probability"] - b["alive_fraction"]) ** 2 for b in bins
    )
    resolution = sum(b["weight"] * (b["alive_fraction"] - mean) ** 2 for b in bins)
    uncertainty = mean * (1 - mean)
    reconstructed = reliability - resolution + uncertainty
    exact = brier(p, y)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_binned": reconstructed,
        "brier_exact": exact,
        "binning_gap": exact - reconstructed,
        "bins": bins,
    }


def paired_group_brier_ci(
    candidate,
    reference,
    y,
    takes,
    groups,
    *,
    n_resamples: int = 10000,
    seed: int = 20260915,
) -> dict:
    """Paired capture-group bootstrap of take-macro Brier reduction.

    Positive means candidate is better. Resample groups with replacement,
    retain all takes in each drawn group, then average take differences.
    Groups containing more takes do NOT get compressed into one equal-weight
    group mean. Percentile order-statistic lookup matches the study script.
    """
    candidate, y = _validated(candidate, y)
    reference, _ = _validated(reference, y)
    takes, groups = _ids(takes, len(y), "takes"), _ids(groups, len(y), "groups")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    by_group = defaultdict(list)
    differences = []
    for take in sorted(set(takes)):
        selected = takes == take
        membership = set(groups[selected])
        if len(membership) != 1:
            raise ValueError("each take must belong to exactly one capture group")
        difference = float(
            np.mean((reference[selected] - y[selected]) ** 2)
            - np.mean((candidate[selected] - y[selected]) ** 2)
        )
        differences.append(difference)
        by_group[next(iter(membership))].append(difference)
    rng, keys, draws = random.Random(seed), list(by_group), []
    for _ in range(n_resamples):
        values = [
            value
            for group in (rng.choice(keys) for _ in keys)
            for value in by_group[group]
        ]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "reduction": float(np.mean(differences)),
        "ci": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]],
        "positive_takes": sum(d > 0 for d in differences),
        "takes": len(differences),
        "groups": len(keys),
        "resamples": n_resamples,
    }


def paired_group_auc_ci(
    candidate, reference, y, groups, *, n_resamples=2000, seed=20260915
) -> dict:
    """Paired group bootstrap for pooled AUC(candidate)-AUC(reference)."""
    candidate, y = _validated(candidate, y)
    reference, _ = _validated(reference, y)
    groups = _ids(groups, len(y), "groups")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    ca, ra = pooled_auc(candidate, y), pooled_auc(reference, y)
    if ca is None:
        return {"difference": None, "ci": None, "valid_resamples": 0}
    keys = sorted(set(groups))
    by_group = {g: np.flatnonzero(groups == g) for g in keys}
    rng, draws = random.Random(seed), []
    for _ in range(n_resamples):
        selected = np.concatenate([by_group[rng.choice(keys)] for _ in keys])
        a, b = (
            pooled_auc(candidate[selected], y[selected]),
            pooled_auc(reference[selected], y[selected]),
        )
        if a is not None:
            draws.append(a - b)
    draws.sort()
    ci = (
        None
        if not draws
        else [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]
    )
    return {"difference": ca - ra, "ci": ci, "valid_resamples": len(draws)}
