"""New tie-correct isotonic fits and an explicit historical replay mode.

Use Isotonic for new experiments. LegacyIsotonic intentionally preserves the
published algorithm's unaggregated equal-score behavior and step lookup.
Choosing the new fit does not retroactively change archived study results.
Fit calibration on training-side labels only and freeze before evaluation.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping
import numpy as np


def _xy(x: Iterable[float], y: Iterable[float]):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("x and y must be equal nonempty one-dimensional arrays")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or ((y < 0) | (y > 1)).any()
    ):
        raise ValueError("finite scores and targets in [0, 1] are required")
    return x, y


def _query(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("prediction scores must be finite")
    return x


class Isotonic:
    """Weighted least-squares isotonic regression with equal-score aggregation.

    Predictions linearly interpolate unique observed-score knots and clip to
    fitted endpoint values outside that range. Tied x values share one fitted
    value regardless of input ordering. This is a new-fit algorithm, not a
    replacement for LegacyIsotonic when reproducing a frozen result.
    """

    def fit(self, x, y, sample_weight=None) -> "Isotonic":
        x, y = _xy(x, y)
        weights = (
            np.ones(len(x))
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if (
            weights.shape != x.shape
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("sample weights must be finite and strictly positive")
        self.x_, inverse = np.unique(x, return_inverse=True)
        totals = np.bincount(inverse, weights=weights)
        sums = np.bincount(inverse, weights=weights * y)
        # Blocks store [left unique index, right unique index, weight, y sum].
        blocks = []
        for index, (weight, total) in enumerate(zip(totals, sums)):
            blocks.append([index, index, float(weight), float(total)])
            while (
                len(blocks) > 1
                and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]
            ):
                right, left = blocks.pop(), blocks.pop()
                blocks.append(
                    [left[0], right[1], left[2] + right[2], left[3] + right[3]]
                )
        self.y_ = np.empty(len(self.x_))
        for left, right, weight, total in blocks:
            self.y_[left : right + 1] = total / weight
        return self

    def predict(self, x):
        if not hasattr(self, "x_"):
            raise ValueError("calibrator has not been fitted")
        return np.interp(_query(x), self.x_, self.y_)

    def to_dict(self) -> dict[str, Any]:
        if not hasattr(self, "x_"):
            raise ValueError("calibrator has not been fitted")
        return {
            "algorithm": "isotonic_ties_linear_v1",
            "x": self.x_.tolist(),
            "y": self.y_.tolist(),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Isotonic":
        if row.get("algorithm") != "isotonic_ties_linear_v1":
            raise ValueError("not a tie-correct isotonic map")
        instance = cls()
        instance.x_, instance.y_ = _xy(row["x"], row["y"])
        if (np.diff(instance.x_) <= 0).any() or (np.diff(instance.y_) < 0).any():
            raise ValueError("invalid isotonic map")
        return instance


class LegacyIsotonic:
    """Exact legacy PAV and left-search step lookup for historical replay.

    Equal input scores are deliberately NOT aggregated. Their ordering can
    affect the fitted map; retaining this behavior is part of replay fidelity.
    The default NumPy argsort and 1e-12 merge tolerance match the old script.
    """

    def fit(self, x, y) -> "LegacyIsotonic":
        x, y = _xy(x, y)
        order = np.argsort(x)
        xs, ys = x[order], y[order].astype(float)
        vals, wts, cnt = list(ys), list(np.ones(len(ys))), [1] * len(ys)
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1] + 1e-12:
                total = wts[i] + wts[i + 1]
                merged = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / total
                vals[i : i + 2] = [merged]
                wts[i : i + 2] = [total]
                cnt[i : i + 2] = [cnt[i] + cnt[i + 1]]
                i = max(i - 1, 0)
            else:
                i += 1
        bx, index = [], 0
        for count in cnt:
            bx.append(xs[index + count - 1])
            index += count
        self.bx_, self.bv_ = np.asarray(bx), np.asarray(vals)
        return self

    def predict(self, x):
        if not hasattr(self, "bx_"):
            raise ValueError("calibrator has not been fitted")
        indices = np.searchsorted(self.bx_, _query(x), side="left")
        return self.bv_[np.clip(indices, 0, len(self.bv_) - 1)]

    def to_dict(self) -> dict[str, Any]:
        if not hasattr(self, "bx_"):
            raise ValueError("calibrator has not been fitted")
        return {
            "algorithm": "legacy_pav_step_v1",
            "bx": self.bx_.tolist(),
            "bv": self.bv_.tolist(),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "LegacyIsotonic":
        if row.get("algorithm", "legacy_pav_step_v1") != "legacy_pav_step_v1":
            raise ValueError("not a legacy PAV map")
        instance = cls()
        instance.bx_, instance.bv_ = _xy(row["bx"], row["bv"])
        if (np.diff(instance.bx_) < 0).any() or (np.diff(instance.bv_) < -1e-12).any():
            raise ValueError("invalid legacy isotonic map")
        return instance
