"""Standalone B0-B4 controls for endpoint validity forecasting.

All fit methods accept TRAINING-SIDE rows only; callers own split selection.
Supervised fitting excludes masked targets and, by default, the held-out
64-second horizon. Optional feature_rows fit unsupervised vocabulary/SVD
on a separately declared training-side input population. Calibration is a
separate operation in halflife.forecast.calibration, never fitted during prediction.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable
import numpy as np

from halflife.data.memory import content_words
from halflife.data.schema import QueryExample, normalize_label

TRAINED_HORIZONS = (8.0, 32.0, 120.0)
SEMANTIC_FIELDS = ("statement", "ctx_visible", "ctx_motion", "ctx_inferred")


def _rows(rows):
    return [r.to_dict() if isinstance(r, QueryExample) else dict(r) for r in rows]


def supervised_rows(rows, trained_horizons=TRAINED_HORIZONS):
    """Keep usable labels at declared horizons; never silently label unknown dead."""
    result = []
    for row in _rows(rows):
        label = normalize_label(row.get("label"))
        if label != "masked" and float(row["delta_sec"]) in trained_horizons:
            result.append(dict(row, label=label))
    if not result:
        raise ValueError("no usable training rows at the trained horizons")
    return result


def log_time_interpolate(horizon: float, values: dict[float, float]) -> float:
    """Piecewise linear interpolation in log(time); no extrapolation."""
    if horizon in values:
        return float(values[horizon])
    knots = sorted(values)
    if (
        not math.isfinite(horizon)
        or horizon <= 0
        or horizon < knots[0]
        or horizon > knots[-1]
    ):
        raise ValueError("horizon lies outside fitted positive-time support")
    high_index = next(i for i, knot in enumerate(knots) if knot > horizon)
    low, high = knots[high_index - 1], knots[high_index]
    weight = (math.log(horizon) - math.log(low)) / (math.log(high) - math.log(low))
    return (1 - weight) * values[low] + weight * values[high]


class BaseRate:
    """B0: empirical ALIVE rate at each trained horizon, log-time interpolation."""

    def __init__(self, horizons=TRAINED_HORIZONS):
        self.horizons = tuple(float(d) for d in horizons)
        if not self.horizons or any(
            not math.isfinite(d) or d <= 0 for d in self.horizons
        ):
            raise ValueError("horizons must be positive")

    def fit(self, rows) -> "BaseRate":
        rows = supervised_rows(rows, self.horizons)
        self.rates_ = {}
        for horizon in self.horizons:
            labels = [
                r["label"] == "alive" for r in rows if float(r["delta_sec"]) == horizon
            ]
            if not labels:
                raise ValueError(f"no usable rows at horizon {horizon}")
            self.rates_[horizon] = sum(labels) / len(labels)
        return self

    def predict(self, rows) -> np.ndarray:
        if not hasattr(self, "rates_"):
            raise ValueError("baseline has not been fitted")
        return np.asarray(
            [
                log_time_interpolate(float(r["delta_sec"]), self.rates_)
                for r in _rows(rows)
            ]
        )


class WordClock(BaseRate):
    """B1: per-word ALIVE counts shrunk toward B0, then averaged per statement.

    Each word counts once per row. Keep words observed at least min_count
    times at that horizon; q=(alive_count + shrinkage*B0)/(count+shrinkage).
    No surviving word falls back to B0. This table is NOT the older caption
    re-emission table used to define the volatile stratum.
    """

    def __init__(self, horizons=TRAINED_HORIZONS, *, min_count=20, shrinkage=20.0):
        super().__init__(horizons)
        if min_count < 1 or shrinkage < 0:
            raise ValueError("invalid minimum count or shrinkage")
        self.min_count, self.shrinkage = min_count, float(shrinkage)

    def fit(self, rows) -> "WordClock":
        rows = supervised_rows(rows, self.horizons)
        super().fit(rows)
        self.tables_ = {}
        for horizon in self.horizons:
            counts, alive = Counter(), Counter()
            for row in rows:
                if float(row["delta_sec"]) != horizon:
                    continue
                words = content_words(row["statement"])
                counts.update(words)
                if row["label"] == "alive":
                    alive.update(words)
            self.tables_[horizon] = {
                word: (alive[word] + self.shrinkage * self.rates_[horizon])
                / (count + self.shrinkage)
                for word, count in counts.items()
                if count >= self.min_count
            }
        return self

    def predict(self, rows) -> np.ndarray:
        if not hasattr(self, "tables_"):
            raise ValueError("baseline has not been fitted")
        predictions = []
        for row in _rows(rows):
            words, rates = content_words(row["statement"]), {}
            for horizon in self.horizons:
                values = [
                    self.tables_[horizon][w]
                    for w in sorted(words)
                    if w in self.tables_[horizon]
                ]
                rates[horizon] = (
                    sum(values) / len(values) if values else self.rates_[horizon]
                )
            predictions.append(log_time_interpolate(float(row["delta_sec"]), rates))
        return np.asarray(predictions)


class LexicalFeatures:
    """B2 matrix: presence, last-seen/30, log(delta), binary content-word BOW."""

    def __init__(self, min_count=20):
        if min_count < 1:
            raise ValueError("min_count must be positive")
        self.min_count = min_count

    def fit(self, rows) -> "LexicalFeatures":
        rows = _rows(rows)
        if not rows:
            raise ValueError("empty feature-fit population")
        counts = Counter(w for r in rows for w in content_words(r["statement"]))
        self.vocabulary_ = tuple(
            sorted(w for w, n in counts.items() if n >= self.min_count)
        )
        self.word_index_ = {word: i for i, word in enumerate(self.vocabulary_)}
        return self

    def transform(self, rows) -> np.ndarray:
        if not hasattr(self, "word_index_"):
            raise ValueError("features have not been fitted")
        rows = _rows(rows)
        matrix = np.zeros((len(rows), 3 + len(self.vocabulary_)), dtype=float)
        for index, row in enumerate(rows):
            delta = float(row["delta_sec"])
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError("delta_sec must be positive")
            offset = row.get("last_seen_off")
            matrix[index, :3] = [
                float(row["presence_frac"]),
                (30.0 if offset is None else float(offset)) / 30.0,
                math.log(delta),
            ]
            for word in content_words(row["statement"]):
                if word in self.word_index_:
                    matrix[index, 3 + self.word_index_[word]] = 1.0
        if not np.isfinite(matrix).all():
            raise ValueError("feature values must be finite")
        return matrix


def masked_mean_pool(hidden_states, attention_mask) -> np.ndarray:
    """MiniLM-style token mean followed by per-text L2 normalization.

    Accept NumPy arrays [batch,tokens,dim] and [batch,tokens]. Tokenizer/model
    execution is optional and outside this CPU-only helper.
    """
    states, mask = np.asarray(hidden_states), np.asarray(attention_mask, dtype=float)
    if states.ndim != 3 or mask.shape != states.shape[:2] or (mask < 0).any():
        raise ValueError("invalid hidden states or attention mask shape")
    mean = (states * mask[..., None]).sum(axis=1) / np.maximum(
        mask.sum(axis=1, keepdims=True), 1e-9
    )
    return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)


class SemanticFeatures:
    """B3 = B2 plus four separate centered, unwhitened SVD projections.

    encoder(texts) returns frozen sentence embeddings [n,dim], e.g. MiniLM
    masked-mean/L2 outputs. Each of statement/visible/motion/inferred receives
    its own training mean and up to 24 principal directions (96 added dims).
    The encoder itself is never trained by this class.
    """

    def __init__(self, encoder: Callable, *, dim=24, min_count=20):
        if dim < 1:
            raise ValueError("projection dimension must be positive")
        self.encoder, self.dim = encoder, dim
        self.lexical = LexicalFeatures(min_count)

    def _embeddings(self, rows):
        texts = list(
            dict.fromkeys(
                str(r.get(field, "[none]")) for r in rows for field in SEMANTIC_FIELDS
            )
        )
        embeddings = np.asarray(self.encoder(texts), dtype=float)
        if (
            embeddings.ndim != 2
            or len(embeddings) != len(texts)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("encoder must return one finite vector per text")
        lookup = {text: embeddings[i] for i, text in enumerate(texts)}
        return [
            np.asarray([lookup[str(r.get(field, "[none]"))] for r in rows])
            for field in SEMANTIC_FIELDS
        ]

    def fit(self, rows) -> "SemanticFeatures":
        rows = _rows(rows)
        self.lexical.fit(rows)
        self.projections_ = []
        for matrix in self._embeddings(rows):
            mean = matrix.mean(axis=0)
            _, _, directions = np.linalg.svd(matrix - mean, full_matrices=False)
            self.projections_.append((mean, directions[: self.dim].T))
        return self

    def transform(self, rows) -> np.ndarray:
        if not hasattr(self, "projections_"):
            raise ValueError("features have not been fitted")
        rows = _rows(rows)
        if not rows:
            return np.empty(
                (
                    0,
                    3
                    + len(self.lexical.vocabulary_)
                    + sum(v.shape[1] for _, v in self.projections_),
                )
            )
        projected = [
            (matrix - mean) @ directions
            for matrix, (mean, directions) in zip(
                self._embeddings(rows), self.projections_
            )
        ]
        return np.hstack([self.lexical.transform(rows)] + projected)


def ridge_fit(features, targets, alpha: float = 1.0) -> np.ndarray:
    """Return weights followed by an UNPENALIZED intercept."""
    matrix, targets = (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
    )
    if matrix.ndim != 2 or targets.shape != (len(matrix),) or not len(matrix):
        raise ValueError("invalid ridge input dimensions")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(targets).all()
        or not math.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("finite ridge inputs and nonnegative alpha required")
    augmented = np.column_stack([matrix, np.ones(len(matrix))])
    if alpha == 0:
        return np.linalg.lstsq(augmented, targets, rcond=None)[0]
    system = augmented.T @ augmented + alpha * np.eye(augmented.shape[1])
    system[-1, -1] -= alpha
    return np.linalg.solve(system, augmented.T @ targets)


def ridge_predict(features, weights, *, clip: bool = False) -> np.ndarray:
    """Raw predictions are used for inner alpha selection; final probabilities clip."""
    matrix, weights = (
        np.asarray(features, dtype=float),
        np.asarray(weights, dtype=float),
    )
    prediction = matrix @ weights[:-1] + weights[-1]
    return np.clip(prediction, 0, 1) if clip else prediction


def select_ridge_alpha(
    train_x,
    train_y,
    validation_x,
    validation_y,
    grid=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
) -> float:
    """Choose alpha by raw validation MSE; callers supply disjoint group splits."""
    if len(validation_y) == 0 or not grid:
        raise ValueError("nonempty validation labels and alpha grid required")
    return min(
        grid,
        key=lambda alpha: float(
            np.mean(
                (
                    np.asarray(validation_y)
                    - ridge_predict(validation_x, ridge_fit(train_x, train_y, alpha))
                )
                ** 2
            )
        ),
    )


class LexicalRidge:
    """B2 fitted on lexical/history features. Predict returns uncalibrated p."""

    def __init__(self, alpha=1.0, *, min_count=20, horizons=TRAINED_HORIZONS):
        self.alpha, self.horizons = alpha, tuple(horizons)
        self.features = LexicalFeatures(min_count)

    def fit(self, rows, *, feature_rows=None):
        rows = _rows(rows)
        usable = supervised_rows(rows, self.horizons)
        self.features.fit(rows if feature_rows is None else feature_rows)
        self.weights_ = ridge_fit(
            self.features.transform(usable),
            [r["label"] == "alive" for r in usable],
            self.alpha,
        )
        return self

    def predict(self, rows):
        if not hasattr(self, "weights_"):
            raise ValueError("baseline has not been fitted")
        return ridge_predict(self.features.transform(rows), self.weights_, clip=True)


class SemanticRidge(LexicalRidge):
    """B3 linear semantic control on the exact B3 feature matrix."""

    def __init__(
        self, encoder, alpha=1.0, *, dim=24, min_count=20, horizons=TRAINED_HORIZONS
    ):
        self.alpha, self.horizons = alpha, tuple(horizons)
        self.features = SemanticFeatures(encoder, dim=dim, min_count=min_count)


class GradientBoostedBaseline:
    """B4 nonlinear control. Accept a FITTED B3 feature builder to share X exactly.

    scikit-learn is imported only in fit. The caller can pass SemanticRidge's
    .features; this class never refits vocabulary or SVD on evaluation rows.
    """

    def __init__(
        self, features: SemanticFeatures, *, horizons=TRAINED_HORIZONS, **model_options
    ):
        self.features, self.horizons = features, tuple(horizons)
        self.options = dict(
            max_iter=400,
            learning_rate=0.08,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=20260915,
            early_stopping=False,
        )
        self.options.update(model_options)

    def fit(self, rows):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise ImportError(
                "B4 requires the optional scikit-learn dependency"
            ) from exc
        usable = supervised_rows(rows, self.horizons)
        self.model_ = HistGradientBoostingClassifier(**self.options)
        self.model_.fit(
            self.features.transform(usable), [r["label"] == "alive" for r in usable]
        )
        return self

    def predict(self, rows):
        if not hasattr(self, "model_"):
            raise ValueError("baseline has not been fitted")
        return self.model_.predict_proba(self.features.transform(rows))[:, 1]


B0, B1, B2, B3, B4 = (
    BaseRate,
    WordClock,
    LexicalRidge,
    SemanticRidge,
    GradientBoostedBaseline,
)
