"""JSON-compatible records shared by memory, labeling and model execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


def normalize_label(value: Any) -> str:
    """Canonical labels; unknown is masked, never a negative example."""
    if value is None or value == "masked":
        return "masked"
    if value is True or value == 1 or str(value).lower() == "alive":
        return "alive"
    if value is False or value == 0 or str(value).lower() == "dead":
        return "dead"
    raise ValueError(f"Unsupported endpoint label: {value!r}")


@dataclass(frozen=True)
class TypedObservation:
    """One clip observation. Only visible facts populate the memory inventory."""

    take_uid: str
    time_sec: float
    visible: tuple[str, ...]
    motion: tuple[str, ...] = ()
    inferred: tuple[str, ...] = ()
    group: str = ""

    def __post_init__(self):
        if not self.take_uid or not math.isfinite(self.time_sec) or self.time_sec < 0:
            raise ValueError(
                "observation requires a take id and nonnegative finite time"
            )
        for name in ("visible", "motion", "inferred"):
            value = getattr(self, name)
            if isinstance(value, str) or any(not isinstance(s, str) for s in value):
                raise ValueError(f"{name} must be a sequence of strings")
            object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "TypedObservation":
        return cls(
            str(row.get("take_uid", row.get("take_id", ""))),
            float(row.get("time_sec", row.get("time_s", 0))),
            tuple(row.get("visible", ())),
            tuple(row.get("motion", ())),
            tuple(row.get("inferred", ())),
            str(row.get("group", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for name in ("visible", "motion", "inferred"):
            row[name] = list(row[name])
        return row


@dataclass(frozen=True)
class QueryExample:
    """A past-only prompt plus a separately attached future supervision label.

    ``source`` records the labeling route. ``volatile`` is an optional frozen
    lexical-return stratum, not a synonym for DEAD or a baseline prediction.
    """

    example_id: str
    take_uid: str
    anchor_sec: float
    delta_sec: float
    statement: str
    prompt: str
    label: str = "masked"
    source: str = "unknown"
    volatile: bool | None = None
    split: str = ""
    group: str = ""
    sig_key: str = ""
    last_seen_off: float | None = None
    presence_frac: float = 0.0
    ctx_visible: str = "[none]"
    ctx_motion: str = "[none]"
    ctx_inferred: str = "[none]"
    is_fit: bool = False

    def __post_init__(self):
        object.__setattr__(self, "label", normalize_label(self.label))
        if not math.isfinite(self.delta_sec) or self.delta_sec <= 0:
            raise ValueError("delta_sec must be finite and positive")
        if not math.isfinite(self.anchor_sec) or self.anchor_sec < 0:
            raise ValueError("anchor_sec must be finite and nonnegative")
        if not 0 <= self.presence_frac <= 1:
            raise ValueError("presence_frac must be in [0, 1]")
        if self.last_seen_off is not None and (
            not math.isfinite(self.last_seen_off) or self.last_seen_off < 0
        ):
            raise ValueError("last_seen_off must be nonnegative or None")

    @property
    def y(self) -> int | None:
        return {"alive": 1, "dead": 0, "masked": None}[self.label]

    @property
    def target(self) -> str | None:
        return None if self.y is None else self.label.upper()

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "QueryExample":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
