"""Causal inventory construction, lexical history and the Forecaster prompt.

No function in this module reads future labels or future observations. Label
attachment happens after queries are built, so masked targets cannot alter
the inventory shown to other queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, Mapping, Any

from halflife.data.schema import QueryExample, TypedObservation

_WORD = re.compile(r"[a-z]+")
_STOP = frozenset(
    "a an the is are was were be been being of in on at to for with and or his her its their this that these those there here as by from into".split()
)


def content_words(text: str) -> frozenset[str]:
    """Frozen lexical signature: lowercase words, no stemming or synonyms."""
    return frozenset(
        w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP
    )


def signature_key(statement: str) -> str:
    return hashlib.sha256(
        "|".join(sorted(content_words(statement))).encode()
    ).hexdigest()[:16]


def lexical_present(
    statement: str, visible: Iterable[str], threshold: float = 0.6
) -> bool:
    query = content_words(statement)
    if not query:
        return False
    for text in visible:
        words = content_words(text)
        if words and len(query & words) / len(query | words) >= threshold:
            return True
    return False


@dataclass(frozen=True)
class MemorySnapshot:
    take_uid: str
    group: str
    anchor_sec: float
    inventory: tuple[str, ...]
    history: tuple[TypedObservation, ...]
    history_window: float = 30.0
    match_threshold: float = 0.6

    def profile(
        self, statement: str
    ) -> tuple[float | None, float, TypedObservation | None]:
        seen = [
            o
            for o in self.history
            if lexical_present(statement, o.visible, self.match_threshold)
        ]
        last = seen[-1] if seen else None
        return (
            None if last is None else round(self.anchor_sec - last.time_sec, 2),
            len(seen) / len(self.history) if self.history else 0.0,
            last,
        )


def build_memory(
    observations: Iterable[TypedObservation | Mapping[str, Any]],
    anchor_sec: float,
    *,
    inventory_lookback: float = 1.0,
    history_window: float = 30.0,
    max_facts: int = 12,
    match_threshold: float = 0.6,
) -> MemorySnapshot:
    """Build one take's memory using observations with timestamp <= anchor.

    Inventory: chronological first occurrence of each signature in the final
    lookback, capped before rendering. Presence denominator: observed clips
    in the history window, including clips with an empty visible list.
    """
    if (
        not math.isfinite(anchor_sec)
        or anchor_sec < 0
        or inventory_lookback < 0
        or history_window <= 0
        or inventory_lookback > history_window
        or max_facts < 1
        or not 0 < match_threshold <= 1
    ):
        raise ValueError("invalid memory window, anchor, cap or match threshold")
    rows = [
        o if isinstance(o, TypedObservation) else TypedObservation.from_dict(o)
        for o in observations
    ]
    if not rows or len({o.take_uid for o in rows}) != 1:
        raise ValueError("build_memory requires observations from exactly one take")
    groups = {o.group for o in rows if o.group}
    if len(groups) > 1:
        raise ValueError("a take cannot belong to multiple capture groups")
    past = sorted(
        (o for o in rows if anchor_sec - history_window <= o.time_sec <= anchor_sec),
        key=lambda o: o.time_sec,
    )
    if len({o.time_sec for o in past}) != len(past):
        raise ValueError("duplicate observation timestamps in one take")
    inventory, seen = [], set()
    for obs in past:
        if obs.time_sec < anchor_sec - inventory_lookback:
            continue
        for statement in obs.visible:
            key = content_words(statement)
            if key not in seen:
                seen.add(key)
                inventory.append(statement)
    return MemorySnapshot(
        rows[0].take_uid,
        next(iter(groups), ""),
        float(anchor_sec),
        tuple(inventory[:max_facts]),
        tuple(past),
        history_window,
        match_threshold,
    )


def render_prompt(snapshot: MemorySnapshot, statement: str, delta_sec: float) -> str:
    """Render the text-only forecast input; no label is accepted here."""
    lines = []
    for index, text in enumerate(sorted(snapshot.inventory), 1):
        off, fraction, _ = snapshot.profile(text)
        age = "?" if off is None else str(off)
        lines.append(f"{index}. [{age}s ago, seen {round(fraction * 100)}%] {text}")
    _, _, last = snapshot.profile(statement)
    motion = " ".join(last.motion) if last and last.motion else "[none]"
    inferred = " ".join(last.inferred) if last and last.inferred else "[none]"
    delta = f"{delta_sec:g}"
    return (
        "You hold a memory of a scene observed until now. Facts currently in memory, each with seconds "
        f"since last observed and the fraction of the last {snapshot.history_window:g} seconds it was observed:\n"
        + "\n".join(lines)
        + "\n"
        f"At the queried fact's last observation, motion: {motion}\n"
        f"Inferred context then: {inferred}\n"
        f"The camera now goes dark for {delta} seconds.\n"
        f'After those {delta} seconds, is this fact still true: "{statement}"?\n'
        "Answer with exactly one word: ALIVE or DEAD."
    )


def make_query_example(
    snapshot: MemorySnapshot,
    statement: str,
    delta_sec: float,
    *,
    label: str = "masked",
    source: str = "unknown",
    split: str = "",
    is_fit: bool = False,
    volatile: bool | None = None,
) -> QueryExample:
    if statement not in snapshot.inventory:
        raise ValueError("query statement must be selected from the causal inventory")
    off, fraction, last = snapshot.profile(statement)
    sig = signature_key(statement)
    identity = f"{snapshot.take_uid}|{snapshot.anchor_sec:.6f}|{delta_sec:.6f}|{sig}"
    return QueryExample(
        example_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
        take_uid=snapshot.take_uid,
        group=snapshot.group,
        anchor_sec=snapshot.anchor_sec,
        delta_sec=float(delta_sec),
        statement=statement,
        sig_key=sig,
        prompt=render_prompt(snapshot, statement, delta_sec),
        label=label,
        source=source,
        split=split,
        is_fit=is_fit,
        volatile=volatile,
        last_seen_off=off,
        presence_frac=round(fraction, 4),
        ctx_visible=" ".join(last.visible[:8]) if last and last.visible else "[none]",
        ctx_motion=" ".join(last.motion) if last and last.motion else "[none]",
        ctx_inferred=" ".join(last.inferred) if last and last.inferred else "[none]",
    )


def build_examples(
    observations: Iterable[TypedObservation | Mapping[str, Any]],
    anchors: Iterable[float],
    horizons: Iterable[float] = (8, 32, 64, 120),
    *,
    split: str = "",
    **memory_options: Any,
) -> list[QueryExample]:
    """Create unlabeled queries for one take. Attach future evidence separately."""
    rows, deltas = list(observations), tuple(horizons)
    examples = []
    for anchor in sorted(set(anchors)):
        snapshot = build_memory(rows, anchor, **memory_options)
        for statement in snapshot.inventory:
            for delta in deltas:
                examples.append(
                    make_query_example(snapshot, statement, delta, split=split)
                )
    return examples
