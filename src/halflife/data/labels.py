"""One-sided caption evidence followed by a three-way visual endpoint label."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from halflife.data.memory import content_words
from halflife.data.schema import QueryExample


@dataclass(frozen=True)
class LabelDecision:
    label: str
    source: str


def assign_label(
    *,
    nli_entailment: float | None = None,
    oracle_answer: str | None = None,
    nli_threshold: float = 0.5,
) -> LabelDecision:
    """An entailed future caption proves ALIVE; non-entailment proves nothing.

    Otherwise the visual instrument must answer still_true/no_longer_true.
    Cannot-tell, absent and malformed responses remain masked.
    """
    if not 0 <= nli_threshold <= 1:
        raise ValueError("nli_threshold must be in [0, 1]")
    if nli_entailment is not None:
        if not math.isfinite(nli_entailment) or not 0 <= nli_entailment <= 1:
            raise ValueError("nli_entailment must be a probability or None")
        if nli_entailment >= nli_threshold:
            return LabelDecision("alive", "caption_return")
    answer = (
        "" if oracle_answer is None else oracle_answer.strip().lower().replace(" ", "_")
    )
    if answer == "still_true":
        return LabelDecision("alive", "oracle")
    if answer == "no_longer_true":
        return LabelDecision("dead", "oracle")
    return LabelDecision("masked", "unknown")


def apply_label(example: QueryExample, decision: LabelDecision) -> QueryExample:
    """Attach supervision without rebuilding or filtering the memory prompt."""
    return replace(example, label=decision.label, source=decision.source)


def volatile_from_return_table(
    statement: str,
    word_rates: Mapping[str, float],
    *,
    base_rate: float,
    threshold: float,
) -> bool:
    """Apply an externally frozen caption *re-emission* rate and threshold.

    Do not pass B1 endpoint-validity rates here: the published stratum uses
    an older return table. This function never refits the table or median.
    """
    rates = [word_rates[w] for w in content_words(statement) if w in word_rates]
    rate = sum(rates) / len(rates) if rates else base_rate
    return rate < threshold
