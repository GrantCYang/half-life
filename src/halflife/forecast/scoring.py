"""Pure numerical scoring of two teacher-forced candidate continuations."""

from __future__ import annotations

import math
from typing import Sequence


def candidate_probability(lp_alive: float, lp_dead: float) -> float:
    """Normalize over ALIVE and DEAD candidates with a stable two-way softmax.

    Inputs are SUMS of token log probabilities, not averages. The returned
    value is conditional on these two candidates, not vocabulary probability.
    """
    if not all(math.isfinite(x) for x in (lp_alive, lp_dead)):
        raise ValueError("candidate log likelihoods must be finite")
    difference = lp_alive - lp_dead
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    ratio = math.exp(difference)
    return ratio / (1.0 + ratio)


def logprob_sum(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(v) and v <= 1e-6 for v in values):
        raise ValueError("expected nonempty finite token log probabilities <= 0")
    return float(sum(values))


def score_answer_tokens(
    alive_lps: Sequence[float],
    dead_lps: Sequence[float],
    *,
    alive_label_tokens: int,
    dead_label_tokens: int,
    eos_tokens: int = 1,
) -> dict[str, dict[str, float]]:
    """Compare suffix, answer-only and answer-plus-EOS likelihoods.

    Sequences contain only continuation tokens, already aligned by the usual
    next-token shift. The caller verifies tokenizer boundaries; counts need
    not match between ALIVE and DEAD. Remaining suffix tokens may be newlines.
    """
    if not isinstance(eos_tokens, int) or eos_tokens < 0:
        raise ValueError("eos_tokens must be a nonnegative integer")
    for values, count in (
        (alive_lps, alive_label_tokens),
        (dead_lps, dead_label_tokens),
    ):
        if not isinstance(count, int) or count < 1 or count + eos_tokens > len(values):
            raise ValueError("invalid label/EOS boundary")
        logprob_sum(values)
    segments = {
        "full": (alive_lps, dead_lps),
        "label_only": (alive_lps[:alive_label_tokens], dead_lps[:dead_label_tokens]),
        "label_eos": (
            alive_lps[: alive_label_tokens + eos_tokens],
            dead_lps[: dead_label_tokens + eos_tokens],
        ),
    }
    result = {}
    for name, (alive, dead) in segments.items():
        a, d = logprob_sum(alive), logprob_sum(dead)
        result[name] = {
            "lp_alive": a,
            "lp_dead": d,
            "p_alive": candidate_probability(a, d),
        }
    return result
