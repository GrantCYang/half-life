"""Numerical and causal invariants; no model downloads or GPU required."""

import math
import unittest

import numpy as np

from halflife.forecast.baselines import (
    BaseRate,
    WordClock,
    LexicalFeatures,
    SemanticFeatures,
    LexicalRidge,
    SemanticRidge,
    masked_mean_pool,
    ridge_fit,
    ridge_predict,
)
from halflife.forecast.calibration import Isotonic, LegacyIsotonic
from halflife.data.labels import assign_label, apply_label, volatile_from_return_table
from halflife.data.memory import (
    build_memory,
    build_examples,
    lexical_present,
    content_words,
)
from halflife.forecast.metrics import (
    brier,
    take_macro_brier,
    pooled_auc,
    ece,
    murphy,
    paired_group_brier_ci,
    paired_group_auc_ci,
)
from halflife.data.schema import TypedObservation, QueryExample
from halflife.forecast.scoring import candidate_probability, score_answer_tokens


def observation(t, visible, motion=(), inferred=()):
    return TypedObservation(
        "take_a", t, tuple(visible), tuple(motion), tuple(inferred), "group_a"
    )


def row(statement, delta=8, label="alive", presence=0.5, offset=0.5):
    return dict(
        statement=statement,
        delta_sec=delta,
        label=label,
        presence_frac=presence,
        last_seen_off=offset,
        ctx_visible="a person",
        ctx_motion="moving",
        ctx_inferred="cooking",
    )


class CausalMemoryTests(unittest.TestCase):
    def test_future_observations_cannot_change_input(self):
        past = [
            observation(9, ["The cook wears glasses."], ["Walking."]),
            observation(9.5, ["The cook is crouching."]),
            observation(10, ["The cook wears glasses."]),
        ]
        future = observation(
            11, ["The cook removed glasses."], ["Flying."], ["Dreaming."]
        )
        before = build_examples(past, [10], [8])
        after = build_examples(past + [future], [10], [8])
        self.assertEqual([r.to_dict() for r in before], [r.to_dict() for r in after])
        self.assertTrue(all(r.label == "masked" and r.target is None for r in before))

    def test_masked_target_does_not_remove_other_memory_items(self):
        examples = build_examples(
            [observation(10, ["A person wears glasses.", "A person is crouching."])],
            [10],
            [8],
        )
        labeled = [
            apply_label(examples[0], assign_label(oracle_answer="cannot_tell")),
            apply_label(examples[1], assign_label(oracle_answer="no_longer_true")),
        ]
        self.assertIsNone(labeled[0].y)
        self.assertEqual(labeled[1].y, 0)
        self.assertIn("A person wears glasses.", labeled[1].prompt)
        self.assertEqual(examples[1].prompt, labeled[1].prompt)

    def test_profile_counts_observed_clips_and_uses_last_matching_context(self):
        rows = [
            observation(0, ["A cook wears glasses."]),
            observation(9, ["A cook wears glasses."], ["Walking."]),
            observation(9.5, []),
            observation(10, ["A cook is crouching."], ["Crouching."]),
        ]
        snapshot = build_memory(rows, 10)
        age, fraction, last = snapshot.profile("A cook wears glasses.")
        self.assertEqual((age, fraction, last.time_sec), (1.0, 0.5, 9))
        ex = next(
            e for e in build_examples(rows, [10], [32]) if "glasses" in e.statement
        )
        self.assertEqual(ex.ctx_motion, "Walking.")
        self.assertIn("[1.0s ago, seen 50%] A cook wears glasses.", ex.prompt)
        self.assertIn("The camera now goes dark for 32 seconds.", ex.prompt)

    def test_inventory_is_deduplicated_then_capped(self):
        rows = [
            observation(9, ["A cook wears glasses.", "A red pan."]),
            observation(10, ["The cook wears glasses.", "A blue cup."]),
        ]
        snapshot = build_memory(rows, 10, max_facts=2)
        self.assertEqual(snapshot.inventory, ("A cook wears glasses.", "A red pan."))
        self.assertTrue(lexical_present("The cook wears glasses.", snapshot.inventory))
        self.assertFalse(lexical_present("the and is", snapshot.inventory))
        self.assertEqual(content_words("The RED red pan!"), frozenset(["red", "pan"]))

    def test_schema_json_roundtrip(self):
        obs = observation(1, ["A red pan."])
        self.assertEqual(TypedObservation.from_dict(obs.to_dict()), obs)
        ex = build_examples([obs], [1], [8])[0]
        self.assertEqual(QueryExample.from_dict(ex.to_dict()), ex)


class LabelTests(unittest.TestCase):
    def test_one_sided_caption_evidence_and_unknown_mask(self):
        self.assertEqual(
            assign_label(nli_entailment=0.5, oracle_answer="no_longer_true").label,
            "alive",
        )
        self.assertEqual(assign_label(nli_entailment=0.49).label, "masked")
        self.assertEqual(assign_label(oracle_answer="still_true").label, "alive")
        self.assertEqual(assign_label(oracle_answer="no_longer_true").label, "dead")
        for answer in (None, "cannot_tell", "maybe no_longer_true", "unparseable"):
            self.assertEqual(assign_label(oracle_answer=answer).label, "masked")

    def test_frozen_return_stratum_is_separate_from_endpoint_label(self):
        self.assertTrue(
            volatile_from_return_table(
                "red pan", {"red": 0.2, "pan": 0.4}, base_rate=0.7, threshold=0.5
            )
        )
        self.assertFalse(
            volatile_from_return_table("unseen word", {}, base_rate=0.7, threshold=0.5)
        )
        self.assertEqual(assign_label(oracle_answer="still_true").label, "alive")


class BaselineTests(unittest.TestCase):
    def test_counts_shrinkage_mask_and_log_interpolation(self):
        rows = (
            [row("red", 32, "alive")] * 20
            + [row("blue", 32, "dead")] * 20
            + [row("red", 120, "dead")] * 20
            + [row("blue", 120, "alive")] * 20
            + [row("red", 64, "dead")] * 100
            + [row("red", 32, "masked")] * 100
        )
        base = BaseRate((32, 120)).fit(rows)
        clock = WordClock((32, 120)).fit(rows)
        self.assertEqual(base.rates_, {32: 0.5, 120: 0.5})
        self.assertEqual(clock.predict([row("red", 32)])[0], 0.75)
        self.assertEqual(clock.predict([row("red", 120)])[0], 0.25)
        self.assertEqual(clock.predict([row("unknown", 32)])[0], 0.5)
        weight = math.log(64 / 32) / math.log(120 / 32)
        self.assertAlmostEqual(
            clock.predict([row("red", 64)])[0], 0.75 * (1 - weight) + 0.25 * weight
        )
        with self.assertRaises(ValueError):
            base.predict([row("red", 240)])

    def test_minimum_count_does_not_count_repeated_word_twice(self):
        clock = WordClock((8,), min_count=2).fit(
            [row("red red red"), row("blue", label="dead")]
        )
        self.assertEqual(clock.tables_[8], {})

    def test_unpenalized_intercept(self):
        weights = ridge_fit(np.zeros((3, 2)), [0, 1, 1], alpha=1e8)
        np.testing.assert_allclose(weights, [0, 0, 2 / 3], atol=1e-14)
        np.testing.assert_allclose(
            ridge_predict(np.ones((2, 2)), weights), [2 / 3, 2 / 3]
        )

    def test_feature_schema_and_frozen_vocabulary(self):
        features = LexicalFeatures(min_count=1).fit(
            [row("red", presence=0.2, offset=None)]
        )
        matrix = features.transform([row("blue", presence=0.7, offset=15)])
        np.testing.assert_allclose(matrix, [[0.7, 0.5, math.log(8), 0]])
        self.assertEqual(features.vocabulary_, ("red",))

    def test_masked_mean_pool_and_four_separate_projections(self):
        pooled = masked_mean_pool([[[1, 0], [0, 1], [100, 100]]], [[1, 1, 0]])
        np.testing.assert_allclose(pooled, [[2**-0.5, 2**-0.5]])

        def encoder(texts):
            return np.array(
                [[len(t), t.count("r"), t.count("a")] for t in texts], dtype=float
            )

        rows = [row("red"), row("blue", label="dead"), row("green")]
        features = SemanticFeatures(encoder, dim=2, min_count=1).fit(rows)
        matrix = features.transform(rows)
        self.assertEqual(matrix.shape, (3, 3 + 3 + 4 * 2))
        np.testing.assert_allclose(matrix[:, 6:].mean(axis=0), 0, atol=1e-14)
        frozen = [(m.copy(), v.copy()) for m, v in features.projections_]
        features.transform([row("previously unseen future")])
        for (mean, directions), (old_mean, old_directions) in zip(
            features.projections_, frozen
        ):
            np.testing.assert_array_equal(mean, old_mean)
            np.testing.assert_array_equal(directions, old_directions)
        self.assertEqual(LexicalRidge(min_count=1).fit(rows).predict(rows).shape, (3,))
        self.assertEqual(
            SemanticRidge(encoder, dim=2, min_count=1).fit(rows).predict(rows).shape,
            (3,),
        )


class CalibrationTests(unittest.TestCase):
    def test_tied_scores_are_order_invariant_for_new_fits(self):
        first = Isotonic().fit([0.5, 0.5], [0, 1])
        second = Isotonic().fit([0.5, 0.5], [1, 0])
        np.testing.assert_array_equal(first.predict([0.1, 0.5, 0.9]), [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(first.predict([0.5]), second.predict([0.5]))
        self.assertAlmostEqual(
            Isotonic().fit([0.5, 0.5], [0, 1], [1, 3]).predict(0.5), 0.75
        )

    def test_known_pav_solution_interpolation_and_serialization(self):
        model = Isotonic().fit([0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1])
        np.testing.assert_allclose(
            model.predict([0, 0.1, 0.15, 0.2, 0.8, 0.9, 1]),
            [0, 0, 0.25, 0.5, 0.5, 1, 1],
        )
        restored = Isotonic.from_dict(model.to_dict())
        np.testing.assert_array_equal(
            restored.predict([0.15, 0.95]), model.predict([0.15, 0.95])
        )
        with self.assertRaises(ValueError):
            Isotonic.from_dict(LegacyIsotonic().fit([0, 1], [0, 1]).to_dict())

    def test_legacy_replay_intentionally_keeps_duplicate_breakpoint_behavior(self):
        first = LegacyIsotonic().fit([0.5, 0.5], [0, 1])
        second = LegacyIsotonic().fit([0.5, 0.5], [1, 0])
        self.assertEqual(first.predict(0.5), 0)
        self.assertEqual(second.predict(0.5), 0.5)
        self.assertEqual(first.predict(0.500001), 1)
        self.assertEqual(LegacyIsotonic.from_dict(first.to_dict()).predict(0.5), 0)
        known = LegacyIsotonic().fit([0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1])
        np.testing.assert_allclose(known.bx_, [0.1, 0.8, 0.9])
        np.testing.assert_allclose(known.bv_, [0, 0.5, 1])


class MetricTests(unittest.TestCase):
    def test_take_macro_is_not_row_average(self):
        p, y, takes = [0, 0, 0, 0], [1, 1, 1, 0], ["a", "a", "a", "b"]
        self.assertEqual(brier(p, y), 0.75)
        self.assertEqual(take_macro_brier(p, y, takes), 0.5)
        with self.assertRaises(ValueError):
            brier([0.1], [None])

    def test_auc_and_calibration_measure_different_properties(self):
        self.assertEqual(pooled_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(pooled_auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(pooled_auc([0.1, 0.2], [0, 0]))
        self.assertEqual(ece([0.5, 0.5], [0, 1]), 0)
        self.assertEqual(brier([0.5, 0.5], [0, 1]), 0.25)
        self.assertEqual(ece([0, 1], [0, 1]), 0)

    def test_binned_murphy_is_explicitly_approximate(self):
        result = murphy([0.11, 0.19], [0, 1])
        self.assertAlmostEqual(result["brier_binned"], (0.15**2 + 0.85**2) / 2)
        self.assertAlmostEqual(result["brier_exact"], (0.11**2 + 0.81**2) / 2)
        self.assertNotEqual(result["binning_gap"], 0)

    def test_paired_bootstrap_preserves_take_weights_inside_groups(self):
        result = paired_group_brier_ci(
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            ["a", "b", "c"],
            ["g", "g", "h"],
            n_resamples=1000,
        )
        self.assertAlmostEqual(result["reduction"], 2 / 3)
        self.assertEqual(result["ci"], [0, 1])
        self.assertEqual((result["takes"], result["groups"]), (3, 2))
        with self.assertRaises(ValueError):
            paired_group_brier_ci(
                [0, 0], [1, 1], [0, 0], ["a", "a"], ["g", "h"], n_resamples=10
            )

    def test_auc_ci_uses_paired_group_draws(self):
        result = paired_group_auc_ci(
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            ["a", "a", "b", "b"],
            n_resamples=20,
        )
        self.assertEqual(result["difference"], 1)
        self.assertEqual(result["ci"], [1, 1])


class ScoringTests(unittest.TestCase):
    def test_known_two_candidate_probability_and_extreme_likelihoods(self):
        self.assertAlmostEqual(
            candidate_probability(-1000, -1001), 1 / (1 + math.exp(-1))
        )
        self.assertEqual(candidate_probability(-1000, 0), 0)
        self.assertEqual(candidate_probability(0, -1000), 1)

    def test_token_boundaries_do_not_assume_one_token_per_label(self):
        scores = score_answer_tokens(
            [math.log(0.6), math.log(0.5), math.log(0.9), math.log(0.8)],
            [math.log(0.4), math.log(0.5), math.log(0.9), math.log(0.8)],
            alive_label_tokens=2,
            dead_label_tokens=2,
        )
        for result in scores.values():
            self.assertAlmostEqual(result["p_alive"], 0.6)
        scores = score_answer_tokens(
            [math.log(0.5), math.log(0.9)],
            [math.log(0.5), math.log(0.1)],
            alive_label_tokens=1,
            dead_label_tokens=1,
        )
        self.assertAlmostEqual(scores["label_only"]["p_alive"], 0.5)
        self.assertAlmostEqual(scores["label_eos"]["p_alive"], 0.9)
        with self.assertRaises(ValueError):
            score_answer_tokens([-1], [-1], alive_label_tokens=1, dead_label_tokens=1)


if __name__ == "__main__":
    unittest.main()
