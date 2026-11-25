"""Cross-stage identities, causal boundaries and the documented CPU workflow."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from halflife.caption.export import export_cases
from halflife.caption.prepare import prepare_requests
from halflife.caption.demo import main as demo
from halflife.data.io import read_jsonl
from halflife.data.memory import build_examples
from halflife.paths import ROOT


def inputs():
    queries = [
        r.to_dict()
        for r in build_examples(
            read_jsonl(ROOT / "examples/synthetic_observations.jsonl"),
            [10],
            [120],
            split="heldout",
        )
    ]
    source = ROOT / "examples/caption_bridge"
    predictions = [
        {**r, "p_alive": r["p_alive_raw"]}
        for r in read_jsonl(source / "raw_predictions.jsonl")
    ]
    return (
        queries,
        predictions,
        read_jsonl(source / "b3.jsonl"),
        read_jsonl(source / "endpoints.jsonl"),
    )


class CaptionExportTests(unittest.TestCase):
    def test_join_is_order_independent_and_excludes_future_labels(self):
        q, p, b, e = inputs()
        expected = export_cases(q, p, b, e)
        q.reverse()
        p.reverse()
        b.reverse()
        for row in q:
            row.update(
                label="dead", future_caption="SECRET_FUTURE", private_judgment="SECRET"
            )
        actual = export_cases(q, p, b, e)
        self.assertEqual(actual, expected)
        self.assertNotIn("SECRET", json.dumps(prepare_requests(*actual)))
        self.assertEqual(actual[0][0]["memory"][0]["current_age_sec"], 120)

    def test_missing_duplicate_mismatched_and_invalid_scores_fail(self):
        for kind in ("missing", "duplicate", "take", "delta", "nan", "range"):
            q, p, b, e = inputs()
            if kind == "missing":
                p.pop()
            if kind == "duplicate":
                b.append(copy.deepcopy(b[0]))
            if kind == "take":
                p[0]["take_uid"] = "another-take"
            if kind == "delta":
                p[0]["delta_sec"] = 32
            if kind == "nan":
                p[0]["p_alive"] = float("nan")
            if kind == "range":
                b[0]["p_alive"] = 1.1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                export_cases(q, p, b, e)

    def test_endpoint_and_history_boundaries(self):
        for kind in (
            "empty",
            "duplicate",
            "future-frame",
            "unknown-last-seen",
            "negative-last-seen",
            "future-memory",
        ):
            q, p, b, e = inputs()
            if kind == "empty":
                e[0]["anchor_sec"] = 9
            if kind == "duplicate":
                e.append(copy.deepcopy(e[0]))
            if kind == "future-frame":
                e[0]["current_frames"][0]["time_sec"] = 131
            if kind == "unknown-last-seen":
                q[0]["last_seen_off"] = None
            if kind == "negative-last-seen":
                q[0]["last_seen_off"] = 11
            if kind == "future-memory":
                q[0]["last_seen_off"] = -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                export_cases(q, p, b, e)

    def test_explicit_historical_fact_ids_are_preserved_and_complete(self):
        q, p, b, e = inputs()
        e[0]["fact_ids"] = {
            row["example_id"]: f"original-{i}" for i, row in enumerate(q)
        }
        cases, scores = export_cases(q, p, b, e)
        self.assertEqual(
            {m["fact_id"] for m in cases[0]["memory"]}, {"original-0", "original-1"}
        )
        self.assertEqual({s["fact_id"] for s in scores}, {"original-0", "original-1"})
        e[0]["fact_ids"].pop(q[0]["example_id"])
        with self.assertRaises(ValueError):
            export_cases(q, p, b, e)

    def test_documented_demo_reaches_checked_caption_requests_without_models(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "demo"
            self.assertEqual(demo(["--out-dir", str(out)]), 0)
            report = json.loads((out / "summary.json").read_text())
            self.assertFalse(report["new_model_inference"])
            self.assertEqual(report["validated_requests"], 3)
            self.assertEqual(report["selections"]["V0"], [])
            self.assertNotEqual(
                report["selections"]["V_B3"], report["selections"]["V_LoRA"]
            )
            with self.assertRaises(ValueError):
                demo(["--out-dir", str(out)])
