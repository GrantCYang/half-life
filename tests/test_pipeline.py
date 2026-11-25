"""Pipeline boundary tests; no weights, network, GPU, or private data required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from halflife.cli import main
from halflife.models import (
    candidate_probability,
    parse_interrogator_output,
    parse_observer_output,
    render_candidate,
)
from halflife.data.pipeline import (
    future_caption_pool,
    label_examples,
    read_jsonl,
    write_jsonl,
    extract_clip_frames,
    unique_by_id,
    validate_split_boundaries,
)
from halflife.forecast.training import supervised_tokens, gradient_accumulation


class CharacterTokenizer:
    """A deliberately simple template to verify token boundaries and masking."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        prefix = "USER:" + messages[0]["content"] + "\nASSISTANT:"
        return (
            prefix
            if add_generation_prompt
            else prefix + messages[-1]["content"] + "<END>\n"
        )

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode("utf-8"))}


class PipelineTests(unittest.TestCase):
    def test_label_route_and_masking(self):
        rows = [
            {"example_id": str(i), "prompt": "unchanged", "statement": "a fact"}
            for i in range(4)
        ]
        supports = [
            {"example_id": "0", "caption_entailment_score": 0.8},
            {"example_id": "1", "caption_entailment_score": 0.1},
        ]
        answers = [
            {"example_id": "0", "answer": "NO_LONGER_TRUE"},
            {"example_id": "1", "answer": "NO_LONGER_TRUE"},
            {"example_id": "2", "answer": "CANNOT_TELL"},
        ]
        result = label_examples(rows, supports, answers)
        self.assertEqual(
            [r["label"] for r in result], ["alive", "dead", "masked", "masked"]
        )
        self.assertTrue(all(r["prompt"] == "unchanged" for r in result))
        self.assertEqual(result[0]["source"], "caption_return")
        self.assertEqual(result[1]["source"], "oracle")

    def test_label_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            label_examples(
                [{"example_id": "a"}],
                [{"example_id": "b", "caption_entailment_score": 0.9}],
                [],
            )

    def test_assistant_mask_includes_answer_and_suffix(self):
        tokenizer = CharacterTokenizer()
        ids, prompt_end, label_end = render_candidate(
            tokenizer, "Is the cup present?", "ALIVE"
        )
        row = supervised_tokens(tokenizer, "Is the cup present?", "ALIVE")
        self.assertEqual(row["labels"][:prompt_end], [-100] * prompt_end)
        self.assertEqual(row["labels"][prompt_end:], ids[prompt_end:])
        self.assertEqual(bytes(ids[prompt_end:label_end]).decode(), "ALIVE")
        self.assertEqual(bytes(ids[label_end:]).decode(), "<END>\n")
        with self.assertRaises(ValueError):
            supervised_tokens(tokenizer, "long input", "DEAD", max_length=3)

    def test_candidate_normalization_is_stable(self):
        self.assertAlmostEqual(candidate_probability(-1, -1), 0.5)
        self.assertEqual(candidate_probability(-1, -10000), 1)
        self.assertEqual(candidate_probability(-10000, -1), 0)
        with self.assertRaises(ValueError):
            candidate_probability(float("nan"), -1)

    def test_global_batch(self):
        self.assertEqual(gradient_accumulation(16, 8), 2)
        self.assertEqual(gradient_accumulation(16, 1), 16)
        with self.assertRaises(ValueError):
            gradient_accumulation(16, 3)

    def test_future_pool_never_reads_other_take(self):
        rows = [
            {"take_uid": "a", "time_sec": 9, "visible": ["Cup on table."]},
            {"take_uid": "a", "time_sec": 11, "visible": ["Cup on table."]},
            {"take_uid": "a", "time_sec": 15, "visible": ["Secret future."]},
            {"take_uid": "b", "time_sec": 10, "visible": ["Another take."]},
        ]
        self.assertEqual(
            future_caption_pool(rows, "a", 10, "Cup on table."), ["Cup on table."]
        )

    def test_strict_instrument_parsing(self):
        self.assertEqual(parse_interrogator_output(" STILL_TRUE \n"), "STILL_TRUE")
        with self.assertRaises(ValueError):
            parse_interrogator_output("It is STILL_TRUE")
        parsed = parse_observer_output('{"visible":["Cup."],"motion":[],"inferred":[]}')
        self.assertEqual(parsed["visible"], ["Cup."])
        with self.assertRaises(ValueError):
            parse_observer_output('{"visible":[],"motion":[],"inferred":[]}')

    def test_extract_calls_ffmpeg_without_shell_or_future_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"fixture")

            def fake_run(command, **kwargs):
                self.assertNotIn("shell", kwargs)
                self.assertLessEqual(float(command[command.index("-ss") + 1]), 1)
                Path(command[-1]).write_bytes(b"frame")

            with patch("halflife.data.pipeline.subprocess.run", side_effect=fake_run):
                paths = extract_clip_frames(video, 1, Path(directory) / "frames")
            self.assertEqual(len(paths), 2)

    def test_cli_build_and_label_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            observations = [
                {
                    "take_uid": "demo",
                    "time_sec": 9.5,
                    "split": "heldout",
                    "visible": ["A person is crouching."],
                    "motion": [],
                    "inferred": [],
                },
                {
                    "take_uid": "demo",
                    "time_sec": 10,
                    "split": "heldout",
                    "visible": ["A person wears a white shirt."],
                    "motion": [],
                    "inferred": [],
                },
                {
                    "take_uid": "demo",
                    "time_sec": 20,
                    "split": "heldout",
                    "visible": ["DO NOT LEAK THIS FUTURE FACT."],
                    "motion": [],
                    "inferred": [],
                },
            ]
            write_jsonl(base / "obs.jsonl", observations)
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "build-queries",
                        "--observations",
                        str(base / "obs.jsonl"),
                        "--anchors",
                        "10",
                        "--horizons",
                        "8",
                        "--out",
                        str(base / "queries.jsonl"),
                    ]
                )
            queries = read_jsonl(base / "queries.jsonl")
            self.assertEqual(len(queries), 2)
            self.assertTrue(all("DO NOT LEAK" not in row["prompt"] for row in queries))
            write_jsonl(
                base / "answers.jsonl",
                [{"example_id": queries[0]["example_id"], "answer": "STILL_TRUE"}],
            )
            main(
                [
                    "label",
                    "--queries",
                    str(base / "queries.jsonl"),
                    "--oracle-answers",
                    str(base / "answers.jsonl"),
                    "--out",
                    str(base / "labeled.jsonl"),
                ]
            )
            labeled = read_jsonl(base / "labeled.jsonl")
            self.assertEqual([r["label"] for r in labeled], ["alive", "masked"])
            self.assertEqual(
                [r["prompt"] for r in labeled], [r["prompt"] for r in queries]
            )

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            unique_by_id([{"example_id": "a"}, {"example_id": "a"}], "test")

    def test_split_boundaries_include_capture_groups(self):
        validate_split_boundaries(
            [
                {"take_uid": "a", "split": "train"},
                {"take_uid": "a", "split": "calibration"},
            ]
        )
        with self.assertRaises(ValueError):
            validate_split_boundaries(
                [
                    {"take_uid": "a", "group": "same", "split": "train"},
                    {"take_uid": "b", "group": "same", "split": "heldout"},
                ]
            )

    def test_metrics_and_train_side_calibration_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = [
                {
                    "example_id": "a",
                    "take_uid": "fit",
                    "delta_sec": 8,
                    "label": "dead",
                    "split": "calibration",
                },
                {
                    "example_id": "b",
                    "take_uid": "fit",
                    "delta_sec": 8,
                    "label": "alive",
                    "split": "calibration",
                },
                {
                    "example_id": "c",
                    "take_uid": "eval",
                    "delta_sec": 8,
                    "label": "alive",
                    "split": "heldout",
                },
                {
                    "example_id": "d",
                    "take_uid": "fit",
                    "delta_sec": 64,
                    "label": "dead",
                    "split": "calibration",
                },
            ]
            predictions = [
                {"example_id": "a", "p_alive_raw": 0.1},
                {"example_id": "b", "p_alive_raw": 0.9},
                {"example_id": "c", "p_alive_raw": 0.7},
                {"example_id": "d", "p_alive_raw": 0.7},
            ]
            write_jsonl(base / "data.jsonl", rows)
            write_jsonl(base / "p.jsonl", predictions)
            main(
                [
                    "evaluate",
                    "--data",
                    str(base / "data.jsonl"),
                    "--predictions",
                    str(base / "p.jsonl"),
                    "--out",
                    str(base / "metrics.json"),
                ]
            )
            result = json.loads((base / "metrics.json").read_text())
            self.assertAlmostEqual(result["cells"][0]["take_macro_brier"], 0.05)
            self.assertAlmostEqual(result["cells"][0]["pooled_auc"], 1.0)
            main(
                [
                    "calibrate",
                    "--data",
                    str(base / "data.jsonl"),
                    "--predictions",
                    str(base / "p.jsonl"),
                    "--maps",
                    str(base / "maps.json"),
                    "--out",
                    str(base / "calibrated.jsonl"),
                ]
            )
            fitted = json.loads((base / "maps.json").read_text())
            self.assertEqual(fitted["fit_examples"], 2)
            outputs = read_jsonl(base / "calibrated.jsonl")
            self.assertAlmostEqual(outputs[2]["p_alive"], 0.75)
            self.assertAlmostEqual(outputs[3]["p_alive"], 0.75)


if __name__ == "__main__":
    unittest.main()
