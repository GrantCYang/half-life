"""Build caption requests from saved cases and calibrated fact probabilities."""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

from halflife.caption.protocol import ARMS, caption_inputs
from halflife.data.io import read_jsonl, write_jsonl


def prepare_requests(cases, predictions):
    scores = defaultdict(dict)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case_id")
    for row in predictions:
        if row["fact_id"] in scores[row["case_id"]]:
            raise ValueError("Duplicate fact prediction")
        scores[row["case_id"]][row["fact_id"]] = row["p_alive"]
    if set(scores) != {c["case_id"] for c in cases}:
        raise ValueError("Prediction case IDs differ from caption cases")
    requests = []
    for case in cases:
        if set(scores[case["case_id"]]) != {m["fact_id"] for m in case["memory"]}:
            raise ValueError("Every inventory fact requires one calibrated probability")
        if not case["current_frames"]:
            raise ValueError("Captioning requires current frames")
        for arm in ARMS:
            inputs = caption_inputs(case, arm, scores[case["case_id"]])
            requests.append(
                {
                    "request_id": case["case_id"] + ":" + arm,
                    "case_id": case["case_id"],
                    "arm": arm,
                    "endpoint_sec": case["endpoint_sec"],
                    "frames": case["current_frames"],
                    "selected_fact_ids": inputs["selected_fact_ids"],
                    "prompt": inputs["prompt"],
                    "protocol": "caption_memory_half_v1",
                }
            )
    return requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL: case_id, fact_id, p_alive (calibrated)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(
            "Use a new output file; source/frozen inputs are never overwritten"
        )
    requests = prepare_requests(read_jsonl(args.cases), read_jsonl(args.predictions))
    write_jsonl(args.out, requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
