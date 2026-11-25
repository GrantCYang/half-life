"""Exercise the query-to-caption file workflow using explicit synthetic fixtures."""

import argparse
import json
from pathlib import Path

from halflife.paths import ROOT
from halflife.data.io import read_jsonl
from halflife.caption.generate import validate_requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out_dir
    if out.exists():
        raise ValueError("Use a new output directory")
    from halflife.cli import main as run

    source = ROOT / "examples/caption_bridge"
    run(
        [
            "build-queries",
            "--observations",
            str(ROOT / "examples/synthetic_observations.jsonl"),
            "--anchors",
            "10",
            "--horizons",
            "120",
            "--split",
            "heldout",
            "--out",
            str(out / "queries.jsonl"),
        ]
    )
    run(
        [
            "apply-calibration",
            "--predictions",
            str(source / "raw_predictions.jsonl"),
            "--maps",
            str(source / "calibration.json"),
            "--out",
            str(out / "calibrated.jsonl"),
        ]
    )
    run(
        [
            "caption-export",
            "--queries",
            str(out / "queries.jsonl"),
            "--predictions",
            str(out / "calibrated.jsonl"),
            "--baselines",
            str(source / "b3.jsonl"),
            "--endpoints",
            str(source / "endpoints.jsonl"),
            "--cases-out",
            str(out / "cases.jsonl"),
            "--predictions-out",
            str(out / "fact_scores.jsonl"),
        ]
    )
    run(
        [
            "caption-prepare",
            "--cases",
            str(out / "cases.jsonl"),
            "--predictions",
            str(out / "fact_scores.jsonl"),
            "--out",
            str(out / "requests.jsonl"),
        ]
    )
    requests = read_jsonl(out / "requests.jsonl")
    validate_requests(requests, source)
    facts = {
        m["fact_id"]: m["statement"]
        for c in read_jsonl(out / "cases.jsonl")
        for m in c["memory"]
    }
    report = {
        "origin": "synthetic_workflow_example",
        "new_model_inference": False,
        "calibration_refitted": False,
        "validated_requests": len(requests),
        "selections": {
            r["arm"]: [facts[i] for i in r["selected_fact_ids"]] for r in requests
        },
        "requests": str(out / "requests.jsonl"),
        "note": "Scores and the gray frame are synthetic fixtures, not research evidence. No captions were generated.",
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
