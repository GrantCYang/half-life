"""HalfLife console interface. Stage implementations live in library modules."""

from __future__ import annotations
import argparse
import importlib
import sys
from halflife.paths import ROOT


def _model_options(parser):
    parser.add_argument(
        "--model",
        required=True,
        help="Authorized local checkpoint path; never auto-downloaded",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--attention", default="eager")


def make_parser():
    parser = argparse.ArgumentParser(
        description="HALF-LIFE: predict the future validity of visual memories"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "build-queries",
        help="Construct prompts only from observations at or before each anchor",
    )
    p.add_argument("--observations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--anchors",
        type=float,
        nargs="+",
        help="Explicit anchors; otherwise require full 2-Hz future coverage",
    )
    p.add_argument("--horizons", type=float, nargs="+", default=[8, 32, 64, 120])
    p.add_argument(
        "--split", choices=("train", "calibration", "heldout", "confirmation")
    )
    p.add_argument("--stride", type=float, default=21)
    p = sub.add_parser(
        "label",
        help="Join caption support and/or saved oracle answers; unknown remains masked",
    )
    p.add_argument("--queries", required=True)
    p.add_argument("--caption-support")
    p.add_argument("--oracle-answers")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "observe", help="Generate typed observations from user-provided videos"
    )
    _model_options(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/observer.json"))
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--stop", type=float)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "caption-support", help="Compute future-caption NLI support for labeling only"
    )
    p.add_argument("--model", required=True, help="Local NLI checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--entailment-index", type=int)
    p.add_argument("--queries", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "interrogate",
        help="Query future clips only where caption support does not confirm ALIVE",
    )
    _model_options(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--caption-support")
    p.add_argument("--config", default=str(ROOT / "configs/interrogator.json"))
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "train", help="Assistant-only LoRA SFT; train split and 8/32/120 seconds only"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/forecaster.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "score", help="Run new candidate-likelihood inference and save raw predictions"
    )
    _model_options(p)
    p.add_argument("--adapter")
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=("full", "label_only"), default="full")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "evaluate", help="Evaluate predictions joined by unique example IDs"
    )
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--probability-key", default="p_alive_raw")
    p.add_argument(
        "--split", choices=("train", "calibration", "heldout", "confirmation")
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "calibrate",
        help="Fit one training-side map on 8/32/120 seconds and apply to all horizons",
    )
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument(
        "--fit-split", choices=("train", "calibration"), default="calibration"
    )
    p.add_argument("--probability-key", default="p_alive_raw")
    p.add_argument("--maps", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "apply-calibration",
        help="Apply a saved map without refitting; supports archived legacy PAV",
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--probability-key", default="p_alive_raw")
    p.add_argument("--out", required=True)
    sub.add_parser("baselines", help="Fit B0–B4 on the declared training split")
    sub.add_parser(
        "caption-prepare", help="Build V0/B3/LoRA caption inputs from saved scores"
    )
    sub.add_parser(
        "caption-export",
        help="Join query IDs, calibrated forecasts, B3 scores and endpoint frames",
    )
    sub.add_parser(
        "caption-generate",
        help="Generate captions from prepared requests and a local model",
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    standalone = {
        "baselines": "halflife.forecast.runner",
        "caption-prepare": "halflife.caption.prepare",
        "caption-export": "halflife.caption.export",
        "caption-generate": "halflife.caption.generate",
    }
    if argv and argv[0] in standalone:
        return importlib.import_module(standalone[argv[0]]).main(argv[1:])
    args = make_parser().parse_args(argv)
    data = {"build-queries", "label", "observe", "caption-support", "interrogate"}
    module = "data" if args.command in data else "forecast"
    return importlib.import_module("halflife.workflows." + module).run(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
