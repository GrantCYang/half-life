"""Training, candidate scoring and calibration workflows."""

from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
from halflife.data.io import (
    read_jsonl,
    write_jsonl,
    unique_by_id,
    validate_split_boundaries,
)
from halflife.workflows.files import _json, _save


def _evaluate(args):
    from halflife.forecast import metrics

    examples = read_jsonl(args.data)
    predictions = unique_by_id(read_jsonl(args.predictions), "predictions")
    example_index = unique_by_id(examples, "evaluation data")
    if set(predictions) - set(example_index):
        raise ValueError("prediction IDs are absent from evaluation data")
    labeled = [
        r
        for r in examples
        if r.get("label") in ("alive", "dead")
        and (args.split is None or r.get("split") == args.split)
    ]
    if not labeled:
        raise ValueError("evaluation contains no labeled examples")
    if any(r["example_id"] not in predictions for r in labeled):
        raise ValueError("every labeled evaluation example must have a prediction")
    groups = defaultdict(list)
    for row in labeled:
        prediction = float(predictions[row["example_id"]][args.probability_key])
        groups[("all", float(row["delta_sec"]))].append((row, prediction))
        if row.get("volatile") is True:
            groups[("volatile", float(row["delta_sec"]))].append((row, prediction))
    result = {
        "probability_key": args.probability_key,
        "split": args.split or "all supplied splits",
        "cells": [],
    }
    for (stratum, horizon), entries in sorted(groups.items()):
        y = [int(row["label"] == "alive") for row, _ in entries]
        p = [prob for _, prob in entries]
        takes = [row["take_uid"] for row, _ in entries]
        result["cells"].append(
            {
                "stratum": stratum,
                "delta_sec": horizon,
                "n": len(y),
                "take_macro_brier": metrics.take_macro_brier(p, y, takes),
                "pooled_auc": metrics.pooled_auc(p, y),
                "pooled_ece": metrics.ece(p, y),
            }
        )
    _save(args.out, result)


def _calibrate(args):
    from halflife.forecast.calibration import Isotonic

    examples = unique_by_id(read_jsonl(args.data), "calibration data")
    validate_split_boundaries(list(examples.values()))
    prediction_rows = read_jsonl(args.predictions)
    predictions = unique_by_id(prediction_rows, "calibration predictions")
    if set(predictions) - set(examples):
        raise ValueError("prediction IDs are absent from calibration data")
    selected = [
        row
        for row in examples.values()
        if row.get("split") == args.fit_split
        and row.get("label") in ("alive", "dead")
        and float(row["delta_sec"]) in (8.0, 32.0, 120.0)
    ]
    if not selected:
        raise ValueError("no labeled training-side calibration rows were selected")
    if any(row["example_id"] not in predictions for row in selected):
        raise ValueError("missing predictions for calibration examples")
    x = [
        float(predictions[row["example_id"]][args.probability_key]) for row in selected
    ]
    y = [int(row["label"] == "alive") for row in selected]
    mapping = Isotonic().fit(x, y)
    outputs = []
    for prediction in prediction_rows:
        outputs.append(
            {
                **prediction,
                "p_alive": float(
                    mapping.predict([prediction[args.probability_key]])[0]
                ),
                "calibration": "corrected_isotonic",
                "calibration_fit_split": args.fit_split,
            }
        )
    _save(
        args.maps,
        {
            "fit_split": args.fit_split,
            "fit_examples": len(selected),
            "fit_horizons_sec": [8, 32, 120],
            "note": "New corrected tie-aggregating maps; not the archived historical step-calibration maps.",
            "pooled_map": mapping.to_dict(),
        },
    )
    write_jsonl(args.out, outputs)


def _apply_calibration(args):
    from halflife.forecast.calibration import Isotonic, LegacyIsotonic
    import hashlib

    if Path(args.out).resolve() in {
        Path(args.predictions).resolve(),
        Path(args.maps).resolve(),
    }:
        raise ValueError("write derived predictions to a separate output file")
    saved = _json(args.maps)
    row = saved.get("pooled_map", saved)
    if row.get("algorithm") == "isotonic_ties_linear_v1":
        mapping = Isotonic.from_dict(row)
    elif (
        row.get("algorithm", "legacy_pav_step_v1") == "legacy_pav_step_v1"
        and "bx" in row
    ):
        mapping = LegacyIsotonic.from_dict(row)
    else:
        raise ValueError("unrecognized calibration map; no map is fitted automatically")
    rows = read_jsonl(args.predictions)
    unique_by_id(rows, "predictions")
    probabilities = mapping.predict([r[args.probability_key] for r in rows])
    digest = hashlib.sha256(Path(args.maps).read_bytes()).hexdigest()
    write_jsonl(
        args.out,
        (
            {
                **r,
                "p_alive": float(p),
                "calibration": mapping.to_dict()["algorithm"],
                "calibration_sha256": digest,
                "calibration_refitted": False,
            }
            for r, p in zip(rows, probabilities)
        ),
    )


def run(args):
    if args.command == "train":
        from halflife.forecast.training import train_lora

        config = _json(args.config)
        source = read_jsonl(args.data)
        validate_split_boundaries(source)
        rows = [
            r
            for r in source
            if r["split"] == "train"
            and r.get("label") in ("alive", "dead")
            and (float(r["delta_sec"]) in config["trained_horizons_sec"])
        ]
        train_lora(
            rows, args.model, args.out, config, device=args.device, resume=args.resume
        )
    elif args.command == "score":
        from halflife.models import Forecaster

        rank, world = (
            int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
        )
        device = (
            f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
            if args.device.startswith("cuda") and world > 1
            else args.device
        )
        model = Forecaster(
            args.model,
            args.adapter,
            device=device,
            dtype=args.dtype,
            attention=args.attention,
            max_length=args.max_length,
        )
        rows = read_jsonl(args.data)
        unique_by_id(rows, "scoring data")
        out = Path(args.out)
        if world > 1:
            out = out.with_name(f"{out.stem}.rank{rank}{out.suffix}")
        write_jsonl(
            out,
            (
                {
                    "example_id": r["example_id"],
                    "take_uid": r["take_uid"],
                    "delta_sec": r["delta_sec"],
                    **model.score(r["prompt"], args.mode),
                }
                for r in rows[rank::world]
            ),
        )
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "calibrate":
        _calibrate(args)
    elif args.command == "apply-calibration":
        _apply_calibration(args)
    return 0
