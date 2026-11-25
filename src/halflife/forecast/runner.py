#!/usr/bin/env python3
"""Fit B0-B4 on a declared fit split and write raw probabilities for all rows.

CPU example:
  halflife baselines --data labeled.jsonl --out outputs/baselines.jsonl

Semantic controls require an authorized local sentence-embedding checkpoint:
  halflife baselines --data labeled.jsonl --out outputs/baselines.jsonl \
      --arms B0 B1 B2 B3 B4 --embedding-model /models/all-MiniLM-L6-v2

The output is a wide JSONL table: example_id, split, B0_raw, B1_raw, etc.
It joins directly with halflife evaluate --probability-key B1_raw. Calibration
is separate. The declared alpha is a new-run choice, not a historical refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Allow direct execution from a clean source checkout, before editable install.


import numpy as np
from halflife.forecast.baselines import (
    BaseRate,
    WordClock,
    LexicalRidge,
    SemanticRidge,
    SemanticFeatures,
    GradientBoostedBaseline,
    TRAINED_HORIZONS,
)
from halflife.data.io import (
    read_jsonl,
    write_jsonl,
    unique_by_id,
    validate_split_boundaries,
)


class LocalSentenceEncoder:
    """Frozen masked-mean/L2 encoder; no model download or training."""

    def __init__(self, path, *, device="cpu", batch_size=256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        )
        self.cache = {}

    def __call__(self, texts):
        pending = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                normalized = self.torch.nn.functional.normalize(mean, p=2, dim=1)
            vectors = normalized.float().cpu().numpy()
            self.cache.update(zip(batch, vectors))
        return np.asarray([self.cache[text] for text in texts])


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("B0", "B1", "B2", "B3", "B4"),
        default=["B0", "B1", "B2"],
    )
    parser.add_argument("--fit-split", choices=("train",), default="train")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Explicit new-run alpha; tune within fit groups separately if required",
    )
    parser.add_argument(
        "--embedding-model",
        help="Authorized local MiniLM-style checkpoint; required for B3/B4",
    )
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--svd-dim", type=int, default=24)
    return parser


def run(args):
    rows = read_jsonl(args.data)
    unique_by_id(rows, "baseline input")
    if not rows or any(not row.get("split") for row in rows):
        raise ValueError("every input row must carry an explicit split")
    validate_split_boundaries(rows)
    fitting = [row for row in rows if row["split"] == args.fit_split]
    if not fitting:
        raise ValueError("no rows in the declared fit split")
    arms = list(dict.fromkeys(args.arms))
    if args.embedding_batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    semantic = any(arm in arms for arm in ("B3", "B4"))
    if semantic and not args.embedding_model:
        raise ValueError(
            "B3/B4 require --embedding-model; no checkpoint is downloaded automatically"
        )
    models = {}
    if "B0" in arms:
        models["B0"] = BaseRate().fit(fitting)
    if "B1" in arms:
        models["B1"] = WordClock(
            min_count=args.min_count, shrinkage=args.shrinkage
        ).fit(fitting)
    if "B2" in arms:
        models["B2"] = LexicalRidge(args.ridge_alpha, min_count=args.min_count).fit(
            fitting
        )
    if semantic:
        encoder = LocalSentenceEncoder(
            args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
        if "B3" in arms:
            models["B3"] = SemanticRidge(
                encoder, args.ridge_alpha, dim=args.svd_dim, min_count=args.min_count
            ).fit(fitting)
            features = models["B3"].features
        else:
            features = SemanticFeatures(
                encoder, dim=args.svd_dim, min_count=args.min_count
            ).fit(fitting)
        if "B4" in arms:
            models["B4"] = GradientBoostedBaseline(features).fit(fitting)
    predictions = {arm: models[arm].predict(rows) for arm in arms}
    output = []
    for index, row in enumerate(rows):
        result = {
            key: row[key] for key in ("example_id", "take_uid", "delta_sec", "split")
        }
        result.update({f"{arm}_raw": float(predictions[arm][index]) for arm in arms})
        output.append(result)
    write_jsonl(args.out, output)
    usable = [
        row
        for row in fitting
        if row.get("label") in ("alive", "dead")
        and float(row["delta_sec"]) in TRAINED_HORIZONS
    ]
    receipt = {
        "mode": "new_baseline_fit",
        "fit_split": args.fit_split,
        "output_rows": len(rows),
        "feature_fit_rows": len(fitting),
        "supervised_fit_rows": len(usable),
        "trained_horizons_sec": list(TRAINED_HORIZONS),
        "arms": arms,
        "min_count": args.min_count,
        "shrinkage": args.shrinkage,
        "ridge_alpha": args.ridge_alpha,
        "svd_dim_per_field": args.svd_dim,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "calibration": "none; predictions are raw",
        "note": "Only the declared fit split fits vocabulary, SVD or predictors. The fixed alpha is an explicit new-run choice, not the archived inner-selected alpha.",
    }
    metadata = Path(args.out).with_suffix(".metadata.json")
    metadata.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return receipt


def main(argv=None):
    run(make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
