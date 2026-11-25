"""Observation, query construction and target-labeling stages."""

from __future__ import annotations
import json
from collections import defaultdict
from halflife.data.io import read_jsonl, write_jsonl
from halflife.workflows.files import _json, _manifest
from halflife.data.pipeline import (
    observe_videos,
    interrogate_examples,
    label_examples,
    compute_caption_support,
)


def _build_queries(args):
    from halflife.data.memory import build_examples

    grouped = defaultdict(list)
    for row in read_jsonl(args.observations):
        grouped[row["take_uid"]].append(row)
    result = []
    for uid in sorted(grouped):
        rows = grouped[uid]
        splits = {r.get("split") for r in rows if r.get("split")}
        if len(splits) > 1:
            raise ValueError("one take cannot span multiple dataset splits")
        split = args.split or (next(iter(splits)) if splits else None)
        if split is None:
            raise ValueError("specify --split or include a split in every observation")
        anchors = args.anchors
        if anchors is None:
            times = sorted({round(float(r["time_sec"]), 6) for r in rows})
            stride_steps = round(args.stride / 0.5)
            if stride_steps < 1 or abs(stride_steps * 0.5 - args.stride) > 1e-8:
                raise ValueError("stride must be a positive multiple of 0.5 seconds")
            observed = set(times)
            post_steps = round((max(args.horizons) + 1) / 0.5)
            anchors = [
                t
                for t in times[::stride_steps]
                if all(
                    round(t + 0.5 * step, 6) in observed
                    for step in range(-2, post_steps + 1)
                )
            ]
        result.extend(
            example.to_dict()
            for example in build_examples(rows, anchors, args.horizons, split=split)
        )
    if not result:
        raise ValueError(
            "no query could be constructed: check inventory and temporal coverage"
        )
    write_jsonl(args.out, result)
    print(
        json.dumps(
            {
                "queries": len(result),
                "output": str(args.out),
                "labels": "not yet attached",
            }
        )
    )


def run(args):
    if args.command == "build-queries":
        _build_queries(args)
    elif args.command == "label":
        rows = label_examples(
            read_jsonl(args.queries),
            read_jsonl(args.caption_support) if args.caption_support else [],
            read_jsonl(args.oracle_answers) if args.oracle_answers else [],
            args.threshold,
        )
        write_jsonl(args.out, rows)
    elif args.command == "observe":
        from halflife.models import VisualInstrument

        model = VisualInstrument(
            args.model, device=args.device, dtype=args.dtype, attention=args.attention
        )
        write_jsonl(
            args.out,
            observe_videos(
                _manifest(args.manifest),
                model,
                _json(args.config),
                start_sec=args.start,
                stop_sec=args.stop,
            ),
        )
    elif args.command == "caption-support":
        from halflife.models import CaptionEntailment

        nli = CaptionEntailment(
            args.model, device=args.device, entailment_index=args.entailment_index
        )
        write_jsonl(
            args.out,
            compute_caption_support(
                read_jsonl(args.queries), read_jsonl(args.observations), nli
            ),
        )
    elif args.command == "interrogate":
        from halflife.models import VisualInstrument

        model = VisualInstrument(
            args.model, device=args.device, dtype=args.dtype, attention=args.attention
        )
        write_jsonl(
            args.out,
            interrogate_examples(
                read_jsonl(args.queries),
                read_jsonl(args.caption_support) if args.caption_support else [],
                _manifest(args.manifest),
                model,
                _json(args.config),
            ),
        )
    return 0
