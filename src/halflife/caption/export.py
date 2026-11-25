"""Join forecast queries and saved scores into caption cases without inference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from halflife.caption.prepare import prepare_requests
from halflife.data.io import read_jsonl, unique_by_id, write_jsonl


def number(value, name, low=0, high=math.inf):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name}: expected a finite number in [{low}, {high}]")
    return float(value)


def identity(row):
    take = row.get("take_uid")
    if not isinstance(take, str) or not take:
        raise ValueError("take_uid must be a nonempty string")
    anchor = number(row.get("anchor_sec"), "anchor_sec")
    delta = number(row.get("delta_sec"), "delta_sec")
    if delta == 0:
        raise ValueError("delta_sec must be positive")
    return take, anchor, delta


def checked_score(index, query, key, name):
    row = index.get(query["example_id"])
    if row is None:
        raise ValueError(f"Missing {name} prediction: {query['example_id']}")
    for field in ("take_uid", "delta_sec", "anchor_sec"):
        if field in row and row[field] != query[field]:
            raise ValueError(f"{name} {field} differs from query identity")
    return number(row.get(key), f"{name}.{key}", high=1)


def export_cases(queries, predictions, baselines, endpoints, *, baseline_key="p_alive"):
    """Use every query at each declared (take, anchor, horizon), without labels.

    New inventories use example_id as fact_id. An endpoint may supply a complete
    fact_ids mapping to retain historical IDs and therefore historical tie breaks.
    Input row order never controls joins or fact selection.
    """
    query_index = unique_by_id(queries, "queries")
    forecast_index = unique_by_id(predictions, "forecast predictions")
    baseline_index = unique_by_id(baselines, "baseline predictions")
    grouped = defaultdict(list)
    for query in query_index.values():
        grouped[identity(query)].append(query)
    if not endpoints:
        raise ValueError("No caption endpoints supplied")
    cases, scores, seen_cases, seen_keys = [], [], set(), set()
    for endpoint in endpoints:
        key = identity(endpoint)
        case_id = endpoint.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ValueError("case_id must be nonempty and unique")
        if key in seen_keys:
            raise ValueError("Duplicate caption take/anchor/horizon")
        seen_cases.add(case_id)
        seen_keys.add(key)
        rows = sorted(grouped.get(key, []), key=lambda r: r["example_id"])
        if not rows:
            raise ValueError(f"No queries match caption endpoint {case_id}")
        fact_ids = endpoint.get(
            "fact_ids", {r["example_id"]: r["example_id"] for r in rows}
        )
        if (
            not isinstance(fact_ids, dict)
            or set(fact_ids) != {r["example_id"] for r in rows}
            or any(not isinstance(v, str) or not v for v in fact_ids.values())
            or len(set(fact_ids.values())) != len(rows)
        ):
            raise ValueError(
                "fact_ids must map every selected example_id to a unique nonempty fact_id"
            )
        take, anchor, delta = key
        target = anchor + delta
        frames = endpoint.get("current_frames", [])
        if not frames:
            raise ValueError("Caption endpoint requires current_frames")
        clean_frames = []
        for frame in frames:
            time = number(frame.get("time_sec"), "frame.time_sec", high=target)
            path, digest = frame.get("path"), frame.get("sha256")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or not re.fullmatch("[0-9a-f]{64}", digest)
            ):
                raise ValueError("Frames require path and lowercase SHA256")
            clean_frames.append({"path": path, "time_sec": time, "sha256": digest})
        times = [f["time_sec"] for f in clean_frames]
        if times != sorted(set(times)):
            raise ValueError("Frame timestamps must be ordered and unique")
        case = {
            "case_id": case_id,
            "take_uid": take,
            "anchor_sec": anchor,
            "endpoint_sec": target,
            "current_frames": clean_frames,
            "memory": [],
        }
        for query in rows:
            fact_id = fact_ids[query["example_id"]]
            offset = number(query.get("last_seen_off"), "last_seen_off", high=anchor)
            statement = query.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError("Query statement is empty")
            case["memory"].append(
                {
                    "fact_id": fact_id,
                    "example_id": query["example_id"],
                    "statement": statement,
                    "last_observed_sec": anchor - offset,
                    "current_age_sec": delta + offset,
                    "presence_frac": number(
                        query.get("presence_frac"), "presence_frac", high=1
                    ),
                    "b3_score": checked_score(
                        baseline_index, query, baseline_key, "B3"
                    ),
                }
            )
            scores.append(
                {
                    "case_id": case_id,
                    "fact_id": fact_id,
                    "example_id": query["example_id"],
                    "p_alive": checked_score(
                        forecast_index, query, "p_alive", "forecaster"
                    ),
                }
            )
        cases.append(case)
    # Validate the consumer contract before writing either output.
    prepare_requests(cases, scores)
    return cases, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "predictions", "baselines", "endpoints"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--baseline-key",
        default="p_alive",
        help="Saved B3 score column; default is calibrated p_alive",
    )
    parser.add_argument("--cases-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = [args.cases_out, args.predictions_out]
    if len({p.resolve() for p in outputs}) != 2 or any(p.exists() for p in outputs):
        raise ValueError("Use two distinct new output files")
    cases, scores = export_cases(
        *(
            read_jsonl(getattr(args, name))
            for name in ("queries", "predictions", "baselines", "endpoints")
        ),
        baseline_key=args.baseline_key,
    )
    write_jsonl(args.cases_out, cases)
    write_jsonl(args.predictions_out, scores)
    print(
        json.dumps(
            {"cases": len(cases), "facts": len(scores), "new_model_inference": False}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
