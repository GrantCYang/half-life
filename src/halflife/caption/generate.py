"""Generate captions from prepared requests using an explicit local checkpoint.

This portable adapter shares the original prompts and memory selection. GPU
generation has not been rerun; frozen judgments are scored by replay-caption.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from halflife.data.io import read_jsonl, write_jsonl


def validate_requests(requests, frame_root):
    if not requests or len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("Requests must be nonempty and have unique IDs")
    root = Path(frame_root).resolve()
    prepared = []
    for request in requests:
        frames = request["frames"]
        endpoint = float(request["endpoint_sec"])
        times = [float(f["time_sec"]) for f in frames]
        if (
            not math.isfinite(endpoint)
            or not times
            or times != sorted(set(times))
            or any(not math.isfinite(t) or not 0 <= t <= endpoint + 1e-6 for t in times)
        ):
            raise ValueError(
                "Frames must be ordered, unique and no later than the endpoint"
            )
        paths = []
        for frame in frames:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Frame path escapes --frame-root")
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                raise ValueError("Frame identity differs: " + frame["path"])
            paths.append(path)
        if not isinstance(request["prompt"], str) or not request["prompt"].strip():
            raise ValueError("Caption prompt is empty")
        prepared.append((request, paths))
    return prepared


def generate_requests(prepared, instrument, max_new_tokens):
    if max_new_tokens < 1:
        raise ValueError("Token limit must be positive")
    for request, paths in prepared:
        response = instrument.generate(paths, request["prompt"], max_new_tokens)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Caption generator returned an empty response")
        yield {
            "request_id": request["request_id"],
            "case_id": request["case_id"],
            "arm": request["arm"],
            "caption": response,
            "input_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest(),
            "origin": "new_local_caption_generation",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Use the generation limit from the relevant frozen configuration",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check requests and frame hashes without loading a model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.validate_only:
        prepared = validate_requests(read_jsonl(args.requests), args.frame_root)
        print(
            json.dumps(
                {"validated_requests": len(prepared), "new_model_inference": False}
            )
        )
        return 0
    if args.model is None or args.max_new_tokens is None or args.out is None:
        parser.error("generation requires --model, --max-new-tokens and --out")
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    if args.out.exists() or temporary.exists():
        raise ValueError(
            "Output or partial output already exists; use a new output path"
        )
    if args.max_new_tokens < 1:
        parser.error("Token limit must be positive")
    prepared = validate_requests(read_jsonl(args.requests), args.frame_root)
    from halflife.models import VisualInstrument

    instrument = VisualInstrument(
        str(args.model), device=args.device, dtype=args.dtype, attention=args.attention
    )
    write_jsonl(temporary, generate_requests(prepared, instrument, args.max_new_tokens))
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
