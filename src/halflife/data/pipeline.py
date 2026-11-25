"""Explicit file boundaries for the clip -> memory -> forecast pipeline."""

from __future__ import annotations
from halflife.data.io import (
    read_jsonl as read_jsonl,
    write_jsonl as write_jsonl,
    validate_split_boundaries as validate_split_boundaries,
)
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable
from halflife.data.io import unique_by_id
from halflife.models import parse_interrogator_output, parse_observer_output
from halflife.data.labels import assign_label
from halflife.data.memory import content_words


def extract_clip_frames(
    video: str | Path,
    endpoint_sec: float,
    directory: str | Path,
    offsets: Iterable[float] = (-2.0, -1.0, 0.0),
) -> list[Path]:
    """Use FFmpeg's decoded frame seek; drop negative leading offsets only."""
    if not math.isfinite(endpoint_sec) or endpoint_sec < 0:
        raise ValueError("clip endpoint must be a finite nonnegative time")
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(video)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    times = [
        endpoint_sec + float(offset)
        for offset in offsets
        if endpoint_sec + float(offset) >= 0
    ]
    if not times or times != sorted(times) or any(t > endpoint_sec for t in times):
        raise ValueError("clip offsets must be ordered and never include future frames")
    frames = []
    for index, time_sec in enumerate(times):
        target = directory / f"frame_{index}.png"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-ss",
            str(time_sec),
            "-frames:v",
            "1",
            str(target),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(
                f"video has no decodable frame at {time_sec:.3f}s: {video}"
            )
        frames.append(target)
    return frames


def validate_manifest(manifest: list[dict]) -> None:
    seen = set()
    for row in manifest:
        uid = row["take_uid"]
        if uid in seen:
            raise ValueError(f"duplicate video take_uid {uid}")
        seen.add(uid)
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        if (
            not math.isfinite(float(row["duration_sec"]))
            or float(row["duration_sec"]) <= 0
        ):
            raise ValueError("duration_sec must be finite and positive")
        if row.get("split") not in {"train", "calibration", "heldout", "confirmation"}:
            raise ValueError(
                "each video must have an explicit train/calibration/heldout/confirmation split"
            )


def observe_videos(
    manifest: list[dict],
    instrument: Any,
    config: dict,
    *,
    start_sec: float = 0.0,
    stop_sec: float | None = None,
):
    validate_manifest(manifest)
    step = float(config["sampling_step_sec"])
    if step <= 0 or start_sec < 0:
        raise ValueError("positive sampling step and nonnegative start are required")
    for video in manifest:
        endpoint = float(video["duration_sec"])
        if stop_sec is not None:
            endpoint = min(endpoint, stop_sec)
        for index in range(max(0, math.ceil((endpoint - start_sec) / step))):
            time_sec = round(start_sec + index * step, 6)
            if time_sec >= endpoint:
                break
            with tempfile.TemporaryDirectory(prefix="halflife-clip-") as temporary:
                frames = extract_clip_frames(
                    video["video_path"],
                    time_sec,
                    temporary,
                    config["frame_offsets_sec"],
                )
                prompt = (
                    config["prompt_preamble_by_frame_count"][str(len(frames))]
                    + "\n\n"
                    + config["prompt_body"]
                )
                raw = instrument.generate(frames, prompt, config["max_new_tokens"])
            parsed = parse_observer_output(raw)
            if len(frames) == 1 and parsed["motion"]:
                raise ValueError("single-frame boundary observation returned motion")
            yield {
                "take_uid": video["take_uid"],
                "time_sec": time_sec,
                "split": video["split"],
                "group": video.get("group", ""),
                **parsed,
                "raw_response": raw,
                "origin": "new_observer_inference",
            }


def future_caption_pool(
    observations: list[dict],
    take_uid: str,
    query_sec: float,
    statement: str,
    half_window: float = 1.0,
    cap: int = 14,
) -> list[str]:
    """Future information is legal here, exclusively on the label side."""
    candidates = dict.fromkeys(
        text
        for row in sorted(observations, key=lambda r: float(r["time_sec"]))
        if row["take_uid"] == take_uid
        and abs(float(row["time_sec"]) - query_sec) <= half_window + 1e-8
        for text in row["visible"]
    )
    words = content_words(statement)
    return sorted(
        candidates, key=lambda text: (-len(words & content_words(text)), text)
    )[:cap]


def compute_caption_support(examples: list[dict], observations: list[dict], nli: Any):
    for row in examples:
        captions = future_caption_pool(
            observations,
            row["take_uid"],
            float(row["anchor_sec"]) + float(row["delta_sec"]),
            row["statement"],
        )
        yield {
            "example_id": row["example_id"],
            "caption_entailment_score": nli.score(row["statement"], captions),
            "candidate_count": len(captions),
            "origin": "new_nli_inference",
        }


def label_examples(
    examples: list[dict],
    caption_support: list[dict],
    oracle_answers: list[dict],
    threshold: float = 0.5,
) -> list[dict]:
    """Caption-confirmed ALIVE first, otherwise an explicit oracle judgment.

    No response / CANNOT_TELL remains masked. Inputs/prompts and inventories
    were constructed before this join and are never changed by label outcomes.
    """
    support = unique_by_id(caption_support, "caption support")
    oracle = unique_by_id(oracle_answers, "oracle answers")
    available = unique_by_id(examples, "examples")
    if (set(support) | set(oracle)) - set(available):
        raise ValueError("label evidence refers to unknown example IDs")
    out = []
    for row in examples:
        score = support.get(row["example_id"], {}).get("caption_entailment_score")
        decision = assign_label(
            nli_entailment=None if score is None else float(score),
            oracle_answer=oracle.get(row["example_id"], {}).get("answer"),
            nli_threshold=threshold,
        )
        label, source = decision.label, decision.source
        out.append(
            {
                **row,
                "label": label,
                "source": source,
                "target": {"alive": "ALIVE", "dead": "DEAD", "masked": None}[label],
                "caption_entailment_score": score,
            }
        )
    return out


def interrogate_examples(
    examples: list[dict],
    caption_support: list[dict],
    manifest: list[dict],
    instrument: Any,
    config: dict,
):
    validate_manifest(manifest)
    videos = {r["take_uid"]: r for r in manifest}
    support = unique_by_id(caption_support, "caption support")
    threshold = float(config["caption_entailment_threshold"])
    # Reuse a judgment for identical take / statement / endpoint requests.
    cache = {}
    for row in examples:
        if (
            float(support.get(row["example_id"], {}).get("caption_entailment_score", 0))
            >= threshold
        ):
            continue
        endpoint = float(row["anchor_sec"]) + float(row["delta_sec"])
        video = videos[row["take_uid"]]
        if endpoint >= float(video["duration_sec"]):
            raise ValueError("oracle endpoint exceeds video coverage")
        key = (row["take_uid"], row["statement"], endpoint)
        if key not in cache:
            with tempfile.TemporaryDirectory(prefix="halflife-oracle-") as temporary:
                frames = extract_clip_frames(
                    video["video_path"],
                    endpoint,
                    temporary,
                    config["frame_offsets_sec"],
                )
                prompt = (
                    config["prompt_preamble_by_frame_count"][str(len(frames))]
                    + "\n\n"
                    + config["prompt_body"].format(statement=row["statement"])
                )
                raw = instrument.generate(frames, prompt, config["max_new_tokens"])
            try:
                answer, parse_ok = parse_interrogator_output(raw), True
            except ValueError:
                answer, parse_ok = "CANNOT_TELL", False
            cache[key] = {"answer": answer, "parse_ok": parse_ok, "raw_response": raw}
        yield {
            "example_id": row["example_id"],
            **cache[key],
            "origin": "new_interrogator_inference",
        }
