"""JSONL boundaries, identity checks and split validation."""

from pathlib import Path
from typing import Iterable
import json


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                result.append(item)
    return result


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def unique_by_id(rows: list[dict], name: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        key = row["example_id"]
        if key in indexed:
            raise ValueError(f"duplicate example_id in {name}: {key}")
        indexed[key] = row
    return indexed


def validate_split_boundaries(rows: list[dict]) -> None:
    """Training and calibration may share a take; evaluation sides may not."""
    seen = {}
    for row in rows:
        split = row.get("split")
        if split not in {"train", "calibration", "heldout", "confirmation"}:
            raise ValueError(
                "each example needs a train/calibration/heldout/confirmation split"
            )
        side = "fit" if split in {"train", "calibration"} else split
        identities = [("take", row["take_uid"])]
        if row.get("group"):
            identities.append(("group", row["group"]))
        for identity in identities:
            if identity in seen and seen[identity] != side:
                raise ValueError(
                    f"{identity[0]} {identity[1]} crosses fitting/evaluation split boundaries"
                )
            seen[identity] = side
