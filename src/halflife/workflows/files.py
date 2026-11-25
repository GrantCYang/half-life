"""Small JSON and manifest boundaries for CLI workflows."""

import json
from pathlib import Path
from halflife.data.io import read_jsonl


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _save(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest(path):
    rows = read_jsonl(path)
    for row in rows:
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            row["video_path"] = str((Path(path).parent / video_path).resolve())
    return rows
