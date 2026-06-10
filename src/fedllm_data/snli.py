from __future__ import annotations

from pathlib import Path
from typing import Any


def build_snli_manifest(root: Path | str) -> dict[str, Any]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"SNLI root does not exist: {root_path}")

    splits = {}
    for split in ["train", "dev", "test"]:
        path = root_path / f"snli_1.0_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"SNLI split is missing: {path}")
        splits[split] = {
            "path": str(path.resolve()),
            "rows": _count_lines(path),
            "bytes": path.stat().st_size,
        }

    return {
        "dataset": "snli_1.0",
        "root": str(root_path.resolve()),
        "splits": splits,
    }


def _count_lines(path: Path) -> int:
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    if path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                line_count += 1
    return line_count
