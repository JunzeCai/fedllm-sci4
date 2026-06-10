from pathlib import Path

from fedllm_data.snli import build_snli_manifest


def test_build_snli_manifest_counts_jsonl_splits(tmp_path: Path):
    root = tmp_path / "snli_1.0"
    root.mkdir()
    for split, rows in {"train": 2, "dev": 1, "test": 3}.items():
        path = root / f"snli_1.0_{split}.jsonl"
        path.write_text("\n".join("{}" for _ in range(rows)) + "\n", encoding="utf-8")

    manifest = build_snli_manifest(root)

    assert manifest["dataset"] == "snli_1.0"
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["dev"]["rows"] == 1
    assert manifest["splits"]["test"]["rows"] == 3
