import json
import os
import subprocess
import sys
from pathlib import Path


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("frame.len,tcp.flags,Attack_type\n74,0x12,MITM\n", encoding="utf-8")


def test_prepare_datasets_relative_paths_are_portable(tmp_path: Path):
    edge_root = tmp_path / "Edge-IIoTset dataset"
    write_csv(edge_root / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv")
    write_csv(edge_root / "Attack traffic" / "MITM_attack.csv")

    snli_root = tmp_path / "snli_1.0"
    snli_root.mkdir()
    for split in ["train", "dev", "test"]:
        (snli_root / f"snli_1.0_{split}.jsonl").write_text("{}\n", encoding="utf-8")

    out_dir = tmp_path / "processed"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_datasets.py",
            "--edge-root",
            str(edge_root),
            "--snli-root",
            str(snli_root),
            "--out-dir",
            str(out_dir),
            "--relative-paths",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest_text = (out_dir / "edgeiiot" / "file_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["generation"]["relative_paths"] is True
    assert not Path(manifest["root"]).is_absolute()
    assert not Path(manifest["files"][0]["path"]).is_absolute()
