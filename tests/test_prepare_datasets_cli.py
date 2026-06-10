import json
import os
import subprocess
import sys
from pathlib import Path


def write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("frame.len,tcp.flags,Attack_type\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_prepare_datasets_cli_writes_expected_artifacts(tmp_path: Path):
    edge_root = tmp_path / "Edge-IIoTset dataset"
    write_csv(edge_root / "Attack traffic" / "MITM_attack.csv", ["74,0x12,MITM"])
    write_csv(edge_root / "Normal traffic" / "Distance" / "Distance.csv", ["80,0x10,Normal"])
    write_csv(
        edge_root / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv",
        ["74,0x12,MITM", "80,0x10,Normal"],
    )

    snli_root = tmp_path / "snli_1.0"
    snli_root.mkdir()
    for split in ["train", "dev", "test"]:
        (snli_root / f"snli_1.0_{split}.jsonl").write_text("{}\n", encoding="utf-8")

    out_dir = tmp_path / "processed"
    env = {**os.environ, "PYTHONPATH": "src"}
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
            "--sample-count",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    file_manifest = json.loads((out_dir / "edgeiiot" / "file_manifest.json").read_text(encoding="utf-8"))
    split_plan = json.loads(
        (out_dir / "edgeiiot" / "source_split_seed20260531.json").read_text(encoding="utf-8")
    )
    label_inventory = json.loads((out_dir / "edgeiiot" / "label_inventory.json").read_text(encoding="utf-8"))
    prompt_lines = (out_dir / "edgeiiot" / "prompt_smoke_samples.jsonl").read_text(encoding="utf-8").splitlines()
    snli_manifest = json.loads((out_dir / "snli" / "manifest.json").read_text(encoding="utf-8"))

    assert file_manifest["file_count"] == 3
    assert split_plan["excluded_sources"] == ["ML"]
    assert label_inventory["selected_label_counts"]["ML-EdgeIIoT-dataset.csv"] == {"MITM": 1, "Normal": 1}
    assert len(prompt_lines) == 2
    assert snli_manifest["splits"]["train"]["rows"] == 1
