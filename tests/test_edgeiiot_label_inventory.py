from pathlib import Path

from fedllm_data.edgeiiot import build_file_manifest, build_label_inventory


def write_csv(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"1,{label}" for label in labels)
    path.write_text(f"feature,Attack_type\n{rows}\n", encoding="utf-8")


def test_build_label_inventory_uses_source_rows_and_scans_selected_files(tmp_path: Path):
    root = tmp_path / "Edge-IIoTset dataset"
    write_csv(root / "Attack traffic" / "MITM_attack.csv", ["MITM", "MITM"])
    write_csv(root / "Normal traffic" / "Distance" / "Distance.csv", ["Normal"])
    write_csv(root / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv", ["MITM", "Normal", "Normal"])
    manifest = build_file_manifest(root, count_rows=True)

    inventory = build_label_inventory(manifest, scan_selected=True)

    assert inventory["raw_source_label_counts"] == {"MITM": 2, "Normal": 1}
    assert inventory["selected_label_counts"]["ML-EdgeIIoT-dataset.csv"] == {"MITM": 1, "Normal": 2}
