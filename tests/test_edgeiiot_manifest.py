from pathlib import Path

from fedllm_data.edgeiiot import build_file_manifest


def write_csv(path: Path, rows: int = 2, label: str = "Normal") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{idx},{label}" for idx in range(rows))
    path.write_text(f"feature,label\n{body}\n", encoding="utf-8")


def test_build_file_manifest_classifies_full_dataset_files(tmp_path: Path):
    root = tmp_path / "Edge-IIoTset dataset"
    write_csv(root / "Attack traffic" / "MITM_attack.csv", rows=3, label="MITM")
    write_csv(root / "Normal traffic" / "Distance" / "Distance.csv", rows=4)
    write_csv(root / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv", rows=5)

    manifest = build_file_manifest(root, count_rows=True)

    assert [item["group"] for item in manifest["files"]] == ["attack", "normal", "selected"]
    assert manifest["files"][0]["label_hint"] == "MITM"
    assert manifest["files"][0]["label_from_first_row"] == "MITM"
    assert manifest["files"][1]["source"] == "Distance"
    assert manifest["files"][1]["label_from_first_row"] == "Normal"
    assert manifest["files"][2]["selected_kind"] == "ML"
    assert [item["rows"] for item in manifest["files"]] == [3, 4, 5]
