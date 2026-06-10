# Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible local data-preparation layer needed before server-side SPECTRA-FedCore and Fed-SB experiments.

**Architecture:** Keep raw datasets immutable under `data/raw/`, generate auditable metadata under `data/processed/`, and put reusable logic in a small standard-library Python package. The preparation layer should not train models; it should discover files, normalize labels/source groups, create deterministic split manifests, and render small instruction examples for smoke tests.

**Tech Stack:** Python 3 standard library, `pytest`, JSON config/artifacts, shell scripts through Python CLIs.

---

## File Structure

- `src/fedllm_data/__init__.py`: package exports.
- `src/fedllm_data/edgeiiot.py`: Edge-IIoTset file discovery, manifest construction, source grouping, deterministic split planning, and small CSV sampling.
- `src/fedllm_data/snli.py`: SNLI JSONL manifest construction for Fed-SB reproduction.
- `src/fedllm_data/prompts.py`: deterministic Edge-IIoT instruction prompt rendering.
- `scripts/prepare_datasets.py`: CLI that writes processed manifests for Edge-IIoTset and SNLI.
- `configs/data/edgeiiot.json`: local raw-data path defaults and split seed.
- `configs/data/snli.json`: local SNLI path defaults.
- `tests/test_edgeiiot_manifest.py`: tests for file classification and source groups.
- `tests/test_edgeiiot_splits.py`: tests for deterministic source-aware split planning.
- `tests/test_prompts.py`: tests for prompt rendering.
- `tests/test_snli_manifest.py`: tests for SNLI split manifesting.
- `data/processed/edgeiiot/*.json`: generated artifacts, not hand-authored.
- `data/processed/snli/*.json`: generated artifacts, not hand-authored.

## Task 1: Edge-IIoT Manifest API

**Files:**
- Create: `tests/test_edgeiiot_manifest.py`
- Create: `src/fedllm_data/edgeiiot.py`
- Create: `src/fedllm_data/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from fedllm_data.edgeiiot import build_file_manifest


def write_csv(path: Path, rows: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("feature,label\n" + "\n".join(f"{idx},Normal" for idx in range(rows)) + "\n")


def test_build_file_manifest_classifies_full_dataset_files(tmp_path: Path):
    root = tmp_path / "Edge-IIoTset dataset"
    write_csv(root / "Attack traffic" / "MITM_attack.csv", rows=3)
    write_csv(root / "Normal traffic" / "Distance" / "Distance.csv", rows=4)
    write_csv(root / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv", rows=5)

    manifest = build_file_manifest(root, count_rows=True)

    assert [item["group"] for item in manifest["files"]] == ["attack", "normal", "selected"]
    assert manifest["files"][0]["label_hint"] == "MITM"
    assert manifest["files"][1]["source"] == "Distance"
    assert manifest["files"][2]["selected_kind"] == "ML"
    assert [item["rows"] for item in manifest["files"]] == [3, 4, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_edgeiiot_manifest.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fedllm_data'`.

- [ ] **Step 3: Write minimal implementation**

Implement `build_file_manifest(root: Path, count_rows: bool = False) -> dict` so it:

```python
{
    "dataset": "edgeiiotset",
    "root": "/absolute/path",
    "files": [
        {
            "relative_path": "Attack traffic/MITM_attack.csv",
            "group": "attack",
            "source": "MITM",
            "label_hint": "MITM",
            "selected_kind": None,
            "rows": 3,
        }
    ],
}
```

Sort files by relative path, follow symlinks, and count rows as `line_count - 1` only when `count_rows=True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_edgeiiot_manifest.py -v`

Expected: PASS.

## Task 2: Source-Aware Split Planning

**Files:**
- Create: `tests/test_edgeiiot_splits.py`
- Modify: `src/fedllm_data/edgeiiot.py`

- [ ] **Step 1: Write the failing test**

```python
from fedllm_data.edgeiiot import make_source_split_plan


def test_make_source_split_plan_is_deterministic_and_keeps_sources_intact():
    manifest = {
        "files": [
            {"relative_path": "Attack traffic/A.csv", "group": "attack", "source": "A"},
            {"relative_path": "Attack traffic/B.csv", "group": "attack", "source": "B"},
            {"relative_path": "Normal traffic/N1/N1.csv", "group": "normal", "source": "N1"},
            {"relative_path": "Normal traffic/N2/N2.csv", "group": "normal", "source": "N2"},
            {"relative_path": "Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv", "group": "selected", "source": "ML"},
        ]
    }

    split_a = make_source_split_plan(manifest, seed=7, train_ratio=0.5, val_ratio=0.25)
    split_b = make_source_split_plan(manifest, seed=7, train_ratio=0.5, val_ratio=0.25)

    assert split_a == split_b
    assigned = split_a["train"] + split_a["val"] + split_a["test"]
    assert sorted(assigned) == sorted(["A", "B", "N1", "N2"])
    assert "ML" not in assigned
    assert split_a["excluded_sources"] == ["ML"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_edgeiiot_splits.py -v`

Expected: FAIL with `ImportError` for `make_source_split_plan`.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic source split planning using `random.Random(seed).shuffle`. Exclude selected merged files from source-aware splits because they do not preserve original source boundaries.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_edgeiiot_splits.py -v`

Expected: PASS.

## Task 3: Prompt Rendering

**Files:**
- Create: `tests/test_prompts.py`
- Create: `src/fedllm_data/prompts.py`

- [ ] **Step 1: Write the failing test**

```python
from fedllm_data.prompts import render_edgeiiot_prompt


def test_render_edgeiiot_prompt_is_deterministic_and_label_bounded():
    row = {
        "frame.len": "74",
        "tcp.flags": "0x00000012",
        "http.request.method": "GET",
        "Attack_type": "MITM",
    }

    prompt = render_edgeiiot_prompt(row, feature_names=["frame.len", "tcp.flags", "http.request.method"])

    assert "Industrial IoT intrusion-detection record" in prompt["instruction"]
    assert "frame.len=74" in prompt["input"]
    assert prompt["output"] == "MITM"
    assert "Valid labels:" in prompt["instruction"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_prompts.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Implement `render_edgeiiot_prompt(row, feature_names, label_key="Attack_type") -> dict[str, str]` returning deterministic `instruction`, `input`, and `output`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_prompts.py -v`

Expected: PASS.

## Task 4: SNLI Reproduction Manifest

**Files:**
- Create: `tests/test_snli_manifest.py`
- Create: `src/fedllm_data/snli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from fedllm_data.snli import build_snli_manifest


def test_build_snli_manifest_counts_jsonl_splits(tmp_path: Path):
    root = tmp_path / "snli_1.0"
    root.mkdir()
    for split, rows in {"train": 2, "dev": 1, "test": 3}.items():
        (root / f"snli_1.0_{split}.jsonl").write_text("\n".join("{}" for _ in range(rows)) + "\n")

    manifest = build_snli_manifest(root)

    assert manifest["dataset"] == "snli_1.0"
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["dev"]["rows"] == 1
    assert manifest["splits"]["test"]["rows"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_snli_manifest.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Implement `build_snli_manifest(root: Path) -> dict` for the three official JSONL splits.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_snli_manifest.py -v`

Expected: PASS.

## Task 5: Dataset Preparation CLI

**Files:**
- Create: `scripts/prepare_datasets.py`
- Create: `configs/data/edgeiiot.json`
- Create: `configs/data/snli.json`
- Modify: `data/README.md`

- [ ] **Step 1: Write CLI smoke test through existing APIs**

Run the API tests first: `PYTHONPATH=src pytest tests -v`

Expected: PASS.

- [ ] **Step 2: Implement CLI**

Implement a standard-library `argparse` script that:

```bash
PYTHONPATH=src python3 scripts/prepare_datasets.py \
  --edge-root data/raw/edgeiiotset/full_dataset \
  --snli-root data/raw/snli/current \
  --out-dir data/processed
```

It writes:

- `data/processed/edgeiiot/file_manifest.json`
- `data/processed/edgeiiot/source_split_seed20260531.json`
- `data/processed/edgeiiot/prompt_smoke_samples.jsonl`
- `data/processed/snli/manifest.json`

- [ ] **Step 3: Run CLI on local datasets**

Run the command above.

Expected:

- 26 Edge-IIoT CSV files in the file manifest.
- 24 source-aware files grouped into train/val/test; 2 selected merged CSVs excluded from source-aware split.
- SNLI counts train/dev/test as 550152/10000/10000.

## Task 6: Verification and Documentation

**Files:**
- Modify: `data/README.md`
- Modify: `docs/superpowers/specs/2026-05-31-spectra-dp-fedcore-design.md`

- [ ] **Step 1: Run tests**

Run: `PYTHONPATH=src pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Validate generated artifacts**

Run:

```bash
python3 -m json.tool data/processed/edgeiiot/file_manifest.json >/dev/null
python3 -m json.tool data/processed/edgeiiot/source_split_seed20260531.json >/dev/null
python3 -m json.tool data/processed/snli/manifest.json >/dev/null
wc -l data/processed/edgeiiot/prompt_smoke_samples.jsonl
```

Expected: JSON validation succeeds and prompt smoke sample count is greater than zero.

- [ ] **Step 3: Update docs**

Record the generated artifact paths, counts, and intended server handoff command in `data/README.md`. Add a short note in the research design that the data-preparation layer now has reproducible local artifacts.
