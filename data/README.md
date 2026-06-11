# Dataset Inventory

Date prepared: 2026-05-31

## Edge-IIoTset

Primary full archive:

```text
data/raw/kagglehub/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/5.archive
```

SHA-256:

```text
42df619d4a222c5ae014a3b8ad2017536dbf49fbc2d35145349e6012a65b74ae
```

Extracted CSV/PDF/README subset:

```text
data/raw/edgeiiotset/full/
```

Convenience symlinks:

```text
data/raw/edgeiiotset/full_dataset
data/raw/edgeiiotset/DNN-EdgeIIoT-dataset.csv
data/raw/edgeiiotset/ML-EdgeIIoT-dataset-full.csv
```

Extracted contents:

- `Attack traffic/*.csv`: 14 attack CSV files.
- `Normal traffic/<sensor>/*.csv`: 10 normal sensor/source CSV files.
- `Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`: 2,219,202 lines.
- `Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`: 157,801 lines.
- `Edge_IIoTset__DatasetFL.pdf`
- `Readme.txt`

Notes:

- PCAP files were intentionally not extracted. The archive still contains them.
- The full CSV extraction preserves file/source structure for file/source-aware splits.
- A smaller Kaggle mirror was also downloaded under `data/raw/kagglehub/datasets/sibasispradhan/edge-iiotset-dataset/versions/2`; it contains selected CSVs only.
- A public GitHub mirror of `ML-EdgeIIoT-dataset.csv` was downloaded at `data/raw/edgeiiotset/ML-EdgeIIoT-dataset.csv` for quick sanity checks.

## SNLI

Source:

```text
https://nlp.stanford.edu/projects/snli/snli_1.0.zip
```

Archive:

```text
data/raw/snli/snli_1.0.zip
```

SHA-256:

```text
afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e
```

Extracted directory:

```text
data/raw/snli/snli_1.0/
```

Convenience symlink:

```text
data/raw/snli/current
```

JSONL line counts:

```text
snli_1.0_train.jsonl  550,152
snli_1.0_dev.jsonl     10,000
snli_1.0_test.jsonl    10,000
```

Use:

- Fed-SB official private federated reproduction.
- Sanity-checking privacy/accounting code before Edge-IIoTset migration.

## Processed Artifacts

Prepared artifacts:

```text
data/processed/edgeiiot/file_manifest.json
data/processed/edgeiiot/source_split_seed20260531.json
data/processed/edgeiiot/label_inventory.json
data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json
data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json
data/processed/edgeiiot/prompt_smoke_samples.jsonl
data/processed/snli/manifest.json
```

Generation command:

```bash
PYTHONPATH=src python3 scripts/prepare_datasets.py \
  --edge-root data/raw/edgeiiotset/full_dataset \
  --snli-root data/raw/snli/current \
  --out-dir data/processed \
  --count-rows \
  --relative-paths
```

Generated Edge-IIoTset summary:

- File manifest: 26 CSV files: 14 attack files, 10 normal source files, 2 selected merged files.
- Source-aware split seed: `20260531`.
- Source-aware split sizes: 16 train sources, 2 validation sources, 6 test sources.
- Excluded from source-aware split: `DNN` and `ML` selected merged files, because they no longer preserve original file/source boundaries.
- Prompt smoke samples: 32 JSONL examples from `Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`.
- Label inventory: 15 normalized labels.
- Main selected-ML stratified split seed: `20260531`.
- Main selected-ML split sizes: 126,233 train rows, 15,774 validation rows, 15,793 test rows.
- Main simulated FL client partition: `K=10`, Dirichlet label-skew `alpha=0.5`, generated from the selected-ML training split.

Label normalization currently applied:

```text
OS_Fingerprinting -> Fingerprinting
```

This is necessary because the raw `Attack traffic/OS_Fingerprinting_attack.csv` file uses `OS_Fingerprinting`, while both selected merged CSVs use `Fingerprinting`.

Generated SNLI summary:

```text
train  550,152
dev     10,000
test    10,000
```

## External Baseline Prepared

Fed-SB public code was cloned to:

```text
data/external/fed-sb
```

Prepared commit:

```text
e9e649822ce63437535f72fbf06c146a22b9e527
```

The official Fed-SB SNLI data location is connected by symlink:

```text
data/external/fed-sb/fed_sb/DP/SNLI/data/snli_1.0 -> data/raw/snli/snli_1.0
```

Server-side reproduction wrappers:

```bash
bash scripts/run_fedsb_snli_fed_private.sh
bash scripts/run_fedsb_snli_central_private.sh
```
