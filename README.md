# SPECTRA-FedCore

SPECTRA-FedCore is a research codebase for a public-spectral, client-level differentially private federated core-adaptation method for Edge-IIoT intrusion detection.

The repository is currently structured to make the server handoff reproducible before large GPU experiments start:

- `src/fedllm_data/`: dataset manifests, label normalization, selected-CSV split generation, prompt smoke samples, and client partitions.
- `src/spectra/`: CPU-testable NumPy core utilities for public SVD bases, spectral core adapters, client-side Gaussian release accounting, FL aggregation, and metrics.
- `scripts/prepare_datasets.py`: regenerates portable dataset artifacts.
- `HANDOFF.md`: server execution guide.
- `docs/superpowers/specs/2026-05-31-spectra-dp-fedcore-design.md`: paper-level design.

## Quick Check

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m compileall src scripts
pytest -q
```

## Data Preparation

Raw datasets are intentionally not committed. After preparing `data/raw/edgeiiotset/full_dataset` and `data/raw/snli/current`, regenerate artifacts with:

```bash
PYTHONPATH=src python scripts/prepare_datasets.py \
  --edge-root data/raw/edgeiiotset/full_dataset \
  --snli-root data/raw/snli/current \
  --out-dir data/processed \
  --count-rows \
  --relative-paths
```

Important generated artifacts:

- `data/processed/edgeiiot/file_manifest.json`
- `data/processed/edgeiiot/label_inventory.json`
- `data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json`
- `data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json`
- `data/processed/snli/manifest.json`

## Experiment Protocol

Main paper protocol:

- Dataset: `ML-EdgeIIoT-dataset.csv`
- Task: normalized 15-class closed-set intrusion classification
- Split: 80/10/10 stratified row-level split, fixed seed `20260531`
- FL simulation: one server process, `K=10` clients, IID sanity plus Dirichlet label-skew `alpha=0.5`
- Privacy: client-level release DP, not record-level DP

The method novelty should be framed around public-backbone spectral bases, budget-aware rank/noise allocation, and client-side DP utility recovery. Do not claim that training a square core alone is novel; Fed-SB is the closest baseline and must be evaluated seriously.
