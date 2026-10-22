# RareGuard-XIDS Implementation

RareGuard-XIDS is the planned second PhD paper implementation. It is incremental over the first-paper NOVA-XIDS baseline and adds:

- rare-attack-aware embedding;
- incremental/prototype memory;
- federated learning with non-IID clients;
- energy-aware 24/7 agentic monitoring;
- CPU-first experiments with optional GPU.

## Local Paths

Project:

```text
/Volumes/LocalDrive1/onlyscholar-projects/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION
```

Datasets:

```text
/Volumes/LocalDrive1/Datasets/Security
```

Google Drive output folder:

```text
https://drive.google.com/drive/folders/1Oh9o3ipRfNZmeyOl2uatBSHdGrnvnfUc
```

## Stage Notebooks

Run in order:

1. `notebooks/00_environment_check.ipynb`
2. `notebooks/01_dataset_discovery.ipynb`
3. `notebooks/02_preprocess_features.ipynb`
4. `notebooks/03_baseline_falconid.ipynb`
5. `notebooks/04_rareguard_embedding.ipynb`
6. `notebooks/05_federated_learning.ipynb`
7. `notebooks/06_agentic_energy_controller.ipynb`
8. `notebooks/07_ablation_and_report.ipynb`

Stages 00-07 are retained as pipeline smoke tests. They are not publication evidence because their preprocessing was fitted before splitting and their original federated comparison mixed architectures.

Publication protocol:

9. `notebooks/08_publication_data.ipynb` (CPU)
10. `notebooks/09_publication_central.ipynb` (CPU; MPS only if required)
11. `notebooks/10_publication_federated.ipynb` (CPU; MPS recommended only if CPU is impractical)
12. `notebooks/11_publication_agentic.ipynb` (CPU)
13. `notebooks/12_publication_audit.ipynb` (CPU)

Publication outputs are isolated under `artifacts/publication/`. Completed smoke outputs are preserved under `artifacts/smoke/`.

Each notebook writes logs/results under `artifacts/` and can sync to Drive with `scripts/sync_to_drive.py`.

Set `DATASET_KEY` inside a notebook to run one dataset only, or keep `all` to run every configured dataset.

## CPU First

All configs default to CPU and sampling limits. Increase `sample_rows_per_dataset`, `epochs`, and `federated.rounds` only after smoke tests pass.

## VS Code / Colab Connector Note

If a notebook runs on a Colab remote runtime, it cannot read the Mac-only path `/Volumes/LocalDrive1/...`. Either run the notebook with a local VS Code Python kernel on the Mac, or copy/sync this `IMPLEMENTATION` folder to Google Drive at:

```text
/content/drive/MyDrive/projects/implementation/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION
```

The notebooks now auto-detect the local project root first, then common Google Drive locations.

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Smoke Test

```bash
python3 scripts/env_check.py
python3 -m rareguard.experiments.dataset_discovery --config configs/default.yaml
```

## Evidence Boundary

The implementation is CPU-first. Only results that pass `artifacts/publication/reports/PUBLICATION_READINESS.json` may be used as paper evidence. The audit forbids energy-in-joules claims unless an external hardware power meter supplies real measurements. Ablations are reported exactly as measured, including any case where a reduced model outperforms the full model.
