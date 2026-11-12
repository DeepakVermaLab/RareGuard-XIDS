#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


SETUP = """from pathlib import Path
import os, sys

LOCAL_CANDIDATES = [
    Path('/Volumes/LocalDrive1/onlyscholar-projects/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION'),
]
DRIVE_CANDIDATES = [
    Path('/content/drive/MyDrive/projects/implementation/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION'),
    Path('/content/drive/MyDrive/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION'),
    Path('/content/gdrive/MyDrive/projects/implementation/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION'),
    Path('/content/gdrive/MyDrive/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION'),
]

def is_project_root(path: Path) -> bool:
    return (path / 'pyproject.toml').is_file() and (path / 'src' / 'rareguard').is_dir()

def find_project_root() -> Path | None:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents, *LOCAL_CANDIDATES, *DRIVE_CANDIDATES]:
        if is_project_root(candidate):
            return candidate
    if Path('/content').is_dir():
        from google.colab import drive
        for mountpoint in (Path('/content/drive'), Path('/content/gdrive')):
            if mountpoint.is_dir() and any(mountpoint.iterdir()):
                continue
            try:
                drive.mount(str(mountpoint), force_remount=False)
            except ValueError as exc:
                if 'must not already contain files' not in str(exc):
                    raise
                continue
            for candidate in DRIVE_CANDIDATES:
                if is_project_root(candidate):
                    return candidate
    return None

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    raise FileNotFoundError('Canonical RareGuard-XIDS IMPLEMENTATION project not found.')
os.chdir(PROJECT_ROOT)
PROJECT_SRC = str(PROJECT_ROOT / 'src')
sys.path.insert(0, PROJECT_SRC)
os.environ['PYTHONPATH'] = PROJECT_SRC + os.pathsep + os.environ.get('PYTHONPATH', '')
print('Project root:', PROJECT_ROOT)
print('Runtime:', 'Colab' if Path('/content').is_dir() else 'local')
print('Python path ready')"""


DEVICE_SPLIT = """import subprocess
import torch

DEVELOPMENT_DATASETS = [
    'cse_cic_ids2018',
    'edge_iiotset',
    'cic_bccc_nrc_tabulariotattack_2024',
    'cic_iiot_2025_datasense',
]
CONFIRMATION_DATASETS = ['ciciot2023', 'ton_iot']

if not torch.cuda.is_available():
    raise RuntimeError('A CUDA runtime is required for the development-dataset stage.')

def run_device_split_stage(module: str) -> None:
    for dataset in DEVELOPMENT_DATASETS:
        subprocess.run(
            [sys.executable, '-m', module, '--config', 'configs/publication_v2.yaml',
             '--datasets', 'configs/datasets_v2.yaml', '--dataset', dataset,
             '--device', 'cuda', '--resume'],
            check=True,
        )
    for dataset in CONFIRMATION_DATASETS:
        subprocess.run(
            [sys.executable, '-m', module, '--config', 'configs/publication_v2.yaml',
             '--datasets', 'configs/datasets_v2.yaml', '--dataset', dataset,
             '--device', 'cpu', '--resume'],
            check=True,
        )"""


NOTEBOOKS = {
    "15_v2_data_preparation.ipynb": [
        markdown("# 15 - V2 data preparation\n\nPrepare versioned leakage-safe splits. The original four datasets are corrective development evidence; CICIoT2023 and TON_IoT are reserved external confirmations."),
        code(SETUP),
        code("!python -m rareguard.experiments.prepare_publication_data --config configs/publication_v2.yaml --datasets configs/datasets_v2.yaml --dataset all --resume"),
        code("from pathlib import Path\nimport json\nfor path in sorted(Path('artifacts/publication_v2/processed').glob('*/split_metadata.json')):\n    data = json.loads(path.read_text())\n    print(path.parent.name, data['split_rows'], 'features=', data['features'], 'classes=', data['classes'], 'ready=', data['ready'])"),
    ],
    "16_v2_tuning_central.ipynb": [
        markdown("# 16 - V2 validation tuning and central evaluation\n\nTune on validation only, freeze each dataset configuration, then evaluate the locked test once."),
        code(SETUP),
        code("DEVICE = 'cpu'\n!python -m rareguard.experiments.run_publication_adaptive_tuning --config configs/publication_v2.yaml --datasets configs/datasets_v2.yaml --dataset all --device {DEVICE} --resume"),
        code("!python -m rareguard.experiments.run_publication_baselines --config configs/publication_v2.yaml --datasets configs/datasets_v2.yaml --dataset all --resume"),
        code(DEVICE_SPLIT + "\n\nrun_device_split_stage('rareguard.experiments.run_publication_central')"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication_v2/results/central_summary_ci.csv'))"),
    ],
    "17_v2_continual.ipynb": [
        markdown("# 17 - V2 continual prototype-herding evaluation\n\nCompare fine-tuning, reservoir replay, and balanced prototype-herding memory with validation-gated imprinting."),
        code(SETUP),
        code(DEVICE_SPLIT + "\n\nrun_device_split_stage('rareguard.experiments.run_publication_continual')"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication_v2/results/continual_seed_metrics.csv'))"),
    ],
    "18_v2_federated.ipynb": [
        markdown("# 18 - V2 federated evaluation\n\nRun the shared-backbone federated comparison on the versioned six-dataset protocol."),
        code(SETUP),
        code(DEVICE_SPLIT + "\n\nrun_device_split_stage('rareguard.experiments.run_publication_federated')"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication_v2/results/federated_seed_metrics.csv'))"),
    ],
    "19_v2_agentic.ipynb": [
        markdown("# 19 - V2 agentic escalation evaluation\n\nEvaluate validation-selected light/heavy routing using measured timing. No joule or production-uptime claim is made."),
        code(SETUP),
        code("!python -m rareguard.experiments.run_publication_agentic --config configs/publication_v2.yaml --dataset all"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication_v2/results/agentic_policy_metrics.csv'))"),
    ],
    "20_v2_robustness.ipynb": [
        markdown("# 20 - V2 robustness evaluation\n\nRun paired deterministic feature masking and bounded Gaussian perturbation tests."),
        code(SETUP),
        code(DEVICE_SPLIT + "\n\nrun_device_split_stage('rareguard.experiments.run_publication_robustness')"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication_v2/results/robustness_summary.csv'))"),
    ],
    "21_v2_publication_audit.ipynb": [
        markdown("# 21 - V2 publication evidence audit\n\nGenerate claim gates, paired tests, figures, and provenance from versioned evidence without changing model outputs."),
        code(SETUP),
        code("!python -m rareguard.experiments.audit_publication --config configs/publication_v2.yaml --datasets configs/datasets_v2.yaml"),
        code("from pathlib import Path\nprint(Path('artifacts/publication_v2/reports/PUBLICATION_EVIDENCE_AUDIT.md').read_text())"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RareGuard v2 publication notebooks.")
    parser.add_argument(
        "--notebook",
        action="append",
        choices=sorted(NOTEBOOKS),
        help="Generate only the selected notebook. Repeat to select multiple notebooks.",
    )
    args = parser.parse_args()
    selected = args.notebook or list(NOTEBOOKS)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name in selected:
        cells = NOTEBOOKS[name]
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "research (Python 3.11)", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (NOTEBOOK_DIR / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
