#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.splitlines()]}


SETUP = """from pathlib import Path
import os, sys

PROJECT_ROOT = Path('/Volumes/LocalDrive1/onlyscholar-projects/RareGuard-XIDS[DeepakSir]/IMPLEMENTATION')
if not (PROJECT_ROOT / 'pyproject.toml').is_file():
    raise FileNotFoundError(f'Canonical local project is unavailable: {PROJECT_ROOT}')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
print('Project root:', PROJECT_ROOT)
print('Use the local VS Code research kernel for this publication stage.')"""


NOTEBOOKS = {
    "08_publication_data.ipynb": [
        markdown("# 08 - Leakage-safe publication data\n\nCPU stage. Range-sample the canonical Drive CSV sources, remove duplicate and identifier leakage, split first, and fit preprocessing on training rows only."),
        code(SETUP),
        code("!python -m rareguard.experiments.prepare_publication_data --config configs/publication.yaml --datasets configs/datasets.yaml --dataset all --resume"),
        code("from pathlib import Path\nimport json\nfor path in sorted(Path('artifacts/publication/processed').glob('*/split_metadata.json')):\n    data=json.loads(path.read_text())\n    print(path.parent.name, data['split_rows'], 'features=', data['features'], 'classes=', data['classes'], 'ready=', data['ready'])"),
    ],
    "09_publication_central.ipynb": [
        markdown("# 09 - Repeated central experiments\n\nCPU by default. Five deterministic seeds, validation-only checkpoint selection, one locked-test evaluation, NOVA-XIDS baseline, RareGuard-XIDS, and real component ablations."),
        code(SETUP),
        code("DEVICE = 'cpu'  # Change to 'mps' only if CPU runtime is impractical.\n!python -m rareguard.experiments.run_publication_central --config configs/publication.yaml --datasets configs/datasets.yaml --dataset all --device {DEVICE} --resume"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication/results/central_summary_ci.csv'))"),
    ],
    "10_publication_federated.ipynb": [
        markdown("# 10 - Fair federated experiments\n\nAll methods use the same RareGuard-XIDS backbone. Round selection uses validation data; the locked test is evaluated once."),
        code(SETUP),
        code("DEVICE = 'cpu'  # 'mps' is allowed if needed for this compute-heavy stage.\n!python -m rareguard.experiments.run_publication_federated --config configs/publication.yaml --datasets configs/datasets.yaml --dataset all --device {DEVICE} --resume"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication/results/federated_seed_metrics.csv'))"),
    ],
    "11_publication_agentic.ipynb": [
        markdown("# 11 - 24/7 agentic escalation evaluation\n\nCPU stage. NOVA-XIDS is the always-on detector and RareGuard-XIDS is invoked only by uncertainty, rare-signal, or drift gates. Runtime projections use measured inference timings; no joule claim is made."),
        code(SETUP),
        code("!python -m rareguard.experiments.run_publication_agentic --config configs/publication.yaml --dataset all"),
        code("import pandas as pd\ndisplay(pd.read_csv('artifacts/publication/results/agentic_policy_metrics.csv'))"),
    ],
    "12_publication_audit.ipynb": [
        markdown("# 12 - Publication evidence audit\n\nCPU stage. Generate figures, paired-seed confidence intervals, completeness checks, and explicit claim gates. This stage never changes model outputs."),
        code(SETUP),
        code("!python -m rareguard.experiments.audit_publication --config configs/publication.yaml --datasets configs/datasets.yaml"),
        code("from pathlib import Path\nprint(Path('artifacts/publication/reports/PUBLICATION_EVIDENCE_AUDIT.md').read_text())"),
    ],
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
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
