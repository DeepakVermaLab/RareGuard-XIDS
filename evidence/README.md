# Final-v2 evidence package

This directory contains the curated, GitHub-safe evidence exported from the canonical final-v2 artifact root.

| Directory | Contents |
|---|---|
| `results/` | Seed-level, summary, statistical, ablation, robustness, continual, federated and agentic tables |
| `logs/` | Saved stage-driver and notebook execution logs |
| `reports/` | Readiness, provenance, selected-hyperparameter and agent-policy reports |
| `figures/` | Publication figures generated from saved result tables |

Raw datasets, processed tensors, predictions and model binaries remain outside Git. `SHA256SUMS` provides an integrity manifest for every file in this evidence package.

The package is complete as an execution record, but its scientific status is `NOT_READY`: 12 of 19 readiness gates pass. Consult `reports/PUBLICATION_EVIDENCE_AUDIT.md` before making any claim.
