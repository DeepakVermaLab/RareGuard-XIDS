# RareGuard-XIDS Publication Protocol

## Incremental Positioning

The predecessor is NOVA-XIDS, which used a 64-dimensional latent representation, CBAM residual convolution, LSTM temporal modeling, self-attention pooling, focal loss, explainability, and ONNX deployment. RareGuard-XIDS retains a NOVA-compatible backbone and adds a normalized rare-class embedding, prototype memory, supervised contrastive learning, rare-aware federated aggregation, and a NOVA-to-RareGuard escalation controller.

The historical NOVA-XIDS values in `configs/first_paper_reference.yaml` are context only. A direct claim of improvement requires a protocol-matched rerun on the same full split or paired evaluation against the implemented NOVA-XIDS baseline.

## Confirmatory Design

- Five fixed seeds: 42, 73, 101, 137, and 211.
- Duplicate removal before splitting.
- Identifier and timestamp columns excluded from predictors.
- Stratified train/validation/test split.
- Imputation and scaling fitted only on training rows.
- Validation macro-F1 selects checkpoints and federated rounds.
- The locked test split is evaluated exactly once per trained run.
- NOVA-XIDS and RareGuard-XIDS use the same input data and core backbone.
- NOVA-XIDS retains the documented class-balanced focal objective; it is not weakened to cross-entropy for comparison.
- Seed-level predictions, per-class metrics, runtime, memory, and model size are retained.
- Report mean, standard deviation, 95% confidence interval, and paired seed gains.
- Report paired Wilcoxon tests, paired effect sizes, and Holm-adjusted p-values across confirmatory metrics.

## Required Metrics

Primary metrics are macro-F1 and rare-class macro-F1. Secondary metrics are weighted-F1, balanced accuracy, MCC, Cohen's kappa, macro OvR ROC-AUC, macro PR-AUC, and 15-bin expected calibration error.

## Ablations

The full RareGuard-XIDS model is compared with fixed variants that remove focal loss, supervised contrastive loss, or the prototype head. Results are never modified if an ablation performs better; the audit reports that outcome and fails the superiority gate when appropriate.

## Federated Controls

FedAvg, FedProx, clustered FL, and RareGuardFed all use the RareGuard-XIDS backbone. Communication is reported in bytes. RareGuardFed uses observed client training throughput and rare-class share; it does not use randomized energy values.

## Agentic Evidence

NOVA-XIDS is the always-on light detector and RareGuard-XIDS is the escalated detector. The controller is compared with always-light and always-heavy policies using the same locked test predictions. Runtime projections use measured inference times. No energy-in-joules claim is permitted without an external hardware power meter.

## Evidence Boundary

Stages 00-07 and `artifacts/smoke/` validate execution only. Paper claims must come from `artifacts/publication/` and must pass `PUBLICATION_READINESS.json`.

The CPU-feasible protocol samples deterministic byte ranges from every canonical Drive CSV, caps each class, and records the source manifest and hashes. It is an all-file bounded range-sample study, not a full-corpus reproduction. The audit labels that evidence separately and forbids wording that implies every source row was processed.
