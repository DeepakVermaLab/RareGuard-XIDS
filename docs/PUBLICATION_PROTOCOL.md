# RareGuard-XIDS Publication Protocol

## Incremental Positioning

The predecessor is NOVA-XIDS, which used a 64-dimensional latent representation, CBAM residual convolution, LSTM temporal modeling, self-attention pooling, focal loss, explainability, and ONNX deployment. RareGuard-XIDS retains a NOVA-compatible backbone and adds a learned dual-path embedding that fuses the inherited contextual encoder with a direct tabular residual path, scaled class-prototype memory, supervised contrastive learning, rare-aware federated aggregation, and a NOVA-to-RareGuard escalation controller.

The historical NOVA-XIDS values in `configs/first_paper_reference.yaml` are context only. A direct claim of improvement requires a protocol-matched rerun on the same full split or paired evaluation against the implemented NOVA-XIDS baseline.

## Confirmatory Design

- CSE-CIC-IDS2018 and Edge-IIoTset are development datasets. TabularIoTAttack-2024 and DataSense 2025 are untouched external confirmations.
- A prespecified six-candidate adaptation grid is applied independently to each dataset using validation data only. It selects the focal/CE mix and rare-sampling strength while keeping the proposed contrastive objective active. The grid, three selection seeds, primary composite, and tie-break rule are fixed before either confirmation test split is opened.
- MPS is restricted to validation-only tuning and execution smoke tests; reported central, continual, and federated metrics run on CPU with deterministic PyTorch algorithms enabled.
- Five fixed seeds: 42, 73, 101, 137, and 211.
- Duplicate removal before splitting.
- Multiclass attack targets are preserved; for TabularIoTAttack-2024, the canonical per-attack CSV filename supplies the class because its internal `Label` field is binary.
- Identifier and timestamp columns excluded from predictors.
- Stratified train/validation/test split.
- Imputation and robust scaling are fitted only on training rows, followed by a fixed `[-10, 10]` clip to bound extreme flow-rate outliers for stable neural optimization.
- The mean of validation macro-F1 and rare-class macro-F1 selects central checkpoints; validation macro-F1 selects continual checkpoints and federated rounds.
- Validation-only temperature scaling calibrates NOVA and RareGuard probabilities before the locked test and agent-routing analysis; raw and calibrated probabilities are both retained.
- The locked test split is evaluated exactly once per trained run.
- NOVA-XIDS and RareGuard-XIDS use the same input data and core backbone.
- NOVA-XIDS retains the documented class-balanced focal objective; it is not weakened to cross-entropy for comparison.
- Seed-level predictions, per-class metrics, runtime, memory, and model size are retained.
- Report mean, standard deviation, 95% confidence interval, and paired seed gains.
- Report paired Wilcoxon tests, paired effect sizes, and Holm-adjusted p-values across confirmatory metrics.
- Freeze a provenance manifest containing the Git state, runtime versions, config hashes, and SHA-256 inventory of publication artifacts.
- The adaptation grid never reads test arrays; its output records `test_access_count = 0`, and the audit verifies one complete grid plus a frozen selection artifact for every dataset.

## Required Metrics

Primary metrics are macro-F1 and rare-class macro-F1. Secondary metrics are weighted-F1, balanced accuracy, MCC, Cohen's kappa, macro OvR ROC-AUC, macro PR-AUC, and 15-bin expected calibration error.

## Ablations

The full RareGuard-XIDS model is compared with fixed variants that remove both effective-number weighting and rare-sample reweighting, supervised contrastive loss, the prototype head, or the direct tabular residual path. Loss settings are selected by the same validation-only adaptation rule and then frozen per dataset. The preregistered ablation endpoint is the mean of macro-F1 and rare-class macro-F1; both component metrics remain reported. Results are never modified if an ablation performs better; the audit reports that outcome and fails the superiority gate when appropriate.

## Continual-Learning Controls

Class-incremental tasks introduce frequency-ordered attack groups using training data only. Fine-tuning without memory, fixed-size reservoir replay, and RareGuard class-balanced prototype replay use the same backbone and optimization objective. Validation data measures forgetting after each task, and the locked test is evaluated once after all classes have been introduced.

## Federated Controls

FedAvg, FedProx, clustered FL, and RareGuardFed all use the RareGuard-XIDS backbone. Communication is reported in bytes. RareGuardFed uses observed client training throughput and rare-class share; it does not use randomized energy values.

## Agentic Evidence

NOVA-XIDS is the always-on light detector and RareGuard-XIDS is the escalated detector. Uncertainty, rare-signal, and drift thresholds are selected from a fixed grid using saved validation predictions and a preregistered accuracy-compute objective before the locked test is opened. The controller is compared with always-light and always-heavy policies using the same locked test predictions. Runtime projections use measured inference times. No energy-in-joules claim is permitted without an external hardware power meter.

The deployable runtime watches an input directory, validates the train-fitted feature schema, applies NOVA first, invokes RareGuard only when uncertainty, rare-class probability, or online drift crosses a configured threshold, and persists decisions, file hashes, errors, and health heartbeats. The offline policy experiment supports routing-quality claims; it does not by itself prove uninterrupted 24/7 field operation.

## Evidence Boundary

Stages 00-07 and `artifacts/smoke/` validate execution only. Paper claims must come from `artifacts/publication/` and must pass `PUBLICATION_READINESS.json`.

The CPU-feasible protocol samples deterministic byte ranges from every canonical Drive CSV, caps each class, and records the source manifest and hashes. It is an all-file bounded range-sample study, not a full-corpus reproduction. The audit labels that evidence separately and forbids wording that implies every source row was processed.
