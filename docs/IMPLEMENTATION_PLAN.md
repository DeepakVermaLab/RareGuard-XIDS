# RareGuard-XIDS Implementation Plan

## Goal

Build the second-paper implementation as an incremental extension of the first FALCON-ID direction: rare-attack-aware embedding, federated non-IID learning, and an energy-aware 24/7 agentic monitoring controller.

## Stage Contract

1. Environment check: verify Python packages, dataset path, artifact path, and `rclone`.
2. Dataset discovery: profile each dataset folder and inspect candidate label columns.
3. Preprocessing: create cached numeric matrices, label maps, and rare-class profiles.
4. Baselines: train classical CPU-first models for sanity and comparison.
5. FALCON-ID/RareGuard embedding: train the first-paper-style neural baseline and the proposed rare-aware embedding model.
6. Federated stage: create reproducible non-IID client partitions and run measured FedAvg, FedProx, CFL-style clustered aggregation, and RareGuardFed training.
7. Agentic stage: evaluate controller triggers and energy proxy using trained RareGuard probabilities when available. Fallback probability runs are clearly labeled and should not be used as final paper evidence.
8. Ablation/report: generate comparison tables, figures, and an implementation audit.

## Evidence Rules

- Measured results can be used only when generated from logs under `artifacts/`.
- Fallback outputs must be labeled in paper notes and cannot be reported as final empirical superiority.
- Final manuscript tables should cite the artifact CSV path and notebook stage that produced each value.
- Drive sync is an artifact backup step; it does not prove scientific validity.

## Outputs

- Processed features: `artifacts/processed/<dataset>/features.npz`
- Feature vectors: `artifacts/models/*_test_vectors.npz`
- Metrics: `artifacts/results/*.csv`
- Figures: `artifacts/figures/*.png`
- Audit: `artifacts/reports/IMPLEMENTATION_AUDIT.md`
