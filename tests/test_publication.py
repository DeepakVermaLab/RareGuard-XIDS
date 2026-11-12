from __future__ import annotations

import numpy as np
import torch

from rareguard.data.publication import _cat_range, compact_source_frame, list_remote_csvs, parquet_compatible, sample_remote_csv, source_filename_labels
from rareguard.evaluation.metrics import classification_metrics
from rareguard.experiments.audit_publication import agent_policy_value, continual_superiority, full_beats_ablations, holm_adjust, paired_statistical_tests, robustness_superiority, selected_matches_adaptive_tuning
from rareguard.experiments.run_publication_agentic import projected_seconds, select_policy_thresholds
from rareguard.experiments.run_publication_central import apply_temperature, fit_temperature
from rareguard.experiments.run_publication_continual import class_incremental_tasks, prototype_herding_memory, update_memory
from rareguard.experiments.run_publication_robustness import perturb_features
from rareguard.models.publication_models import NovaXIDSNet, RareGuardXIDSNet


def test_publication_models_share_output_contract() -> None:
    x = torch.randn(6, 12)
    nova = NovaXIDSNet(12, 3)
    rareguard = RareGuardXIDSNet(12, 3)
    rareguard_without_direct_path = RareGuardXIDSNet(12, 3, use_direct_path=False)
    assert nova(x).shape == (6, 3)
    logits, vectors = rareguard(x)
    assert logits.shape == (6, 3)
    assert vectors.shape == (6, 64)
    assert rareguard_without_direct_path(x)[0].shape == (6, 3)


def test_prototype_branch_starts_conservatively() -> None:
    model = RareGuardXIDSNet(
        12,
        3,
        prototype_initial_scale=1.0,
        prototype_gate_bias=-2.0,
    )
    assert torch.isclose(model.prototype_log_scale.exp(), torch.tensor(1.0))
    with torch.no_grad():
        gate = model.gate(torch.zeros(4, 64))
    assert float(gate.max()) < 0.13


def test_publication_metrics_include_rare_and_calibration() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    metrics = classification_metrics(np.asarray([0, 1, 1]), probabilities, {1})
    assert metrics["f1_macro"] == 1.0
    assert metrics["rare_f1_macro"] == 1.0
    assert 0.0 <= metrics["ece_15"] <= 1.0


def test_ablation_gate_uses_preregistered_primary_composite() -> None:
    import pandas as pd

    rows = []
    for seed in (42, 73):
        rows.extend(
            [
                {"dataset": "demo", "seed": seed, "model": "RareGuard-XIDS", "f1_macro": 0.79, "rare_f1_macro": 0.75},
                {"dataset": "demo", "seed": seed, "model": "RareGuard-no-focal", "f1_macro": 0.80, "rare_f1_macro": 0.60},
            ]
        )
    passed, comparisons = full_beats_ablations(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 3


def test_paired_statistics_and_holm_adjustment() -> None:
    import pandas as pd

    rows = []
    for seed, gain in zip((42, 73, 101, 137, 211), (0.01, 0.02, 0.03, 0.01, 0.02)):
        for model, offset in (("NOVA-XIDS", 0.0), ("RareGuard-XIDS", gain)):
            rows.append(
                {
                    "dataset": "demo",
                    "seed": seed,
                    "model": model,
                    "f1_macro": 0.7 + offset,
                    "rare_f1_macro": 0.5 + offset,
                    "balanced_accuracy": 0.65 + offset,
                    "mcc": 0.6 + offset,
                    "ece_15": 0.1 - offset,
                }
            )
    result = paired_statistical_tests(pd.DataFrame(rows))
    assert len(result) == 5
    assert result["holm_adjusted_p"].notna().all()
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_selected_configuration_must_match_saved_validation_winner() -> None:
    import pandas as pd

    rows = []
    for focal_mix, score in ((0.0, 0.75), (0.5, 0.80)):
        for seed in (42, 73, 101):
            rows.append(
                {
                    "dataset": "demo",
                    "seed": seed,
                    "focal_mix": focal_mix,
                    "rare_sampler_power": 0.0,
                    "contrastive_weight": 0.1,
                    "validation_selection_score": score,
                    "validation_f1_macro": score + 0.05,
                    "validation_rare_f1_macro": score - 0.05,
                    "test_access_count": 0,
                }
            )
    selected = {
        "dataset": "demo",
        "focal_mix": 0.5,
        "rare_sampler_power": 0.0,
        "contrastive_weight": 0.1,
        "validation_selection_score": 0.80,
        "test_access_count": 0,
    }
    assert selected_matches_adaptive_tuning(pd.DataFrame(rows), {"demo": selected}, ["demo"])
    selected["focal_mix"] = 0.0
    assert not selected_matches_adaptive_tuning(pd.DataFrame(rows), {"demo": selected}, ["demo"])


def test_confirmation_configuration_is_frozen_from_development_only() -> None:
    import pandas as pd

    rows = []
    for dataset, offset in (("dev_a", 0.0), ("dev_b", 0.02)):
        for focal_mix, score in ((0.0, 0.70 + offset), (0.5, 0.80 + offset)):
            for seed in (42, 73):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "focal_mix": focal_mix,
                        "rare_sampler_power": 0.0,
                        "contrastive_weight": 0.1,
                        "validation_selection_score": score,
                        "validation_f1_macro": score + 0.05,
                        "validation_rare_f1_macro": score - 0.05,
                        "test_access_count": 0,
                    }
                )
    selected = {
        dataset: {
            "dataset": dataset,
            "focal_mix": 0.5,
            "rare_sampler_power": 0.0,
            "contrastive_weight": 0.1,
            "validation_selection_score": 0.80 + offset,
            "test_access_count": 0,
        }
        for dataset, offset in (("dev_a", 0.0), ("dev_b", 0.02))
    }
    selected["confirm"] = {
        "dataset": "confirm",
        "focal_mix": 0.5,
        "rare_sampler_power": 0.0,
        "contrastive_weight": 0.1,
        "validation_selection_score": 0.81,
        "selection_source": "development_datasets_only_global_mean",
        "selection_datasets": ["dev_a", "dev_b"],
        "test_access_count": 0,
    }

    assert selected_matches_adaptive_tuning(
        pd.DataFrame(rows),
        selected,
        ["dev_a", "dev_b", "confirm"],
        {"dev_a", "dev_b"},
        {"confirm"},
    )
    contaminated = pd.concat(
        [pd.DataFrame(rows), pd.DataFrame([{**rows[0], "dataset": "confirm"}])],
        ignore_index=True,
    )
    assert not selected_matches_adaptive_tuning(
        contaminated,
        selected,
        ["dev_a", "dev_b", "confirm"],
        {"dev_a", "dev_b"},
        {"confirm"},
    )


def test_parquet_compatibility_normalizes_mixed_objects() -> None:
    import pandas as pd

    frame = pd.DataFrame({"mixed": pd.Series([1, b"2", None], dtype=object), "numeric": [1.0, 2.0, 3.0]})
    normalized = parquet_compatible(frame)
    assert str(normalized["mixed"].dtype) == "string"
    assert normalized["numeric"].dtype == frame["numeric"].dtype


def test_source_filename_labels_preserve_attack_types() -> None:
    import pandas as pd

    labels = source_filename_labels(pd.Series(["root/DoS UDP Flood.csv", "root/Benign Traffic.csv"]))
    assert labels.tolist() == ["DoS UDP Flood", "Benign Traffic"]


def test_mounted_drive_backend_lists_and_reads_csv_ranges(tmp_path) -> None:
    source = tmp_path / "Demo" / "csv" / "sample.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"label,value\nbenign,1\nattack,2\n")
    (source.parent / "._ignored.csv").write_bytes(b"metadata")

    remote = tmp_path.as_uri()
    files = list_remote_csvs(remote, "Demo", "unused-for-file-backend", "csv")

    assert [(item.path, item.size) for item in files] == [("Demo/csv/sample.csv", source.stat().st_size)]
    assert _cat_range(remote, files[0].path, "unused-for-file-backend", 6, 5) == b"value"


def test_source_frame_compaction_is_balanced_and_deterministic() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "feature": np.arange(300),
            "Label": ["common"] * 240 + ["rare_a"] * 45 + ["rare_b"] * 15,
        }
    )
    first = compact_source_frame(frame, ["Label"], 30, 42)
    second = compact_source_frame(frame, ["Label"], 30, 42)

    assert first.index.tolist() == second.index.tolist()
    assert first["Label"].value_counts().to_dict() == {"common": 10, "rare_a": 10, "rare_b": 10}


def test_full_file_sampling_reads_each_csv_row_once(tmp_path) -> None:
    source = tmp_path / "Demo" / "sample.csv"
    source.parent.mkdir(parents=True)
    source.write_text("label,value\nbenign,1\nattack,2\n", encoding="utf-8")
    remote_file = list_remote_csvs(tmp_path.as_uri(), "Demo", "unused")[0]

    sampled = sample_remote_csv(tmp_path.as_uri(), remote_file, "unused", 8, 4, full_file=True)

    assert sampled[["label", "value"]].to_dict("records") == [
        {"label": "benign", "value": 1},
        {"label": "attack", "value": 2},
    ]


def test_feature_hash_duplicates_are_removed_before_split() -> None:
    import tempfile
    from pathlib import Path

    import pandas as pd

    from rareguard.data.publication import prepare_train_only_splits

    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "feature_a": np.concatenate([rng.normal(0, 0.1, 20), rng.normal(2, 0.1, 20)]),
            "feature_b": np.concatenate([rng.normal(0, 0.1, 20), rng.normal(2, 0.1, 20)]),
            "Label": ["benign"] * 20 + ["attack"] * 20,
        }
    )
    near_duplicate = frame.iloc[[0]].copy()
    near_duplicate["feature_a"] += 1e-8
    frame = pd.concat([frame, near_duplicate], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw.parquet"
        frame.to_parquet(raw, index=False)
        metadata = prepare_train_only_splits(raw, "Label", root, 42, 0.2, 0.1, 6, 10.0)
        arrays = np.load(root / "splits.npz")
    assert metadata["ready"]
    assert metadata["feature_duplicate_rows_removed"] == 1
    assert all(value == 0 for value in metadata["duplicate_hash_overlap"].values())
    assert max(float(np.abs(arrays[name]).max()) for name in ("X_train", "X_val", "X_test")) <= 10.0


def test_continual_tasks_and_balanced_memory() -> None:
    labels = np.asarray([0] * 20 + [1] * 10 + [2] * 5 + [3] * 2)
    tasks = class_incremental_tasks(labels, 3)
    assert sorted(value for task in tasks for value in task) == [0, 1, 2, 3]
    memory = update_memory("RareGuard-Continual", np.arange(len(labels)), labels, 8, np.random.default_rng(42))
    counts = np.bincount(labels[memory], minlength=4)
    assert counts.tolist() == [2, 2, 2, 2]


def test_prototype_herding_memory_is_balanced_and_deterministic() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(24, 12)).astype(np.float32)
    y = np.repeat(np.arange(3), 8)
    model = RareGuardXIDSNet(12, 3, prototype_initial_scale=1.0, prototype_gate_bias=-2.0)
    first = prototype_herding_memory(model, x, y, np.arange(24), 9, 8, "cpu")
    second = prototype_herding_memory(model, x, y, np.arange(24), 9, 8, "cpu")
    assert np.array_equal(first, second)
    assert np.bincount(y[first], minlength=3).tolist() == [3, 3, 3]


def test_continual_superiority_requires_better_rare_f1_and_forgetting() -> None:
    import pandas as pd

    rows = []
    for method, rare_f1, forgetting in (
        ("RareGuard-FineTune", 0.4, 0.30),
        ("RareGuard-ReservoirReplay", 0.5, 0.20),
        ("RareGuard-Continual", 0.6, 0.10),
    ):
        rows.append(
            {
                "dataset": "demo",
                "seed": 42,
                "method": method,
                "rare_f1_macro": rare_f1,
                "mean_validation_forgetting": forgetting,
            }
        )
    passed, comparisons = continual_superiority(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 4


def test_agent_timing_and_policy_gate() -> None:
    import pandas as pd

    assert projected_seconds("always_light_nova", 10, 0, 0.1, 0.5) == 1.0
    assert projected_seconds("always_heavy_rareguard", 10, 10, 0.1, 0.5) == 5.0
    assert projected_seconds("rareguard_agent", 10, 2, 0.1, 0.5) == 2.0
    rows = [
        {"dataset": "demo", "policy": "always_light_nova", "f1_macro": 0.7, "rare_f1_macro": 0.4, "heavy_call_ratio": 0.0},
        {"dataset": "demo", "policy": "always_heavy_rareguard", "f1_macro": 0.9, "rare_f1_macro": 0.8, "heavy_call_ratio": 1.0},
        {"dataset": "demo", "policy": "rareguard_agent", "f1_macro": 0.8, "rare_f1_macro": 0.6, "heavy_call_ratio": 0.3},
    ]
    passed, comparisons = agent_policy_value(pd.DataFrame(rows))
    assert passed
    assert len(comparisons) == 3


def test_agent_thresholds_are_selected_from_validation_only() -> None:
    y = np.asarray([0, 0, 1, 1])
    light = np.asarray([[0.9, 0.1], [0.55, 0.45], [0.6, 0.4], [0.2, 0.8]])
    heavy = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]])
    settings = {
        "uncertainty_grid": [0.2, 0.4],
        "rare_probability_grid": [0.3, 0.7],
        "drift_grid": [0.5],
        "min_heavy_call_ratio": 0.1,
        "max_heavy_call_ratio": 0.9,
        "compute_penalty": 0.0,
    }
    selected = select_policy_thresholds(y, light, heavy, {1}, settings)
    assert selected["validation_composite"] >= 0.5
    assert 0.1 <= selected["validation_heavy_call_ratio"] <= 0.9


def test_agent_thresholds_adapt_when_fixed_grid_over_escalates() -> None:
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0])
    light = np.asarray(
        [
            [0.51, 0.49], [0.55, 0.45], [0.60, 0.40], [0.45, 0.55],
            [0.40, 0.60], [0.35, 0.65], [0.30, 0.70], [0.80, 0.20],
        ]
    )
    heavy = np.asarray(
        [
            [0.90, 0.10], [0.85, 0.15], [0.80, 0.20], [0.10, 0.90],
            [0.15, 0.85], [0.20, 0.80], [0.10, 0.90], [0.90, 0.10],
        ]
    )
    settings = {
        "uncertainty_grid": [0.0],
        "rare_probability_grid": [0.0],
        "drift_grid": [0.0],
        "adaptive_quantiles": [0.50, 0.75, 0.90],
        "min_heavy_call_ratio": 0.1,
        "max_heavy_call_ratio": 0.9,
        "compute_penalty": 0.0,
    }
    selected = select_policy_thresholds(y, light, heavy, {1}, settings)
    assert selected["threshold_grid_source"] == "configured_plus_validation_quantiles"
    assert 0.1 <= selected["validation_heavy_call_ratio"] <= 0.9


def test_robustness_perturbations_are_deterministic_and_bounded() -> None:
    x = np.ones((10, 5), dtype=np.float32)
    first = perturb_features(x, "feature_mask", 0.2, 42, 10.0)
    second = perturb_features(x, "feature_mask", 0.2, 42, 10.0)
    noisy = perturb_features(x, "gaussian_noise", 50.0, 42, 10.0)
    assert np.array_equal(first, second)
    assert (first == 0).any()
    assert float(np.abs(noisy).max()) <= 10.0

    import pandas as pd

    rows = []
    for seed in (42, 73):
        rows.extend(
            [
                {"dataset": "demo", "seed": seed, "model": "NOVA-XIDS", "perturbation": "feature_mask", "strength": 0.1, "f1_macro": 0.7, "rare_f1_macro": 0.5},
                {"dataset": "demo", "seed": seed, "model": "RareGuard-XIDS", "perturbation": "feature_mask", "strength": 0.1, "f1_macro": 0.8, "rare_f1_macro": 0.6},
            ]
        )
    passed, comparisons = robustness_superiority(pd.DataFrame(rows))
    assert passed and len(comparisons) == 1


def test_temperature_scaling_is_validation_only_and_preserves_predictions() -> None:
    probabilities = np.asarray([[0.99, 0.01], [0.90, 0.10], [0.10, 0.90], [0.01, 0.99]])
    labels = np.asarray([0, 1, 1, 1])
    temperature = fit_temperature(probabilities, labels)
    scaled = apply_temperature(probabilities, temperature)
    assert temperature > 1.0
    assert np.array_equal(probabilities.argmax(axis=1), scaled.argmax(axis=1))
    assert np.allclose(scaled.sum(axis=1), 1.0)
