from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rareguard.agent.runtime import ContinuousRareGuardRuntime, RuntimePaths
from rareguard.config import load_yaml
from rareguard.experiments.adaptive_config import apply_dataset_hyperparameters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the resumable RareGuard-XIDS monitoring service.")
    parser.add_argument("--config", default="configs/publication.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inbox", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    root = Path(config["paths"]["artifact_root"])
    config = apply_dataset_hyperparameters(config, root, args.dataset)
    policy_path = root / "reports" / f"AGENT_POLICY_{args.dataset}.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        selected = policy.get("per_seed", {}).get(str(args.seed), policy.get("deployment_default", {}))
        config = copy.deepcopy(config)
        config["agent"].update(selected)
    processed = root / "processed" / args.dataset
    model_dir = root / "models" / args.dataset / f"seed_{args.seed}"
    metadata = json.loads((processed / "split_metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(processed / "splits.npz", allow_pickle=True)
    feature_names = [str(value) for value in arrays["feature_names"]]
    runtime_root = root / "agent_runtime" / args.dataset
    runtime = ContinuousRareGuardRuntime(
        config=config,
        metadata=metadata,
        feature_names=feature_names,
        transformer_path=processed / "train_fitted_transformer.joblib",
        nova_model_path=model_dir / "NOVA-XIDS.pt",
        rareguard_model_path=model_dir / "RareGuard-XIDS.pt",
        paths=RuntimePaths(
            inbox=Path(args.inbox),
            decisions=runtime_root / "decisions.jsonl",
            state=runtime_root / "state.json",
            heartbeat=runtime_root / "heartbeat.json",
        ),
        device=args.device,
    )
    if args.continuous:
        try:
            runtime.run_forever(args.poll_seconds)
        except KeyboardInterrupt:
            return 0
    else:
        print(json.dumps(runtime.run_once(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
