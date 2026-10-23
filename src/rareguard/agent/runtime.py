from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from rareguard.experiments.run_publication_central import Variant, build_model, resolve_device


@dataclass(frozen=True)
class RuntimePaths:
    inbox: Path
    decisions: Path
    state: Path
    heartbeat: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContinuousRareGuardRuntime:
    def __init__(
        self,
        config: dict,
        metadata: dict,
        feature_names: list[str],
        transformer_path: Path,
        nova_model_path: Path,
        rareguard_model_path: Path,
        paths: RuntimePaths,
        device: str = "cpu",
    ):
        self.config = config
        self.metadata = metadata
        self.feature_names = feature_names
        self.transformer = joblib.load(transformer_path)
        self.paths = paths
        self.device = resolve_device(device)
        classes = int(metadata["classes"])
        input_dim = len(feature_names)
        self.nova = build_model(Variant("NOVA-XIDS", False), input_dim, classes, config).to(self.device)
        self.rareguard = build_model(Variant("RareGuard-XIDS", True), input_dim, classes, config).to(self.device)
        self.nova.load_state_dict(torch.load(nova_model_path, map_location=self.device, weights_only=True))
        self.rareguard.load_state_dict(torch.load(rareguard_model_path, map_location=self.device, weights_only=True))
        self.nova.eval()
        self.rareguard.eval()
        self.reference: np.ndarray | None = None
        self.processed = self._load_state()

    def _load_state(self) -> dict[str, str]:
        if not self.paths.state.exists():
            return {}
        data = json.loads(self.paths.state.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in data.get("processed", {}).items()}

    def _save_state(self) -> None:
        self.paths.state.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": utc_now(), "processed": self.processed}
        self.paths.state.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_input(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        else:
            raise ValueError(f"Unsupported input format: {path}")
        missing = [feature for feature in self.feature_names if feature not in frame.columns]
        if missing:
            raise ValueError(f"Input {path.name} is missing {len(missing)} required features: {missing[:10]}")
        numeric = frame[self.feature_names].apply(pd.to_numeric, errors="coerce")
        return pd.DataFrame(self.transformer.transform(numeric), columns=self.feature_names)

    def _predict(self, model: torch.nn.Module, features: np.ndarray, rareguard: bool) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            output = model(tensor)
            logits = output[0] if rareguard else output
            key = "rareguard_temperature" if rareguard else "nova_temperature"
            temperature = max(float(self.config["agent"].get(key, 1.0)), 1e-6)
            return torch.softmax(logits / temperature, dim=1).cpu().numpy()

    def _drift(self, probabilities: np.ndarray, decay: float = 0.99) -> np.ndarray:
        scores = np.zeros(len(probabilities), dtype=float)
        for index, current in enumerate(probabilities):
            if self.reference is None:
                self.reference = current.copy()
            scores[index] = 0.5 * float(np.abs(current - self.reference).sum())
            self.reference = decay * self.reference + (1.0 - decay) * current
        return np.clip(scores, 0.0, 1.0)

    def process_file(self, path: Path) -> int:
        digest = file_sha256(path)
        if self.processed.get(str(path.resolve())) == digest:
            return 0
        transformed = self._read_input(path)
        features = transformed.to_numpy(dtype=np.float32)
        light = self._predict(self.nova, features, False)
        drift = self._drift(light)
        counts = np.asarray(self.metadata["split_class_counts"]["train"])
        cutoff = max(1, int(float(self.config["data"]["min_rare_class_ratio"]) * counts.sum()))
        rare_ids = np.flatnonzero(counts <= cutoff)
        settings = self.config["agent"]
        uncertainty = 1.0 - light.max(axis=1)
        rare_probability = light[:, rare_ids].max(axis=1) if len(rare_ids) else np.zeros(len(light))
        escalate = (
            (uncertainty >= float(settings["uncertainty_threshold"]))
            | (rare_probability >= float(settings["rare_probability_threshold"]))
            | (drift >= float(settings["drift_threshold"]))
        )
        uncertainty_trigger = uncertainty >= float(settings["uncertainty_threshold"])
        rare_trigger = rare_probability >= float(settings["rare_probability_threshold"])
        drift_trigger = drift >= float(settings["drift_threshold"])
        final = light.copy()
        if escalate.any():
            final[escalate] = self._predict(self.rareguard, features[escalate], True)
        self.paths.decisions.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.decisions.open("a", encoding="utf-8") as handle:
            for index in range(len(features)):
                record = {
                    "timestamp": utc_now(),
                    "source": str(path.resolve()),
                    "source_sha256": digest,
                    "row": index,
                    "action": "invoke_rareguard" if escalate[index] else "keep_nova",
                    "prediction": int(final[index].argmax()),
                    "confidence": float(final[index].max()),
                    "uncertainty": float(uncertainty[index]),
                    "rare_probability": float(rare_probability[index]),
                    "drift_score": float(drift[index]),
                    "trigger_reason": ";".join(
                        reason
                        for reason, active in (
                            ("uncertainty", uncertainty_trigger[index]),
                            ("rare_attack_signal", rare_trigger[index]),
                            ("traffic_drift", drift_trigger[index]),
                        )
                        if active
                    )
                    or "none",
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.processed[str(path.resolve())] = digest
        self._save_state()
        return len(features)

    def run_once(self) -> dict[str, int | str]:
        self.paths.inbox.mkdir(parents=True, exist_ok=True)
        files = sorted(
            path
            for path in [*self.paths.inbox.glob("*.csv"), *self.paths.inbox.glob("*.parquet")]
            if not path.name.startswith("._")
        )
        processed_files = 0
        processed_rows = 0
        failures = 0
        for path in files:
            try:
                rows = self.process_file(path)
                processed_files += int(rows > 0)
                processed_rows += rows
            except Exception as exc:
                failures += 1
                error_path = self.paths.decisions.with_name("agent_errors.jsonl")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"timestamp": utc_now(), "source": str(path), "error": str(exc)}) + "\n")
        heartbeat = {
            "timestamp": utc_now(),
            "status": "healthy" if failures == 0 else "degraded",
            "processed_files": processed_files,
            "processed_rows": processed_rows,
            "failures": failures,
            "device": self.device,
        }
        self.paths.heartbeat.parent.mkdir(parents=True, exist_ok=True)
        self.paths.heartbeat.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
        return heartbeat

    def run_forever(self, poll_seconds: float) -> None:
        while True:
            self.run_once()
            time.sleep(max(0.1, poll_seconds))
