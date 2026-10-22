from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from rareguard.models.torch_models import FalconIDNet, FocalLoss, RareGuardNet, supervised_contrastive_loss


@dataclass
class TorchTrainResult:
    dataset: str
    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    model_path: str
    vectors_path: str


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str, rareguard: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in _loader(x, y, 2048, False):
            xb = xb.to(device)
            if rareguard:
                logits, z = model(xb)
            else:
                logits = model(xb)
                z = model.encode(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            vectors.append(z.cpu().numpy())
    return np.concatenate(preds), np.concatenate(vectors)


def train_torch_model(
    dataset: str,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    focal_gamma: float = 2.0,
    contrastive_temperature: float = 0.2,
    device: str = "cpu",
) -> TorchTrainResult:
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    classes = int(np.max(y_train)) + 1
    input_dim = x_train.shape[1]
    rareguard = model_name == "RareGuard"
    model = RareGuardNet(input_dim, classes, latent_dim, hidden_dim, dropout) if rareguard else FalconIDNet(input_dim, classes, latent_dim, hidden_dim, dropout)
    model.to(device)
    criterion = FocalLoss(gamma=focal_gamma) if rareguard else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _loader(x_train, y_train, batch_size, True)
    best_val = -1.0
    best_state = None
    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if rareguard:
                logits, z = model(xb)
                loss = criterion(logits, yb) + 0.1 * supervised_contrastive_loss(z, yb, contrastive_temperature)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_pred, _ = evaluate(model, x_val, y_val, device, rareguard)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_val:
            best_val = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    pred, vectors = evaluate(model, x_test, y_test, device, rareguard)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{dataset}_{model_name}.pt"
    vectors_path = out / f"{dataset}_{model_name}_test_vectors.npz"
    torch.save(model.state_dict(), model_path)
    np.savez_compressed(vectors_path, vectors=vectors, y=y_test, pred=pred)

    result = TorchTrainResult(
        dataset=dataset,
        model=model_name,
        accuracy=float(accuracy_score(y_test, pred)),
        precision_macro=float(precision_score(y_test, pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_test, pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        model_path=str(model_path),
        vectors_path=str(vectors_path),
    )
    pd.DataFrame([asdict(result)]).to_csv(out / f"{dataset}_{model_name}_metrics.csv", index=False)
    return result

