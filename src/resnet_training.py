"""ResNet-based regression training pipeline for tabular data.

This module refactors the original ResNet training script into a reusable
command-line program. It handles data preparation, model training,
metric reporting, and artifact persistence (model and scalers).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    """Configuration container for the training pipeline."""

    data_path: Path
    save_dir: Path
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    log_every: int = 20
    use_weighted_loss: bool = False
    loss_weights: Tuple[float, float, float] = (1.0, 10.0, 5.0)


class TabularDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapper for numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ResidualBlock1D(nn.Module):
    """Simple residual block for 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """ResNet18-inspired model for tabular regression."""

    def __init__(self, num_features: int, num_outputs: int = 3) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
        )
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_outputs)
        self.num_features = num_features

    def _make_layer(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock1D(in_channels, out_channels, stride)]
        layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.num_features:
            raise ValueError(f"Expected input of shape (batch, {self.num_features}), got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        return self.fc(x)


def set_deterministic(seed: int) -> None:
    """Set pseudo-random seeds for reproducibility."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mse = (pred - target) ** 2
    return (mse * weights).mean()


def prepare_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    train_loader = DataLoader(TabularDataset(X_train_scaled, y_train_scaled), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TabularDataset(X_test_scaled, y_test_scaled), batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y


def run_training(config: TrainingConfig) -> None:
    set_deterministic(config.random_state)

    if not config.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.data_path}")

    data_frame = pd.read_csv(config.data_path)
    input_cols = [
        "L1",
        "N1",
        "T",
        "L2",
        "N2",
        "N_PWELL",
        "N_SUB",
        "L_JFET",
        "GP",
    ]
    output_cols = ["BV", "QGD", "RON"]

    missing_inputs = set(input_cols) - set(data_frame.columns)
    missing_outputs = set(output_cols) - set(data_frame.columns)
    if missing_inputs or missing_outputs:
        missing = ", ".join(sorted(missing_inputs | missing_outputs))
        raise KeyError(f"Dataset is missing required columns: {missing}")

    X = data_frame[input_cols].to_numpy(dtype=np.float32)
    y = data_frame[output_cols].to_numpy(dtype=np.float32)

    train_loader, test_loader, scaler_X, scaler_y = prepare_dataloaders(X, y, config)

    device = get_device()
    model = ResNet1D(num_features=X.shape[1], num_outputs=y.shape[1]).to(device)

    criterion = nn.MSELoss()
    weight_tensor: torch.Tensor | None = None
    if config.use_weighted_loss:
        weight_tensor = torch.tensor(config.loss_weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            if weight_tensor is not None:
                loss = weighted_mse_loss(predictions, y_batch, weight_tensor)
            else:
                loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                if weight_tensor is not None:
                    loss = weighted_mse_loss(predictions, y_batch, weight_tensor)
                else:
                    loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)

        if epoch == 1 or epoch % config.log_every == 0:
            print(f"[Epoch {epoch:3d}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    metrics = evaluate(model, test_loader, scaler_y, device)
    print_metrics(metrics)

    save_dir = config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "resnet1d_model.pt")
    joblib.dump(scaler_X, save_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y.pkl")

    metrics_path = save_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nArtifacts saved to: {save_dir.resolve()}")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    scaler_y: StandardScaler,
    device: torch.device,
) -> dict:
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(y_batch.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    y_true_inv = scaler_y.inverse_transform(y_true)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    overall_metrics = {
        "MSE": float(mean_squared_error(y_true_inv, y_pred_inv)),
        "MAPE": float(mean_absolute_percentage_error(y_true_inv, y_pred_inv)),
        "R2": float(r2_score(y_true_inv, y_pred_inv)),
    }

    target_names = ["BV", "QGD", "RON"]
    for idx, name in enumerate(target_names):
        overall_metrics[name] = {
            "MSE": float(mean_squared_error(y_true_inv[:, idx], y_pred_inv[:, idx])),
            "MAPE": float(mean_absolute_percentage_error(y_true_inv[:, idx], y_pred_inv[:, idx])),
            "R2": float(r2_score(y_true_inv[:, idx], y_pred_inv[:, idx])),
        }
    return overall_metrics


def print_metrics(metrics: dict) -> None:
    print("\nTest set performance")
    print(f" - MSE : {metrics['MSE']:.6f}")
    print(f" - MAPE: {metrics['MAPE'] * 100:.2f}%")
    print(f" - R²   : {metrics['R2']:.4f}")
    for name in ("BV", "QGD", "RON"):
        component = metrics[name]
        print(f"\n[{name}]")
        print(f"   MSE : {component['MSE']:.6f}")
        print(f"   MAPE: {component['MAPE'] * 100:.2f}%")
        print(f"   R²   : {component['R2']:.4f}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a ResNet1D model on tabular data.")
    parser.add_argument("data_path", type=Path, help="Path to the CSV dataset.")
    parser.add_argument("save_dir", type=Path, help="Directory to store the trained model and scalers.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam (default: 1e-5)")
    parser.add_argument("--log-every", type=int, default=20, help="Logging interval in epochs (default: 20)")
    parser.add_argument(
        "--use-weighted-loss",
        action="store_true",
        help="Apply per-target weights to the MSE loss.",
    )
    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs=3,
        default=(1.0, 10.0, 5.0),
        metavar=("BV", "QGD", "RON"),
        help="Weights for the weighted MSE loss (default: 1.0 10.0 5.0)",
    )
    args = parser.parse_args()

    return TrainingConfig(
        data_path=args.data_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_every=args.log_every,
        use_weighted_loss=args.use_weighted_loss,
        loss_weights=tuple(args.loss_weights),
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
