import argparse
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.xgboost
import mlflow.pytorch
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


def load_data():
    """Load the diabetes dataset from scikit-learn."""

    data = load_diabetes()
    return data.data, data.target


def perform_eda(X: np.ndarray, y: np.ndarray):
    """Save and log a simple statistical summary of the dataset."""

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    summary = df.describe()
    summary_path = "eda_summary.csv"
    summary.to_csv(summary_path)
    mlflow.log_artifact(summary_path)
    for col in df.columns:
        mlflow.log_metric(f"{col}_mean", summary.loc["mean", col])
    os.remove(summary_path)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train an XGBoost regressor and log metrics to MLflow."""

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.xgboost.log_model(model, "model")


def to_sequence(X: np.ndarray) -> np.ndarray:
    """Reshape tabular features as sequences for the LSTM."""

    return X.reshape(X.shape[0], X.shape[1], 1)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
) -> None:
    """Train an LSTM model and log results to MLflow."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(input_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(test_tensor).cpu().numpy()
    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.pytorch.log_model(model, "model")


def main():
    parser = argparse.ArgumentParser(description="MLflow EDA pipeline")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm"],
        default="xgboost",
        help="Model type to train",
    )
    args = parser.parse_args()

    mlflow.set_experiment("eda_pipeline")
    with mlflow.start_run():
        X, y = load_data()
        perform_eda(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        if args.model == "xgboost":
            train_xgboost(X_train, y_train, X_test, y_test)
        else:
            train_lstm(
                to_sequence(X_train),
                y_train,
                to_sequence(X_test),
                y_test,
            )


if __name__ == "__main__":
    main()
