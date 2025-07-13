import argparse
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Optional heavy imports for LSTM
try:
    from tensorflow import keras
except Exception:
    keras = None
try:
    import xgboost as xgb
except Exception:
    xgb = None


def generate_regression_data(samples=1000, features=20, noise=0.1):
    X, y = make_regression(
        n_samples=samples, n_features=features, noise=noise, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def generate_timeseries(length=1000):
    t = np.arange(length)
    data = np.sin(0.02 * t) + 0.5 * np.random.randn(length)
    return pd.DataFrame({"value": data})


def run_xgboost():
    X, y = generate_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {"objective": "reg:squarederror"}
    model = xgb.train(params, dtrain)
    preds = model.predict(dtest)
    # scikit-learn 1.7 removed the ``squared`` parameter from
    # :func:`mean_squared_error`. Compute the root mean squared error
    # manually for compatibility.
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.xgboost.log_model(model, artifact_path="model")


def run_lstm():
    if keras is None:
        raise ImportError("TensorFlow/Keras not available")
    df = generate_timeseries()
    values = df["value"].values
    sequence_length = 20
    X = []
    y = []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])
    X = np.array(X)[:, :, None]
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, 1)),
            keras.layers.LSTM(16),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=0
    )
    loss = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("loss", loss)
    mlflow.keras.log_model(model, artifact_path="model")


def save_eda(df, prefix="eda"):
    desc = df.describe().to_csv()
    path = f"{prefix}_summary.csv"
    with open(path, "w") as f:
        f.write(desc)
    mlflow.log_artifact(path)


def main(model_type: str):
    mlflow.set_experiment("eda-pipeline")
    with mlflow.start_run():
        if model_type == "xgboost":
            X, y = generate_regression_data()
            save_eda(pd.concat([X, y.rename("target")], axis=1))
            run_xgboost()
        elif model_type == "lstm":
            df = generate_timeseries()
            save_eda(df)
            run_lstm()
        else:
            raise ValueError(f"Unknown model type {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow EDA pipeline")
    parser.add_argument("--model", choices=["xgboost", "lstm"], default="xgboost")
    args = parser.parse_args()
    main(args.model)
