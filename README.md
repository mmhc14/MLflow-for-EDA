# MLflow for EDA

This repository contains a simple example of using **MLflow** to track
experiments during exploratory data analysis (EDA). The `pipeline.py`
script can train either an XGBoost regression model or a small LSTM
model while logging metrics and artifacts to MLflow.

## Requirements

- Python 3.12
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`, and
  `tensorflow` (optional, only needed for the LSTM model)

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the packages manually:

```bash
pip install pandas numpy scikit-learn xgboost mlflow tensorflow
```

## Usage

Run the pipeline with the desired model type. By default the XGBoost
regressor is used.

```bash
python pipeline.py --model xgboost
```

To train the LSTM model instead:

```bash
python pipeline.py --model lstm
```

The script automatically generates synthetic data, performs a minimal
EDA step that logs a CSV summary, trains the selected model and logs
metrics and the trained model to MLflow.

After running, start the MLflow UI to browse the logged results:

```bash
mlflow ui
```
