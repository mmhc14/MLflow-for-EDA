# MLflow-for-EDA

This repository demonstrates a simple MLflow pipeline for exploratory data analysis (EDA) and model training. The script uses the Diabetes dataset from `scikit-learn` and supports training either an XGBoost regressor or a small LSTM model implemented with PyTorch.

## Requirements

- Python 3.12
- `mlflow`
- `pandas`
- `scikit-learn`
- `xgboost`
- `torch`

Install the dependencies with pip:

```bash
pip install mlflow pandas scikit-learn xgboost torch
```

## Usage

Run the pipeline choosing the desired model:

```bash
python eda_mlflow_pipeline.py --model xgboost
```

or

```bash
python eda_mlflow_pipeline.py --model lstm
```

The script logs dataset statistics as MLflow metrics, trains the selected model, and stores the trained model in the current MLflow experiment.
