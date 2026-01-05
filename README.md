# Toward Failure Detection with Statistical Guarantees

## Setup
- Python 3.11.
- Create a virtual environment and install dependencies:
  `python -m venv .venv && source .venv/bin/activate && pip install -e .`
  (or `pip install -r requirements.txt` if you prefer a non-editable install).
- Checkpoints are available at:
  `https://github.com/edadaltocg/relative-uncertainty/releases/tag/checkpoints`

## Configuration
- Dataset and checkpoint roots are controlled via env vars:
  - `DATA_DIR` (default `./data`)
  - `CHECKPOINTS_DIR` (default `./checkpoints`)
- MLflow logging uses:
  - `MLFLOW_TRACKING_URI` (remote server)
  - `MLFLOW_EXPERIMENT_NAME` (default `error-estimation`)

## Running Experiments
- Main run:
  `error-estimation run --config-dataset configs/datasets/cifar10/cifar10_ablation.yml \
    --config-model configs/models/cifar10_resnet34.yml \
    --config-detection configs/postprocessors/clustering/cifar10_resnet34_gmm.yml`
- n_cal ablation:
  `error-estimation ablation --n-cal 1000 2000 3000`
- Hyperparameter ablation:
  `error-estimation ablation-hyperparams --config-detection configs/postprocessors/clustering/cifar10_resnet34_gmm.yml`
- Use `--dry-run` to validate configs without loading data or checkpoints.

## Project Layout
- `src/error_estimation/` core package and experiment runners.
- `configs/` YAML configs for datasets, models, and postprocessors.
- `scripts/` shell entry points for paper runs and ablations.
