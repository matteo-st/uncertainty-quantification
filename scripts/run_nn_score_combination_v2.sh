#!/bin/bash
# Run NN Score Combination v2 experiment on ImageNet with ViT-Base16
# Key changes from v1:
#   - Uses 3 scores (gini, margin, msp) instead of 4 (removed entropy)
#   - Uses per-seed optimal hyperparameters from grid search runs
#   - Compares against individual score baselines

set -e

# GPU selection (default: GPU 1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Default values
RUN_TAG="nn-score-combination-v2-$(date +%Y%m%d)"
NO_MLFLOW="--no-mlflow"
RESULTS_FAMILY="score_combination"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --with-mlflow)
            NO_MLFLOW=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "NN Score Combination v2 Experiment"
echo "=========================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Run tag: ${RUN_TAG}"
echo "MLflow: $([ -z "$NO_MLFLOW" ] && echo 'enabled' || echo 'disabled')"
echo ""
echo "Config changes from v1:"
echo "  - Base scores: [gini, margin, msp] (removed entropy)"
echo "  - Hyperparameters: per-seed optimal from grid search"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run the experiment
error-estimation run \
    --config-dataset configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml \
    --config-model configs/models/imagenet_timm-vit-base16.yml \
    --config-detection configs/postprocessors/nn_score_combination/imagenet_vit_base16_nn.yml \
    --results-family "${RESULTS_FAMILY}" \
    ${NO_MLFLOW} \
    --run-tag "${RUN_TAG}"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results: results/${RESULTS_FAMILY}/imagenet/timm_vit_base16_ce/mlp/runs/${RUN_TAG}/"
echo "=========================================="
