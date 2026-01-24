#!/bin/bash
# MLP Architecture Ablation Study
# Tests 126 configurations: 7 architectures × 3 weights × 3 dropouts × 2 activations

set -e

# GPU selection (default: GPU 1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Default values
RUN_TAG="mlp-ablation-$(date +%Y%m%d)"
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
echo "MLP Architecture Ablation Study"
echo "=========================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Run tag: ${RUN_TAG}"
echo "MLflow: $([ -z "$NO_MLFLOW" ] && echo 'enabled' || echo 'disabled')"
echo ""
echo "Grid search: 7 architectures × 3 weights × 3 dropouts × 2 activations = 126 configs"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run the ablation experiment with grid evaluation
error-estimation run \
    --config-dataset configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml \
    --config-model configs/models/imagenet_timm-vit-base16.yml \
    --config-detection configs/postprocessors/nn_score_combination/imagenet_vit_base16_nn_ablation.yml \
    --results-family "${RESULTS_FAMILY}" \
    --eval-grid \
    ${NO_MLFLOW} \
    --run-tag "${RUN_TAG}"

echo ""
echo "=========================================="
echo "Ablation completed!"
echo "Results: results/${RESULTS_FAMILY}/imagenet/timm_vit_base16_ce/mlp/runs/${RUN_TAG}/"
echo "Check grid_results.csv for all 126 configurations"
echo "=========================================="
