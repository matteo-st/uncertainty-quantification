#!/bin/bash
# Minimal MLP Ablation Test (1 config, 1 seed)
# Tests that the self.results fix works before running full 126-config grid

set -e

# GPU selection (default: GPU 1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Default values
RUN_TAG="mlp-ablation-test"
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
echo "MLP Ablation TEST (Minimal)"
echo "=========================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Run tag: ${RUN_TAG}"
echo "MLflow: $([ -z "$NO_MLFLOW" ] && echo 'enabled' || echo 'disabled')"
echo ""
echo "Testing: 1 config (linear model), 1 seed split"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run the minimal test (1 config, 1 seed)
error-estimation run \
    --config-dataset configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml \
    --config-model configs/models/imagenet_timm-vit-base16.yml \
    --config-detection configs/postprocessors/nn_score_combination/imagenet_vit_base16_nn_ablation_test.yml \
    --results-family "${RESULTS_FAMILY}" \
    --save-search-results \
    --seed-splits 1 \
    ${NO_MLFLOW} \
    --run-tag "${RUN_TAG}"

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Results: results/${RESULTS_FAMILY}/imagenet/timm_vit_base16_ce/mlp/runs/${RUN_TAG}/"
echo ""
echo "If this worked, run the full ablation:"
echo "  bash scripts/run_nn_ablation.sh"
echo "=========================================="
