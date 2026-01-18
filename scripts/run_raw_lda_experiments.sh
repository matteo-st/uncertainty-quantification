#!/bin/bash
#
# Run Raw LDA experiments for all dataset/model combinations
#
# This script evaluates raw LDA score combinations (without binning):
# - Individual scores: gini, margin, msp
# - Combinations: gini+margin, gini+msp, margin+msp, gini+margin+msp
#
# Prerequisites:
# - Doctor/margin/odin grid search must be run first (for per-score hyperparameters)
#   Run with: --mode search_res to select best hyperparameters per seed
#
# Usage:
#   ./scripts/run_raw_lda_experiments.sh [RUN_TAG]
#
# Example:
#   ./scripts/run_raw_lda_experiments.sh raw-lda-20260118
#

set -e

# Configuration
RUN_TAG="${1:-raw-lda-$(date +%Y%m%d-%H%M)}"
SEEDS="1 2 3 4 5 6 7 8 9"

# Datasets and models
DATASETS=("cifar10" "cifar100")
MODELS_CIFAR10=("resnet34" "densenet121")
MODELS_CIFAR100=("resnet34" "densenet121")

echo "========================================"
echo "Raw LDA Experiments"
echo "========================================"
echo "Run tag: ${RUN_TAG}"
echo "Seeds: ${SEEDS}"
echo ""

# Function to run experiment for a dataset/model combination
run_experiment() {
    local dataset=$1
    local model=$2

    echo "----------------------------------------"
    echo "Running: ${dataset} / ${model}"
    echo "----------------------------------------"

    python -m error_estimation.experiments.run_detection \
        --config-dataset "configs/datasets/${dataset}/${dataset}_raw_lda_allseeds.yml" \
        --config-model "configs/models/${dataset}_${model}.yml" \
        --config-detection "configs/postprocessors/raw_lda/all_combinations.yml" \
        --mode eval_grid \
        --eval-grid \
        --run-tag "${RUN_TAG}" \
        --no-mlflow

    echo "Completed: ${dataset} / ${model}"
    echo ""
}

# Run all combinations
for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "cifar10" ]; then
        models=("${MODELS_CIFAR10[@]}")
    else
        models=("${MODELS_CIFAR100[@]}")
    fi

    for model in "${models[@]}"; do
        run_experiment "$dataset" "$model"
    done
done

echo "========================================"
echo "All experiments completed!"
echo "Results saved with tag: ${RUN_TAG}"
echo "========================================"
echo ""
echo "Results locations:"
echo "  results/baselines/cifar10/resnet34_ce/raw_lda/runs/${RUN_TAG}/"
echo "  results/baselines/cifar10/densenet121_ce/raw_lda/runs/${RUN_TAG}/"
echo "  results/baselines/cifar100/resnet34_ce/raw_lda/runs/${RUN_TAG}/"
echo "  results/baselines/cifar100/densenet121_ce/raw_lda/runs/${RUN_TAG}/"
