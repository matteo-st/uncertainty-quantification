#!/bin/bash
#
# Run Score Combination experiments for all dataset/model combinations
#
# This script evaluates whether combining uncertainty scores (via LDA) can
# improve raw score quality. This is NOT binning/certification - it's about
# improving the uncertainty score itself.
#
# Data split: res=0, cal=5000, test=5000
# Results stored in: results/score_combination/<dataset>/<model>/<postprocessor>/
#
# Phase 1: Run individual score grids (doctor, odin, margin)
# Phase 2: Run raw_lda combinations (loads best hyperparameters from Phase 1)
#
# Usage:
#   ./scripts/run_score_combination_experiments.sh [RUN_TAG_PREFIX]
#
# Example:
#   ./scripts/run_score_combination_experiments.sh sc-20260118
#

set -e

# Configuration
RUN_TAG_PREFIX="${1:-sc-$(date +%Y%m%d)}"
SEEDS="1 2 3 4 5 6 7 8 9"

# Datasets and models
DATASETS=("cifar10" "cifar100")
MODELS_CIFAR10=("resnet34" "densenet121")
MODELS_CIFAR100=("resnet34" "densenet121")

echo "========================================"
echo "Score Combination Experiments"
echo "========================================"
echo "Run tag prefix: ${RUN_TAG_PREFIX}"
echo "Seeds: ${SEEDS}"
echo ""
echo "Phase 1: Individual score grids"
echo "Phase 2: LDA combinations"
echo ""

# Function to run individual score grid
run_score_grid() {
    local dataset=$1
    local model=$2
    local score=$3

    echo "----------------------------------------"
    echo "Running: ${dataset} / ${model} / ${score}"
    echo "----------------------------------------"

    python -m error_estimation.experiments.run_detection \
        --config-dataset "configs/datasets/${dataset}/${dataset}_score_combination.yml" \
        --config-model "configs/models/${dataset}_${model}.yml" \
        --config-detection "configs/postprocessors/${score}/score_combination_grid.yml" \
        --results-family score_combination \
        --mode eval_grid \
        --eval-grid \
        --run-tag "${score}-grid-${RUN_TAG_PREFIX}" \
        --no-mlflow

    echo "Completed: ${dataset} / ${model} / ${score}"
    echo ""
}

# Function to run LDA combinations
run_lda_combinations() {
    local dataset=$1
    local model=$2

    echo "----------------------------------------"
    echo "Running LDA combinations: ${dataset} / ${model}"
    echo "----------------------------------------"

    python -m error_estimation.experiments.run_detection \
        --config-dataset "configs/datasets/${dataset}/${dataset}_score_combination.yml" \
        --config-model "configs/models/${dataset}_${model}.yml" \
        --config-detection "configs/postprocessors/score_combination/raw_lda_combinations.yml" \
        --results-family score_combination \
        --mode eval_grid \
        --eval-grid \
        --run-tag "raw-lda-${RUN_TAG_PREFIX}" \
        --no-mlflow

    echo "Completed LDA combinations: ${dataset} / ${model}"
    echo ""
}

# Phase 1: Run individual score grids
echo "========================================"
echo "PHASE 1: Individual Score Grids"
echo "========================================"

SCORES=("doctor" "msp" "margin")

for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "cifar10" ]; then
        models=("${MODELS_CIFAR10[@]}")
    else
        models=("${MODELS_CIFAR100[@]}")
    fi

    for model in "${models[@]}"; do
        for score in "${SCORES[@]}"; do
            run_score_grid "$dataset" "$model" "$score"
        done
    done
done

echo ""
echo "Phase 1 completed. Individual score grids saved."
echo ""

# Phase 2: Run LDA combinations
echo "========================================"
echo "PHASE 2: LDA Combinations"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "cifar10" ]; then
        models=("${MODELS_CIFAR10[@]}")
    else
        models=("${MODELS_CIFAR100[@]}")
    fi

    for model in "${models[@]}"; do
        run_lda_combinations "$dataset" "$model"
    done
done

echo "========================================"
echo "All experiments completed!"
echo "Run tag prefix: ${RUN_TAG_PREFIX}"
echo "========================================"
echo ""
echo "Results locations:"
echo "  results/score_combination/cifar10/resnet34_ce/"
echo "  results/score_combination/cifar10/densenet121_ce/"
echo "  results/score_combination/cifar100/resnet34_ce/"
echo "  results/score_combination/cifar100/densenet121_ce/"
echo ""
echo "Each contains: doctor/, odin/, margin/, raw_lda/"
