#!/bin/bash
# OptBinning 2D Auto Monotonic Grid Search (ImageNet)
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Runs expanded hyperparameter grid for 2D OptBinning (gini-margin) with:
#   - monotonic_trend='auto' (let OptBinning detect optimal trend per variable)
#   - Extended grid: n_clusters, max_n_prebins, min_bin_size
#
# Grid: 5 x 1 x 2 x 1 x 2 x 2 = 40 configurations per model
#   - n_clusters: [5, 10, 20, 30, 50]
#   - alpha: [0.05]
#   - score: [upper, mean]
#   - optbinning_monotonic: [auto]
#   - optbinning_max_n_prebins: [50, 100]
#   - optbinning_min_bin_size: [0.02, 0.05]
#
# Usage:
#   ./scripts/run_optbinning_2d_auto_grid.sh           # Run both models
#   ./scripts/run_optbinning_2d_auto_grid.sh tiny      # Run ViT-Tiny16 only
#   ./scripts/run_optbinning_2d_auto_grid.sh base      # Run ViT-Base16 only
#   ./scripts/run_optbinning_2d_auto_grid.sh status    # Check running jobs
#
# Results will be saved to:
#   results/partition_binning/imagenet/<model>/partition/runs/<run_tag>/

set -e

RUN_TAG="optbinning-2d-auto-grid-20260123"
DATASET_IMAGENET="configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

mkdir -p logs

run_tiny() {
    echo "=== Running OptBinning 2D Auto Grid: ViT-Tiny16 ==="
    echo "Run tag: ${RUN_TAG}"
    echo "Grid: 40 configurations (5 n_clusters x 2 score x 2 max_n_prebins x 2 min_bin_size)"
    echo ""

    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_optbinning_gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_2d_auto_vit_tiny16.log 2>&1 &
    PID=$!
    echo "Started ViT-Tiny16 with PID: $PID"
    echo "Monitor: tail -f logs/optbinning_2d_auto_vit_tiny16.log"
}

run_base() {
    echo "=== Running OptBinning 2D Auto Grid: ViT-Base16 ==="
    echo "Run tag: ${RUN_TAG}"
    echo "Grid: 40 configurations (5 n_clusters x 2 score x 2 max_n_prebins x 2 min_bin_size)"
    echo ""

    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_optbinning_gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_2d_auto_vit_base16.log 2>&1 &
    PID=$!
    echo "Started ViT-Base16 with PID: $PID"
    echo "Monitor: tail -f logs/optbinning_2d_auto_vit_base16.log"
}

check_status() {
    echo "=== Checking running optbinning 2D auto processes ==="
    ps aux | grep "[p]ython.*run_detection.*optbinning.*gini-margin" || echo "No optbinning 2D processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/optbinning_2d_auto_*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 5 lines) ---"
            tail -5 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-all}" in
    tiny)
        run_tiny
        ;;
    base)
        run_base
        ;;
    all)
        run_tiny
        echo ""
        run_base
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {tiny|base|all|status}"
        echo ""
        echo "OptBinning 2D Auto Monotonic Grid Search (ImageNet)"
        echo ""
        echo "Runs expanded grid for 2D OptBinning (gini-margin) with auto monotonic detection."
        echo "This should fix the issue where ascending trend was incorrectly applied to margin."
        echo ""
        echo "  tiny   - Run ViT-Tiny16 only"
        echo "  base   - Run ViT-Base16 only"
        echo "  all    - Run both models (default)"
        echo "  status - Check running processes"
        echo ""
        echo "Results: results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG}/"
        exit 1
        ;;
esac
