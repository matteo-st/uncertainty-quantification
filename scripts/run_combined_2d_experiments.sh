#!/bin/bash
# 2D Combined Scores Binning Experiments (ImageNet)
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Tests 2D score combinations:
#   - Doctor + MSP (gini-msp)
#   - Doctor + Margin (gini-margin)
#
# Usage:
#   ./scripts/run_combined_2d_experiments.sh gini-msp    # Run Doctor+MSP experiments
#   ./scripts/run_combined_2d_experiments.sh gini-margin # Run Doctor+Margin experiments
#   ./scripts/run_combined_2d_experiments.sh all         # Run all 2D experiments
#   ./scripts/run_combined_2d_experiments.sh status      # Check running jobs
#
# Results will be saved to:
#   results/partition_binning/imagenet/<model>/partition/runs/<run_tag>/

set -e

RUN_TAG_MSP="combined-gini-msp-20260121"
RUN_TAG_MARGIN="combined-gini-margin-20260121"
DATASET_IMAGENET="configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

mkdir -p logs

run_gini_msp() {
    echo "=== Running Doctor + MSP (2D) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_MSP}"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (gini-msp)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_combined-kmeans-gini-msp.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_MSP" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_gini_msp_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (gini-msp)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_combined-kmeans-gini-msp.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_MSP" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_gini_msp_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "Doctor + MSP experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/combined_gini_msp_vit_*.log"
}

run_gini_margin() {
    echo "=== Running Doctor + Margin (2D) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_MARGIN}"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_combined-kmeans-gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_gini_margin_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_combined-kmeans-gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_gini_margin_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "Doctor + Margin experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/combined_gini_margin_vit_*.log"
}

check_status() {
    echo "=== Checking running 2D combined scores processes ==="
    ps aux | grep "[p]ython.*run_detection.*combined-kmeans-gini" || echo "No 2D combined processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/combined_gini_*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 3 lines) ---"
            tail -3 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-}" in
    gini-msp|msp)
        run_gini_msp
        ;;
    gini-margin|margin)
        run_gini_margin
        ;;
    all)
        run_gini_msp
        echo ""
        run_gini_margin
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {gini-msp|gini-margin|all|status}"
        echo ""
        echo "2D Combined Scores Binning Experiments (ImageNet)"
        echo ""
        echo "  gini-msp    - Run Doctor + MSP experiments"
        echo "  gini-margin - Run Doctor + Margin experiments"
        echo "  all         - Run all 2D experiments"
        echo "  status      - Check running processes"
        echo ""
        echo "Results:"
        echo "  gini-msp:    results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_MSP}/"
        echo "  gini-margin: results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_MARGIN}/"
        exit 1
        ;;
esac
