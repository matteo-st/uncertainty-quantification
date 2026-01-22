#!/bin/bash
# OptBinning Experiments (ImageNet)
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Tests OptBinning method for supervised optimal binning with respect to error labels.
# Partition fitted on CAL split (20000 samples).
#
# Score spaces:
#   - Doctor (gini) 1D
#   - Doctor + Margin (gini-margin) 2D - uses BinningProcess + WoE sum
#
# Usage:
#   ./scripts/run_optbinning_experiments.sh gini         # Run 1D Doctor experiments
#   ./scripts/run_optbinning_experiments.sh gini-margin  # Run 2D Doctor+Margin experiments
#   ./scripts/run_optbinning_experiments.sh all          # Run all experiments
#   ./scripts/run_optbinning_experiments.sh status       # Check running jobs
#
# Results will be saved to:
#   results/partition_binning/imagenet/<model>/partition/runs/<run_tag>/

set -e

RUN_TAG_GINI="optbinning-gini-cal-fit-20260122"
RUN_TAG_GINI_MARGIN="optbinning-gini-margin-cal-fit-20260122"
DATASET_IMAGENET="configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

mkdir -p logs

run_gini() {
    echo "=== Running OptBinning (1D Doctor) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_GINI}"
    echo "Partition fit split: CAL (20000 samples)"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (optbinning gini)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_optbinning_gini.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_gini_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (optbinning gini)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_optbinning_gini.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_gini_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "OptBinning (gini) experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/optbinning_gini_vit_*.log"
}

run_gini_margin() {
    echo "=== Running OptBinning (2D Doctor+Margin) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_GINI_MARGIN}"
    echo "Partition fit split: CAL (20000 samples)"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (optbinning gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_optbinning_gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_gini_margin_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (optbinning gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_optbinning_gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/optbinning_gini_margin_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "OptBinning (gini-margin) experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/optbinning_gini_margin_vit_*.log"
}

check_status() {
    echo "=== Checking running optbinning processes ==="
    ps aux | grep "[p]ython.*run_detection.*optbinning" || echo "No optbinning processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/optbinning_*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 3 lines) ---"
            tail -3 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-}" in
    gini)
        run_gini
        ;;
    gini-margin|margin)
        run_gini_margin
        ;;
    all)
        run_gini
        echo ""
        run_gini_margin
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {gini|gini-margin|all|status}"
        echo ""
        echo "OptBinning Experiments (ImageNet)"
        echo ""
        echo "Tests OptBinning for supervised optimal binning with respect to error labels."
        echo "Partition fitted on CAL split (20000 samples)."
        echo ""
        echo "  gini        - Run 1D Doctor experiments"
        echo "  gini-margin - Run 2D Doctor+Margin experiments"
        echo "  all         - Run all experiments"
        echo "  status      - Check running processes"
        echo ""
        echo "Results:"
        echo "  gini:        results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_GINI}/"
        echo "  gini-margin: results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_GINI_MARGIN}/"
        exit 1
        ;;
esac
