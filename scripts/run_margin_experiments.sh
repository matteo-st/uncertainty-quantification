#!/bin/bash
# Margin Baseline + Uniform Mass Binning Experiments
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Usage:
#   ./scripts/run_margin_experiments.sh step1    # Run margin baseline (grid search)
#   ./scripts/run_margin_experiments.sh step2    # Run uniform mass binning
#   ./scripts/run_margin_experiments.sh status   # Check running jobs
#
# Results will be saved to:
#   Step 1: results/<dataset>/<model>/margin/runs/<run_tag>/
#   Step 2: results/<dataset>/<model>/partition/runs/<run_tag>/

set -e

RUN_TAG_BASELINE="margin-baseline-20260120"
RUN_TAG_BINNING="margin-unif-mass-20260120"
DATASET_CIFAR10="configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000_all-seeds.yml"
DATASET_CIFAR100="configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000_all-seeds.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

# Ensure logs directory exists
mkdir -p logs

run_step1() {
    echo "=== Step 1: Running Margin Baseline with Grid Evaluation ==="
    echo "Results: results/<dataset>/<model>/margin/runs/${RUN_TAG_BASELINE}/"
    echo "Starting 4 experiments in parallel..."

    # CIFAR-10 ResNet-34
    echo "[1/4] CIFAR-10 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_resnet34.yml \
        --config-detection configs/postprocessors/margin/cifar10_resnet34_hyperparams_search.yml \
        --mode search_res --eval-grid \
        --run-tag "$RUN_TAG_BASELINE" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --no-mlflow \
        > logs/margin_cifar10_resnet34.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # CIFAR-10 DenseNet-121
    echo "[2/4] CIFAR-10 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_densenet121.yml \
        --config-detection configs/postprocessors/margin/cifar10_densenet121_hyperparams_search.yml \
        --mode search_res --eval-grid \
        --run-tag "$RUN_TAG_BASELINE" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --no-mlflow \
        > logs/margin_cifar10_densenet121.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    # CIFAR-100 ResNet-34
    echo "[3/4] CIFAR-100 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_resnet34.yml \
        --config-detection configs/postprocessors/margin/cifar100_resnet34_hyperparams_search.yml \
        --mode search_res --eval-grid \
        --run-tag "$RUN_TAG_BASELINE" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --no-mlflow \
        > logs/margin_cifar100_resnet34.log 2>&1 &
    PID3=$!
    echo "    PID: $PID3"

    # CIFAR-100 DenseNet-121
    echo "[4/4] CIFAR-100 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_densenet121.yml \
        --config-detection configs/postprocessors/margin/cifar100_densenet121_hyperparams_search.yml \
        --mode search_res --eval-grid \
        --run-tag "$RUN_TAG_BASELINE" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --no-mlflow \
        > logs/margin_cifar100_densenet121.log 2>&1 &
    PID4=$!
    echo "    PID: $PID4"

    echo ""
    echo "All Step 1 experiments started!"
    echo "PIDs: $PID1, $PID2, $PID3, $PID4"
    echo ""
    echo "Monitor with:"
    echo "  tail -f logs/margin_cifar10_resnet34.log"
    echo "  tail -f logs/margin_cifar10_densenet121.log"
    echo "  tail -f logs/margin_cifar100_resnet34.log"
    echo "  tail -f logs/margin_cifar100_densenet121.log"
}

run_step2() {
    echo "=== Step 2: Running Uniform Mass Binning on Margin Score ==="
    echo "Results: results/<dataset>/<model>/partition/runs/${RUN_TAG_BINNING}/"
    echo "Starting 4 experiments in parallel..."

    # CIFAR-10 ResNet-34
    echo "[1/4] CIFAR-10 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_resnet34.yml \
        --config-detection configs/postprocessors/partition/cifar10_resnet34_margin-unif-mass.yml \
        --mode search_res \
        --run-tag "$RUN_TAG_BINNING" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/margin_unif_mass_cifar10_resnet34.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # CIFAR-10 DenseNet-121
    echo "[2/4] CIFAR-10 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_densenet121.yml \
        --config-detection configs/postprocessors/partition/cifar10_densenet121_margin-unif-mass.yml \
        --mode search_res \
        --run-tag "$RUN_TAG_BINNING" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/margin_unif_mass_cifar10_densenet121.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    # CIFAR-100 ResNet-34
    echo "[3/4] CIFAR-100 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_resnet34.yml \
        --config-detection configs/postprocessors/partition/cifar100_resnet34_margin-unif-mass.yml \
        --mode search_res \
        --run-tag "$RUN_TAG_BINNING" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/margin_unif_mass_cifar100_resnet34.log 2>&1 &
    PID3=$!
    echo "    PID: $PID3"

    # CIFAR-100 DenseNet-121
    echo "[4/4] CIFAR-100 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_densenet121.yml \
        --config-detection configs/postprocessors/partition/cifar100_densenet121_margin-unif-mass.yml \
        --mode search_res \
        --run-tag "$RUN_TAG_BINNING" \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/margin_unif_mass_cifar100_densenet121.log 2>&1 &
    PID4=$!
    echo "    PID: $PID4"

    echo ""
    echo "All Step 2 experiments started!"
    echo "PIDs: $PID1, $PID2, $PID3, $PID4"
    echo ""
    echo "Monitor with:"
    echo "  tail -f logs/margin_unif_mass_cifar10_resnet34.log"
    echo "  tail -f logs/margin_unif_mass_cifar10_densenet121.log"
    echo "  tail -f logs/margin_unif_mass_cifar100_resnet34.log"
    echo "  tail -f logs/margin_unif_mass_cifar100_densenet121.log"
}

check_status() {
    echo "=== Checking running error-estimation processes ==="
    ps aux | grep "[p]ython.*run_detection" || echo "No error-estimation processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/margin*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 3 lines) ---"
            tail -3 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-}" in
    step1)
        run_step1
        ;;
    step2)
        run_step2
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {step1|step2|status}"
        echo ""
        echo "  step1  - Run Margin baseline with hyperparameter grid search"
        echo "           Grid: 6 temps × 7 magnitudes = 42 combinations"
        echo "           Results: results/<dataset>/<model>/margin/runs/${RUN_TAG_BASELINE}/"
        echo ""
        echo "  step2  - Run Uniform Mass Binning on Margin score"
        echo "           Grid: 5 n_clusters × 3 alphas × 2 scores = 30 combinations"
        echo "           Results: results/<dataset>/<model>/partition/runs/${RUN_TAG_BINNING}/"
        echo ""
        echo "  status - Check running processes and log activity"
        exit 1
        ;;
esac
