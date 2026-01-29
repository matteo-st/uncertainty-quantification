#!/bin/bash
#
# Run conservative calibration validation for all dataset/model combinations.
#
# Usage:
#   bash scripts/analysis/run_all_calibration_validation.sh [--n-repetitions N]
#
# Default: 200 repetitions (as specified in plan)
#

set -e

N_REPS=${1:-200}
SCRIPT="scripts/analysis/validate_conservative_calibration.py"
BASE_OUTPUT="results/calibration_validation"

echo "============================================================"
echo "Conservative Calibration Validation"
echo "Running with $N_REPS repetitions per configuration"
echo "============================================================"

# --------------------------------------------------------------------------
# Helper function to run validation
# --------------------------------------------------------------------------
run_validation() {
    local dataset=$1
    local model=$2
    local score_type=$3
    local temperature=$4
    local normalize=$5
    local n_res=$6
    local n_cal=$7
    local n_eval=$8
    local n_bins=$9

    local latent_dir="latent/${dataset}_${model}_ce/transform-test_n-epochs-1"
    local output_dir="${BASE_OUTPUT}/${dataset}/${model}_ce/${score_type}/val-${N_REPS}seeds"

    local normalize_flag=""
    if [ "$normalize" = "true" ]; then
        normalize_flag="--normalize"
    else
        normalize_flag="--no-normalize"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Dataset: $dataset | Model: $model | Score: $score_type"
    echo "Temperature: $temperature | Normalize: $normalize"
    echo "Output: $output_dir"
    echo "------------------------------------------------------------"

    python $SCRIPT \
        --latent-dir "$latent_dir" \
        --n-res $n_res \
        --n-cal $n_cal \
        --n-eval $n_eval \
        --n-bins $n_bins \
        --alpha 0.01 0.02 0.05 0.1 0.2 \
        --n-repetitions $N_REPS \
        --score-type $score_type \
        --temperature $temperature \
        $normalize_flag \
        --base-seed 1 \
        --m-min 20 \
        --output-dir "$output_dir"
}


# ==========================================================================
# CIFAR-10 (n_total=10000, n_res=1000, n_cal=4000, n_eval=5000)
# ==========================================================================

# ---- ResNet34 ----
# Hyperparameters from seed-split-1 experiments (adjust as needed)
run_validation "cifar10" "resnet34" "msp"    1.0 false 1000 4000 5000 50
run_validation "cifar10" "resnet34" "doctor" 1.0 true  1000 4000 5000 50
run_validation "cifar10" "resnet34" "margin" 1.0 false 1000 4000 5000 50

# ---- DenseNet121 ----
run_validation "cifar10" "densenet121" "msp"    1.0 false 1000 4000 5000 50
run_validation "cifar10" "densenet121" "doctor" 1.0 true  1000 4000 5000 50
run_validation "cifar10" "densenet121" "margin" 1.0 false 1000 4000 5000 50


# ==========================================================================
# CIFAR-100 (n_total=10000, n_res=1000, n_cal=4000, n_eval=5000)
# ==========================================================================

# ---- ResNet34 ----
run_validation "cifar100" "resnet34" "msp"    1.0 false 1000 4000 5000 50
run_validation "cifar100" "resnet34" "doctor" 1.0 true  1000 4000 5000 50
run_validation "cifar100" "resnet34" "margin" 1.0 false 1000 4000 5000 50

# ---- DenseNet121 ----
run_validation "cifar100" "densenet121" "msp"    1.0 false 1000 4000 5000 50
run_validation "cifar100" "densenet121" "doctor" 1.0 true  1000 4000 5000 50
run_validation "cifar100" "densenet121" "margin" 1.0 false 1000 4000 5000 50


# ==========================================================================
# ImageNet (n_total=50000, n_res=5000, n_cal=20000, n_eval=25000)
# Hyperparameters from existing seed-split-1 results:
#   MSP:    temperature=0.6, normalize=false
#   Doctor: temperature=0.7, normalize=true
#   Margin: temperature=0.7, normalize=false
# ==========================================================================

# ---- ViT Base16 ----
run_validation "imagenet" "timm_vit_base16" "msp"    0.6 false 5000 20000 25000 100
run_validation "imagenet" "timm_vit_base16" "doctor" 0.7 true  5000 20000 25000 100
run_validation "imagenet" "timm_vit_base16" "margin" 0.7 false 5000 20000 25000 100

# ---- ViT Tiny16 ----
run_validation "imagenet" "timm_vit_tiny16" "msp"    0.6 false 5000 20000 25000 100
run_validation "imagenet" "timm_vit_tiny16" "doctor" 0.7 true  5000 20000 25000 100
run_validation "imagenet" "timm_vit_tiny16" "margin" 0.7 false 5000 20000 25000 100


echo ""
echo "============================================================"
echo "All validation runs completed!"
echo "Results saved to: $BASE_OUTPUT"
echo "============================================================"
