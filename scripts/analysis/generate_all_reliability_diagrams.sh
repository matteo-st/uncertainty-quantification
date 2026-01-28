#!/bin/bash
#
# Generate combined reliability diagrams for ALL dataset/model combinations.
#
# Creates reliability diagrams comparing:
#   1. Conservative calibration (upper bounds)
#   2. Calibrated mean
#   3. Raw score
#
# For each combination of:
#   - Datasets: cifar10, cifar100, imagenet
#   - Models: resnet34, densenet121 (CIFAR) | timm_vit_base16, timm_vit_tiny16 (ImageNet)
#   - Scores: msp, doctor, margin
#
# Usage:
#   ./scripts/analysis/generate_all_reliability_diagrams.sh
#
# Run from error-estimation/ directory.

set -e

# Configuration
N_CLUSTERS=50
ALPHA=0.05
SEED_SPLIT=1

# Base directories
RESULTS_BASE="results/partition_binning"
LATENT_BASE="latent"
OUTPUT_BASE="results/analysis/reliability_diagram"

# Score-specific parameters
# Format: score_type:temperature:normalize_flag
# normalize_flag: 1 for --normalize, 0 for --no-normalize
SCORE_PARAMS=(
    "msp:1.0:1"
    "doctor:0.7:0"
    "margin:0.7:1"
)

# Dataset/model combinations
# Format: dataset:model:latent_model
COMBINATIONS=(
    "cifar10:resnet34:resnet34"
    "cifar10:densenet121:densenet121"
    "cifar100:resnet34:resnet34"
    "cifar100:densenet121:densenet121"
    "imagenet:timm_vit_base16:timm_vit_base16"
    "imagenet:timm_vit_tiny16:timm_vit_tiny16"
)

# Function to generate suffix for output directories
get_score_suffix() {
    local score=$1
    if [[ "$score" == "msp" ]]; then
        echo ""
    else
        echo "_${score}"
    fi
}

# Function to run a single reliability diagram generation
generate_single_diagram() {
    local dataset=$1
    local model=$2
    local latent_model=$3
    local score=$4
    local temperature=$5
    local normalize_flag=$6
    local mode=$7  # calibrated, calibrated-mean, raw-score

    # Construct paths
    local run_dir="${RESULTS_BASE}/${dataset}/${model}_ce/partition/runs/${score}-unif-mass-sim-grid-20260120/seed-split-${SEED_SPLIT}"
    local latent_dir="${LATENT_BASE}/${dataset}_${latent_model}_ce/transform-test_n-epochs-1"

    # Determine output suffix based on mode
    local score_suffix=$(get_score_suffix "$score")
    local output_suffix=""
    case "$mode" in
        "calibrated")
            output_suffix=""
            ;;
        "calibrated-mean")
            output_suffix="_mean"
            ;;
        "raw-score")
            output_suffix="_raw"
            ;;
    esac

    local output_dir="${OUTPUT_BASE}/${dataset}_${model}${score_suffix}${output_suffix}"

    # Build normalize argument
    local normalize_arg=""
    if [[ "$normalize_flag" == "0" ]]; then
        normalize_arg="--no-normalize"
    fi

    echo "  [${mode}] ${dataset}/${model}/${score}"
    echo "    run_dir: ${run_dir}"
    echo "    latent_dir: ${latent_dir}"
    echo "    output_dir: ${output_dir}"

    # Check if run directory exists
    if [[ ! -d "$run_dir" ]]; then
        echo "    WARNING: Run directory does not exist, skipping..."
        return 1
    fi

    # Check if latent directory exists
    if [[ ! -d "$latent_dir" ]]; then
        echo "    WARNING: Latent directory does not exist, skipping..."
        return 1
    fi

    # Run the reliability diagram script
    python scripts/analysis/upper_reliability_diagram.py \
        --run-dir "$run_dir" \
        --latent-dir "$latent_dir" \
        --n-clusters "$N_CLUSTERS" \
        --alpha "$ALPHA" \
        --mode "$mode" \
        --score-type "$score" \
        --temperature "$temperature" \
        $normalize_arg \
        --output-dir "$output_dir"

    return 0
}

# Function to generate combined diagram
generate_combined_diagram() {
    local dataset=$1
    local model=$2
    local score=$3

    local score_suffix=$(get_score_suffix "$score")

    # Input JSON paths
    local conservative_json="${OUTPUT_BASE}/${dataset}_${model}${score_suffix}/reliability_stats.json"
    local mean_json="${OUTPUT_BASE}/${dataset}_${model}${score_suffix}_mean/calibrated_mean_reliability_stats.json"
    local raw_json="${OUTPUT_BASE}/${dataset}_${model}${score_suffix}_raw/raw_score_reliability_stats.json"

    # Output directory
    local output_dir="${OUTPUT_BASE}/combined_${dataset}_${model}${score_suffix}"

    echo "  [combined] ${dataset}/${model}/${score}"
    echo "    output_dir: ${output_dir}"

    # Check if all input files exist
    if [[ ! -f "$conservative_json" ]]; then
        echo "    WARNING: Conservative JSON not found: $conservative_json"
        return 1
    fi
    if [[ ! -f "$mean_json" ]]; then
        echo "    WARNING: Mean JSON not found: $mean_json"
        return 1
    fi
    if [[ ! -f "$raw_json" ]]; then
        echo "    WARNING: Raw JSON not found: $raw_json"
        return 1
    fi

    # Run combined diagram script
    python scripts/analysis/combined_reliability_diagram.py \
        --conservative-json "$conservative_json" \
        --mean-json "$mean_json" \
        --raw-json "$raw_json" \
        --output-dir "$output_dir" \
        --score-type "$score"

    return 0
}

# Main execution
echo "=========================================="
echo "Generating Reliability Diagrams for All Combinations"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  N_CLUSTERS: $N_CLUSTERS"
echo "  ALPHA: $ALPHA"
echo "  SEED_SPLIT: $SEED_SPLIT"
echo ""

# Track progress
total_diagrams=0
successful_diagrams=0
failed_diagrams=0

# Loop over all combinations
for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r dataset model latent_model <<< "$combo"

    echo ""
    echo "=========================================="
    echo "Processing: ${dataset} / ${model}"
    echo "=========================================="

    for score_param in "${SCORE_PARAMS[@]}"; do
        IFS=':' read -r score temperature normalize_flag <<< "$score_param"

        echo ""
        echo "------------------------------------------"
        echo "Score: ${score} (temp=${temperature}, normalize=${normalize_flag})"
        echo "------------------------------------------"

        # Generate all three modes
        for mode in "calibrated" "calibrated-mean" "raw-score"; do
            ((total_diagrams++)) || true
            if generate_single_diagram "$dataset" "$model" "$latent_model" "$score" "$temperature" "$normalize_flag" "$mode"; then
                ((successful_diagrams++)) || true
            else
                ((failed_diagrams++)) || true
            fi
        done

        # Generate combined diagram
        ((total_diagrams++)) || true
        if generate_combined_diagram "$dataset" "$model" "$score"; then
            ((successful_diagrams++)) || true
        else
            ((failed_diagrams++)) || true
        fi
    done
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total diagrams attempted: $total_diagrams"
echo "Successful: $successful_diagrams"
echo "Failed: $failed_diagrams"
echo ""
echo "Output directory: ${OUTPUT_BASE}/"
echo ""

# List generated combined diagrams
echo "Combined diagrams generated:"
find "${OUTPUT_BASE}" -name "combined_*" -type d 2>/dev/null | sort | while read -r dir; do
    if [[ -f "${dir}/combined_reliability_diagram.pdf" ]]; then
        echo "  ✓ ${dir}/combined_reliability_diagram.pdf"
    fi
done

echo ""
echo "Done!"
