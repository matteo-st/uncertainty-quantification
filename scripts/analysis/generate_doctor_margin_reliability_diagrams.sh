#!/bin/bash
# Generate reliability diagrams for Doctor and Margin scores
# Run this script from the error-estimation directory on the server

set -e

# Base directories
RESULTS_BASE="results/partition_binning/imagenet/timm_vit_base16_ce/partition/runs"
LATENT_DIR="latent/imagenet_timm_vit_base16_ce/transform-test_n-epochs-1"
OUTPUT_BASE="results/analysis/reliability_diagram"

echo "=== Generating Doctor Score Reliability Diagrams ==="

# Doctor - Conservative calibration
echo "Doctor: Conservative calibration..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/doctor-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --alpha 0.05 \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_doctor"

# Doctor - Calibrated mean
echo "Doctor: Calibrated mean..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/doctor-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --mode calibrated-mean \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_doctor_mean"

# Doctor - Raw score
echo "Doctor: Raw score..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/doctor-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --mode raw-score \
    --score-type doctor \
    --temperature 0.7 \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_doctor_raw"

echo "=== Generating Margin Score Reliability Diagrams ==="

# Margin - Conservative calibration
echo "Margin: Conservative calibration..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/margin-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --alpha 0.05 \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_margin"

# Margin - Calibrated mean
echo "Margin: Calibrated mean..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/margin-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --mode calibrated-mean \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_margin_mean"

# Margin - Raw score
echo "Margin: Raw score..."
python scripts/analysis/upper_reliability_diagram.py \
    --run-dir "${RESULTS_BASE}/margin-unif-mass-sim-grid-20260120/seed-split-1" \
    --latent-dir "${LATENT_DIR}" \
    --n-clusters 50 \
    --mode raw-score \
    --score-type margin \
    --temperature 0.7 \
    --output-dir "${OUTPUT_BASE}/imagenet_vit_base16_margin_raw"

echo "=== Generating Combined Plots ==="

# Doctor combined
echo "Doctor: Combined plot..."
python scripts/analysis/combined_reliability_diagram.py \
    --conservative-json "${OUTPUT_BASE}/imagenet_vit_base16_doctor/reliability_stats.json" \
    --mean-json "${OUTPUT_BASE}/imagenet_vit_base16_doctor_mean/calibrated_mean_reliability_stats.json" \
    --raw-json "${OUTPUT_BASE}/imagenet_vit_base16_doctor_raw/raw_score_reliability_stats.json" \
    --output-dir "${OUTPUT_BASE}/combined_doctor" \
    --score-type doctor

# Margin combined
echo "Margin: Combined plot..."
python scripts/analysis/combined_reliability_diagram.py \
    --conservative-json "${OUTPUT_BASE}/imagenet_vit_base16_margin/reliability_stats.json" \
    --mean-json "${OUTPUT_BASE}/imagenet_vit_base16_margin_mean/calibrated_mean_reliability_stats.json" \
    --raw-json "${OUTPUT_BASE}/imagenet_vit_base16_margin_raw/raw_score_reliability_stats.json" \
    --output-dir "${OUTPUT_BASE}/combined_margin" \
    --score-type margin

echo "=== Done ==="
echo ""
echo "Generated outputs:"
echo "  Doctor: ${OUTPUT_BASE}/combined_doctor/"
echo "  Margin: ${OUTPUT_BASE}/combined_margin/"
