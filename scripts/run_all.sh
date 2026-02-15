#!/usr/bin/env bash
# run_all.sh — Full reproducible pipeline
# Usage: bash scripts/run_all.sh
set -euo pipefail

echo "=== Methane Portfolio Pipeline ==="
echo ""
WEIGHT_METHOD="${WEIGHT_METHOD:-avg}"

echo "Step 1/7: Validation"
methane-portfolio validate

echo "Step 2/7: Shapley Decomposition"
methane-portfolio shapley --weight-method "${WEIGHT_METHOD}"

echo "Step 3/7: Robust Optimization"
methane-portfolio optimize --delta 0.10 --lam 0.5 --alpha 0.90

echo "Step 4/7: Uncertainty Propagation"
methane-portfolio uncertainty

echo "Step 5/7: Tables"
methane-portfolio tables

echo "Step 6/7: Figures"
methane-portfolio figures

echo "Step 7/7: Methods Appendix"
methane-portfolio report

echo ""
echo "=== Pipeline complete ==="
echo "Outputs: outputs/"
echo "Figures: outputs/figures/"
