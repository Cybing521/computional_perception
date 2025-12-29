#!/bin/bash
# Copy generated images to Report/images directory

REPORT_DIR="../../../Report/images"

# Create directory if not exists
mkdir -p $REPORT_DIR

# Copy images
cp scenario_analysis/scenario_comparison.png $REPORT_DIR/
cp efficiency_analysis/efficiency_accuracy_scatter.png $REPORT_DIR/
cp efficiency_analysis/pareto_frontier.png $REPORT_DIR/
cp sensitivity_analysis/lambda_grad_sensitivity.png $REPORT_DIR/
cp pr_curves/pr_curves_by_class.png $REPORT_DIR/

echo 'Images copied to Report/images/'
