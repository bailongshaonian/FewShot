# FewShot experiment visualization summary

Source: `results_data.txt`
Experiments parsed: **12**

## Main observations

1. **CLIP LoRA** achieves the highest reported test accuracy at **89.82%**.
2. It improves on CLIP zero-shot by **6.07 percentage points**.
3. The strongest non-CLIP result is **ResNet18 linear probe** at **87.28%**.
4. Validation and test accuracy are close for most methods, but these are single-seed results and do not quantify run-to-run uncertainty.

## Generated files

- `01_test_accuracy_ranking.png`: overall comparison and method families.
- `02_validation_test_gap.png`: validation/test gap for each experiment.
- `03_clip_parameter_efficiency.png`: accuracy against reported trainable parameters.
- `fewshot_visualizations.pdf`: all figures in one multi-page PDF.
- `experiment_summary.csv`: parsed, analysis-ready experiment table.
- `visualize_results.py`: reproducible visualization code.

## Interpretation note

The parameter-efficiency figure includes only methods whose logs report trainable-parameter counts. The data contains one seed (`42`), so confidence intervals and significance claims are not supported.
