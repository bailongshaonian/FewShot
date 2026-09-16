# FewShot experiment visualization summary

Source: `results_data.txt`
Experiments parsed: **12**

## Main observations

1. **CLIP LoRA** achieves the highest reported test accuracy at **89.82%**.
2. It improves on CLIP zero-shot by **6.07 percentage points**.
3. The strongest non-CLIP result is **ResNet18 linear probe** at **87.60%**.
4. Validation and test accuracy are close for most methods, but these are single-seed results and do not quantify run-to-run uncertainty.

## Parameter training and training-free CLIP adaptation

| Method | Test accuracy |
| --- | ---: |
| CLIP LoRA | 89.82% |
| CLIP-Adapter | 89.64% |
| CLIP CoOp | 88.93% |
| CLIP linear probe | 88.65% |
| CLIP hard prompt | 86.98% |
| CLIP Tip-Adapter | 86.67% |

All four tested CLIP methods with parameter training outperform both Hard Prompt and the training-free Tip-Adapter in this run. Their accuracies span **88.65%–89.82%**.
Hard Prompt adds WordNet descriptions without labeled support images. Tip-Adapter uses a labeled image-feature cache (alpha=1, beta=5), not an external semantic knowledge base. It has hyperparameters but no gradient-based parameter training in this implementation.
These comparisons describe this dataset, configuration and seed; they do not establish universal superiority or statistical significance. LoRA and CLIP-Adapter differ by only 0.18 percentage points.

## Generated files

- `01_test_accuracy_ranking.png`: overall comparison and method families.
- `02_validation_test_gap.png`: validation/test gap for each experiment.
- `03_clip_parameter_efficiency.png`: accuracy against reported trainable parameters.
- `04_clip_adaptation_comparison.png`: CLIP parameter-training versus training-free results.
- `fewshot_visualizations.pdf`: all figures in one multi-page PDF.
- `experiment_summary.csv`: parsed, analysis-ready experiment table.
- `visualize_results.py`: reproducible visualization code.

## Interpretation note

The parameter-efficiency figure includes only methods whose logs report trainable-parameter counts. The data contains one seed (`42`), so confidence intervals and significance claims are not supported.
