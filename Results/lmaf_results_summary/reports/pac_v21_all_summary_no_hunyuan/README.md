# PAC v2.1 Result Summary

Generated at: `2026-07-08 10:08:15`

## Scope

This report summarizes the currently available PAC v2.1 queue results. Scores are separated into success-only accuracy and conservative accuracy, where API errors count as zero.

## Run Coverage

- Total planned/evaluated rows in current raw files: `1668`
- Successful API/evaluable rows: `1377`
- Coverage: `82.6%`

## Top Models By Conservative Accuracy

| model | n_total | n_eval | coverage | accuracy_success | accuracy_all_conservative |
| --- | --- | --- | --- | --- | --- |
| qwen35_122b_a10b | 231 | 231 | 1.0000 | 0.6061 | 0.6061 |
| qwen35_27b | 147 | 147 | 1.0000 | 0.5850 | 0.5850 |
| qwen35_9b | 219 | 219 | 1.0000 | 0.3425 | 0.3425 |
| qwen35_35b_a3b | 231 | 231 | 1.0000 | 0.3160 | 0.3160 |
| seed_oss_36b | 231 | 108 | 0.4675 | 0.4630 | 0.2165 |

## Key Interpretation

- PAC-A and PAC-B show clear degradation under position and high-similarity interference pressure.
- PAC-C is the cleanest current binding-capacity signal.
- PAC-D v2.1 is effective for exposing multihop field-binding failures, but the current sample size is small.
- Low-coverage models should be interpreted cautiously until failed API rows are topped up.

## Tables

- dataset_summary: `tables\dataset_summary.csv`
- error_examples: `tables\error_examples.csv`
- error_types: `tables\error_types.csv`
- pac_A_position_accuracy_pivot: `tables\pac_A_position_accuracy_pivot.csv`
- pac_B_interference_accuracy_pivot: `tables\pac_B_interference_accuracy_pivot.csv`
- pac_C_binding_accuracy_pivot: `tables\pac_C_binding_accuracy_pivot.csv`
- pac_D_v21_multihop_accuracy_pivot: `tables\pac_D_v21_multihop_accuracy_pivot.csv`
- raw_result_index: `tables\raw_result_index.csv`
- summary_by_condition_model: `tables\summary_by_condition_model.csv`
- summary_by_model_overall: `tables\summary_by_model_overall.csv`
- summary_by_subset_model: `tables\summary_by_subset_model.csv`
- summary_by_subset_overall: `tables\summary_by_subset_overall.csv`
- table_explanations: `tables\table_explanations.csv`

## Figures

- fig_decoy_capture: `figures\pac_v21_decoy_capture_heatmap.png`
- fig_error_profile: `figures\pac_v21_error_type_profile.png`
- fig_field_vs_exact: `figures\pac_v21_field_vs_exact_scatter.png`
- fig_model_ranking: `figures\pac_v21_model_ranking_conservative.png`
- fig_pac_a_position: `figures\pac_A_position_accuracy_lines.png`
- fig_pac_b_interference: `figures\pac_B_interference_accuracy_lines.png`
- fig_pac_c_binding: `figures\pac_C_binding_capacity_heatmap.png`
- fig_pac_d_multihop: `figures\pac_D_v21_multihop_heatmap.png`
- fig_subset_accuracy: `figures\pac_v21_subset_accuracy_heatmap.png`
- fig_subset_accuracy_all: `figures\pac_v21_subset_accuracy_all_heatmap.png`
- fig_subset_coverage: `figures\pac_v21_subset_coverage_heatmap.png`