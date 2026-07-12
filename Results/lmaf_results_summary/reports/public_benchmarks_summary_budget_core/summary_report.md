# Public Benchmark Result Summary

## Scope

本报告整理 LongBench、普通 NIAH 和普通 RULER 的已完成实验结果，用作论文/报告中公开基准阶段的基础证据。分数默认只在成功 API 调用上计算；API 额度、限流或连接错误不计为模型能力错误，而是在 coverage/error_rate 中单独呈现。

## Included Result Roots

- longbench: `results\raw\budget_core\budget_core_main\longbench`
- niah: `results\raw\budget_core\budget_core_main\niah`
- ruler: `results\raw\budget_core\budget_core_main\ruler`

## High-Level Findings

- longbench: 当前成功样本均分最高的模型为 `qwen35_122b_a10b`，score=0.384，coverage=85.3%。
- niah: 当前成功样本均分最高的模型为 `qwen35_9b`，score=1.000，coverage=100.0%。
- ruler: 当前成功样本均分最高的模型为 `qwen3_14b_no_thinking`，score=0.887，coverage=56.7%。

NIAH 普通检索任务的平均成功样本准确率约为 0.988，可用于说明基础检索能力较强，但不宜单独作为复杂记忆衰减证据。

RULER 当前普通/ fallback 配置平均准确率约为 0.839，主要用于有效上下文边界与基础 synthetic 任务筛查。

整体解释口径：LongBench 作为通用长上下文能力基线，NIAH 验证基础检索能力，RULER 检查普通 synthetic 任务和有效上下文边界。若这些公开基准出现高分或区分度不足，应解释为公开基准天花板效应，而不是实验失败；这正是后续 PAC-Test 2.0 转向高相似干扰、实体绑定和多跳假链的动机。

## Tables

- benchmark_summary: `tables\benchmark_summary.csv`
- code_inventory: `tables\code_inventory.csv`
- coverage_by_benchmark_model: `tables\coverage_by_benchmark_model.csv`
- longbench_by_category_model: `tables\longbench_by_category_model.csv`
- longbench_by_model: `tables\longbench_by_model.csv`
- longbench_by_task_model: `tables\longbench_by_task_model.csv`
- niah_by_condition_model: `tables\niah_by_condition_model.csv`
- niah_by_subtask_model: `tables\niah_by_subtask_model.csv`
- niah_middle_drop: `tables\niah_middle_drop.csv`
- niah_single_position: `tables\niah_single_position.csv`
- raw_public_rows_dedup: `tables\raw_public_rows_dedup.csv`
- ruler_by_task_length_model: `tables\ruler_by_task_length_model.csv`
- ruler_by_task_model: `tables\ruler_by_task_model.csv`
- ruler_effective_context: `tables\ruler_effective_context.csv`
- source_runs: `tables\source_runs.csv`

## Figures

- fig_coverage_heatmap: `figures\coverage_heatmap.png`
- fig_longbench_category_heatmap: `figures\longbench_category_heatmap.png`
- fig_longbench_model_ranking: `figures\longbench_model_ranking.png`
- fig_longbench_task_heatmap: `figures\longbench_task_heatmap.png`
- fig_niah_single_position_heatmap: `figures\niah_single_position_32768.png`
- fig_niah_single_position_lines: `figures\niah_single_position_lines_32768.png`
- fig_niah_subtask_heatmap: `figures\niah_variant_heatmap.png`
- fig_ruler_context_length_lines: `figures\ruler_context_length_lines.png`
- fig_ruler_task_heatmap: `figures\ruler_task_heatmap.png`
- fig_ruler_task_length_heatmap: `figures\ruler_task_length_heatmap.png`

## Coverage Snapshot

| benchmark | model | n_total | n_eval | n_error | coverage | error_rate | score_mean | score_all | f1_mean | rouge_l_mean | exact_match_mean | latency_mean | prompt_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longbench | qwen35_9b | 300 | 300 | 0 | 1.0000 | 0.0000 | 0.3133 | 0.3133 | 0.5920 | 0.0580 | 0.2600 | 3.4433 | 11594.6233 |
| longbench | qwen3_8b | 300 | 281 | 19 | 0.9367 | 0.0633 | 0.2205 | 0.2065 | 0.3995 | 0.1160 | 0.0939 | 4.8523 | 11458.0733 |
| longbench | qwen35_27b | 300 | 281 | 19 | 0.9367 | 0.0633 | 0.3293 | 0.3084 | 0.6235 | 0.0710 | 0.2818 | 4.5869 | 11506.8333 |
| longbench | qwen35_35b_a3b | 300 | 300 | 0 | 1.0000 | 0.0000 | 0.3440 | 0.3440 | 0.6179 | 0.0620 | 0.3150 | 35.2893 | 11594.6233 |
| longbench | qwen35_122b_a10b | 300 | 256 | 44 | 0.8533 | 0.1467 | 0.3838 | 0.3276 | 0.6942 | 0.0505 | 0.3916 | 16.9158 | 11462.5400 |
| longbench | qwen3_14b_no_thinking | 300 | 277 | 23 | 0.9233 | 0.0767 | 0.3205 | 0.2959 | 0.5959 | 0.1308 | 0.2147 | 3.1553 | 11455.9933 |
| longbench | qwen3_14b_thinking | 300 | 254 | 46 | 0.8467 | 0.1533 | 0.3180 | 0.2693 | 0.6628 | 0.0655 | 0.3182 | 19.0287 | 11444.7133 |
| longbench | hunyuan_a13b | 300 | 275 | 25 | 0.9167 | 0.0833 | 0.2479 | 0.2273 | 0.5207 | 0.0594 | 0.1899 | 2.2980 | 11440.0233 |
| longbench | seed_oss_36b | 300 | 276 | 24 | 0.9200 | 0.0800 | 0.3686 | 0.3391 | 0.6778 | 0.0753 | 0.3636 | 27.8967 | 11536.2033 |
| niah | qwen35_9b | 300 | 300 | 0 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |  |  |  | 7.7664 | 30355.1900 |
| niah | qwen3_8b | 300 | 237 | 63 | 0.7900 | 0.2100 | 0.9451 | 0.7467 |  |  |  | 5.2110 | 30345.4767 |
| niah | qwen35_27b | 300 | 240 | 60 | 0.8000 | 0.2000 | 1.0000 | 0.8000 |  |  |  | 10.8188 | 30347.2200 |
| niah | qwen35_35b_a3b | 300 | 286 | 14 | 0.9533 | 0.0467 | 0.9860 | 0.9400 |  |  |  | 5.9489 | 30352.9000 |
| niah | qwen35_122b_a10b | 300 | 153 | 147 | 0.5100 | 0.4900 | 1.0000 | 0.5100 |  |  |  | 6.6919 | 30333.9467 |
| niah | qwen3_14b_no_thinking | 300 | 138 | 162 | 0.4600 | 0.5400 | 0.9855 | 0.4533 |  |  |  | 3.7768 | 30330.1333 |
| niah | qwen3_14b_thinking | 300 | 131 | 169 | 0.4367 | 0.5633 | 0.9924 | 0.4333 |  |  |  | 4.8370 | 30327.3667 |
| niah | hunyuan_a13b | 300 | 165 | 135 | 0.5500 | 0.4500 | 0.9867 | 0.9867 |  |  |  | 6.3114 | 30329.7967 |
| niah | seed_oss_36b | 300 | 114 | 186 | 0.3800 | 0.6200 | 1.0000 | 0.3800 |  |  |  | 5.9492 | 30325.5367 |
| ruler | qwen35_9b | 180 | 180 | 0 | 1.0000 | 0.0000 | 0.8718 | 0.8718 | 0.8718 |  | 0.6667 | 5.0896 | 17785.1111 |
| ruler | qwen3_8b | 180 | 177 | 3 | 0.9833 | 0.0167 | 0.8715 | 0.8570 | 0.8715 |  | 0.6497 | 4.4380 | 17781.3778 |
| ruler | qwen35_27b | 180 | 180 | 0 | 1.0000 | 0.0000 | 0.8440 | 0.8440 | 0.8681 |  | 0.6389 | 17.4106 | 17785.1111 |
| ruler | qwen35_35b_a3b | 180 | 173 | 7 | 0.9611 | 0.0389 | 0.8776 | 0.8435 | 0.8776 |  | 0.6763 | 4.0403 | 17783.6444 |
| ruler | qwen35_122b_a10b | 180 | 115 | 65 | 0.6389 | 0.3611 | 0.8562 | 0.5470 | 0.8562 |  | 0.6261 | 5.6838 | 17771.8500 |
| ruler | qwen3_14b_no_thinking | 180 | 102 | 78 | 0.5667 | 0.4333 | 0.8869 | 0.5026 | 0.8869 |  | 0.7059 | 4.1796 | 17767.2389 |
| ruler | qwen3_14b_thinking | 180 | 115 | 65 | 0.6389 | 0.3611 | 0.8796 | 0.5620 | 0.8796 |  | 0.6870 | 7.5734 | 17766.9722 |
| ruler | hunyuan_a13b | 180 | 92 | 88 | 0.5111 | 0.4889 | 0.4720 | 0.2413 | 0.7361 |  | 0.2935 | 3.5593 | 17761.1278 |
| ruler | seed_oss_36b | 180 | 95 | 85 | 0.5278 | 0.4722 | 0.8826 | 0.4658 | 0.8826 |  | 0.6947 | 6.5374 | 17765.0833 |

## Benchmark Summary

| benchmark | n_total | n_eval | n_error | coverage | error_rate | score_mean | score_all | f1_mean | rouge_l_mean | exact_match_mean | latency_mean | prompt_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longbench | 2700 | 2500 | 200 | 0.9259 | 0.0741 | 0.3158 | 0.2924 | 0.6002 | 0.0769 | 0.2683 | 13.0518 | 11499.2919 |
| niah | 2700 | 1764 | 936 | 0.6533 | 0.3467 | 0.9874 | 0.6945 |  |  |  | 6.3679 | 30338.6185 |
| ruler | 1620 | 1229 | 391 | 0.7586 | 0.2414 | 0.8399 | 0.6372 | 0.8667 |  | 0.6371 | 6.5013 | 17774.1685 |
