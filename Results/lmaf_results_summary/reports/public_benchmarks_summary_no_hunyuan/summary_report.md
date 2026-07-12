# Public Benchmark Result Summary

## Scope

本报告整理 LongBench、普通 NIAH 和普通 RULER 的已完成实验结果，用作论文/报告中公开基准阶段的基础证据。分数默认只在成功 API 调用上计算；API 额度、限流或连接错误不计为模型能力错误，而是在 coverage/error_rate 中单独呈现。

## Included Result Roots

- longbench: `results\raw\budget_core\budget_core_main\longbench`
- niah: `results\raw\budget_core\budget_core_main\niah`
- ruler: `results\raw\longbench_ruler_batch\framework_v2\longbench_ruler_main\ruler`

## High-Level Findings

- longbench: 当前成功样本均分最高的模型为 `qwen35_122b_a10b`，score=0.384，coverage=85.3%。
- niah: 当前成功样本均分最高的模型为 `qwen35_9b`，score=1.000，coverage=100.0%。
- ruler: 当前成功样本均分最高的模型为 `qwen35_9b`，score=0.950，coverage=40.7%。

NIAH 普通检索任务的平均成功样本准确率约为 0.988，可用于说明基础检索能力较强，但不宜单独作为复杂记忆衰减证据。

RULER 当前普通/ fallback 配置平均准确率约为 0.899，主要用于有效上下文边界与基础 synthetic 任务筛查。

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
- table_explanations: `tables\table_explanations.csv`

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
| longbench | seed_oss_36b | 300 | 276 | 24 | 0.9200 | 0.0800 | 0.3686 | 0.3391 | 0.6778 | 0.0753 | 0.3636 | 27.8967 | 11536.2033 |
| niah | qwen35_9b | 300 | 300 | 0 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |  |  |  | 7.7664 | 30355.1900 |
| niah | qwen3_8b | 300 | 237 | 63 | 0.7900 | 0.2100 | 0.9451 | 0.7467 |  |  |  | 5.2110 | 30345.4767 |
| niah | qwen35_27b | 300 | 240 | 60 | 0.8000 | 0.2000 | 1.0000 | 0.8000 |  |  |  | 10.8188 | 30347.2200 |
| niah | qwen35_35b_a3b | 300 | 286 | 14 | 0.9533 | 0.0467 | 0.9860 | 0.9400 |  |  |  | 5.9489 | 30352.9000 |
| niah | qwen35_122b_a10b | 300 | 153 | 147 | 0.5100 | 0.4900 | 1.0000 | 0.5100 |  |  |  | 6.6919 | 30333.9467 |
| niah | qwen3_14b_no_thinking | 300 | 138 | 162 | 0.4600 | 0.5400 | 0.9855 | 0.4533 |  |  |  | 3.7768 | 30330.1333 |
| niah | qwen3_14b_thinking | 300 | 131 | 169 | 0.4367 | 0.5633 | 0.9924 | 0.4333 |  |  |  | 4.8370 | 30327.3667 |
| niah | seed_oss_36b | 300 | 114 | 186 | 0.3800 | 0.6200 | 1.0000 | 0.3800 |  |  |  | 5.9492 | 30325.5367 |
| ruler | qwen35_9b | 600 | 244 | 356 | 0.4067 | 0.5933 | 0.9496 | 0.3862 | 0.9496 |  | 0.8689 | 5.9960 | 17761.7117 |
| ruler | qwen3_8b | 600 | 394 | 206 | 0.6567 | 0.3433 | 0.9033 | 0.5932 | 0.9033 |  | 0.7437 | 2.3486 | 17769.7783 |
| ruler | qwen35_27b | 600 | 332 | 268 | 0.5533 | 0.4467 | 0.8163 | 0.4517 | 0.8313 |  | 0.5512 | 13.8043 | 17769.1800 |
| ruler | qwen35_35b_a3b | 600 | 56 | 544 | 0.0933 | 0.9067 | 0.8832 | 0.0824 | 0.8832 |  | 0.6964 | 1.5779 | 17750.9950 |
| ruler | qwen35_122b_a10b | 600 | 44 | 556 | 0.0733 | 0.9267 | 0.9038 | 0.0663 | 0.9038 |  | 0.7500 | 0.3292 | 17750.2733 |
| ruler | qwen3_14b_no_thinking | 600 | 34 | 566 | 0.0567 | 0.9433 | 0.9208 | 0.0522 | 0.9208 |  | 0.7941 | 0.3105 | 17749.5050 |
| ruler | qwen3_14b_thinking | 600 | 40 | 560 | 0.0667 | 0.9333 | 0.8942 | 0.0596 | 0.8942 |  | 0.7250 | 0.6545 | 17749.6217 |
| ruler | seed_oss_36b | 600 | 30 | 570 | 0.0500 | 0.9500 | 0.8846 | 0.0442 | 0.8846 |  | 0.7000 | 0.9159 | 17749.1983 |

## Benchmark Summary

| benchmark | n_total | n_eval | n_error | coverage | error_rate | score_mean | score_all | f1_mean | rouge_l_mean | exact_match_mean | latency_mean | prompt_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longbench | 2400 | 2225 | 175 | 0.9271 | 0.0729 | 0.3242 | 0.3005 | 0.6091 | 0.0790 | 0.2780 | 14.3960 | 11506.7004 |
| niah | 2400 | 1599 | 801 | 0.6663 | 0.3337 | 0.9875 | 0.6579 |  |  |  | 6.3750 | 30339.7212 |
| ruler | 4800 | 1174 | 3626 | 0.2446 | 0.7554 | 0.8871 | 0.2170 | 0.8916 |  | 0.7129 | 3.2421 | 17756.2829 |
