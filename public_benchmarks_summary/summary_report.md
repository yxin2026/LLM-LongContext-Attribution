# Public Benchmark Result Summary

## Scope

本报告整理 LongBench、普通 NIAH 和普通 RULER 的已完成实验结果，用作论文/报告中公开基准阶段的基础证据。分数默认只在成功 API 调用上计算；API 额度、限流或连接错误不计为模型能力错误，而是在 coverage/error_rate 中单独呈现。

## Included Result Roots

- longbench: `results\raw\longbench_ruler_batch\framework_v2\longbench_ruler_main\longbench`
- niah: `results\raw\niah_batch\framework_v2_without_fast16k\framework_v2_extra`
- ruler: `results\raw\longbench_ruler_batch\framework_v2\longbench_ruler_main\ruler`

## High-Level Findings

- longbench: 当前成功样本均分最高的模型为 `seed_oss_36b`，score=0.392，coverage=91.1%。
- niah: 当前成功样本均分最高的模型为 `qwen35_9b`，score=1.000，coverage=73.5%。
- ruler: 当前成功样本均分最高的模型为 `qwen3_14b_thinking`，score=0.804，coverage=5.2%。

NIAH 普通检索任务的平均成功样本准确率约为 0.975，可用于说明基础检索能力较强，但不宜单独作为复杂记忆衰减证据。

RULER 当前普通/ fallback 配置平均准确率约为 0.602，主要用于有效上下文边界与基础 synthetic 任务筛查。

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
| longbench | qwen35_9b | 1750 | 1749 | 1 | 0.9994 | 0.0006 | 0.3588 | 0.3586 | 0.6705 | 0.0975 | 0.3185 | 3.4914 | 12157.0657 |
| longbench | qwen3_8b | 1750 | 1622 | 128 | 0.9269 | 0.0731 | 0.2327 | 0.2157 | 0.4158 | 0.1480 | 0.1010 | 5.2065 | 11990.5674 |
| longbench | qwen35_27b | 1750 | 1634 | 116 | 0.9337 | 0.0663 | 0.3848 | 0.3593 | 0.6963 | 0.1136 | 0.3576 | 7.8012 | 12074.4909 |
| longbench | qwen35_35b_a3b | 1750 | 1683 | 67 | 0.9617 | 0.0383 | 0.3692 | 0.3551 | 0.6711 | 0.1021 | 0.3444 | 28.9876 | 12127.6949 |
| longbench | qwen35_122b_a10b | 1750 | 1324 | 426 | 0.7566 | 0.2434 | 0.3878 | 0.2934 | 0.7037 | 0.0912 | 0.3626 | 4.2808 | 11977.4246 |
| longbench | qwen3_14b_no_thinking | 1750 | 1570 | 180 | 0.8971 | 0.1029 | 0.3388 | 0.3040 | 0.6238 | 0.1575 | 0.2467 | 3.5698 | 11971.2177 |
| longbench | qwen3_14b_thinking | 1750 | 774 | 976 | 0.4423 | 0.5577 | 0.3602 | 0.1593 | 0.7819 | 0.0660 | 0.4340 | 8.2677 | 11874.3937 |
| longbench | hunyuan_a13b | 1750 | 1398 | 352 | 0.7989 | 0.2011 | 0.2790 | 0.2229 | 0.5622 | 0.0928 | 0.2252 | 1.7756 | 11940.3189 |
| longbench | seed_oss_36b | 1750 | 1594 | 156 | 0.9109 | 0.0891 | 0.3922 | 0.3572 | 0.6825 | 0.1099 | 0.3754 | 24.5468 | 12093.6097 |
| niah | qwen35_9b | 750 | 551 | 199 | 0.7347 | 0.2653 | 1.0000 | 0.7347 |  |  |  | 4.5598 | 30343.4760 |
| niah | qwen3_8b | 750 | 391 | 359 | 0.5213 | 0.4787 | 0.9719 | 0.5067 |  |  |  | 3.1089 | 30332.8413 |
| niah | qwen35_27b | 750 | 378 | 372 | 0.5040 | 0.4960 | 1.0000 | 0.5040 |  |  |  | 6.8601 | 30332.8453 |
| niah | qwen35_35b_a3b | 750 | 333 | 417 | 0.4440 | 0.5560 | 1.0000 | 0.4440 |  |  |  | 2.2592 | 30329.9280 |
| niah | qwen35_122b_a10b | 750 | 40 | 710 | 0.0533 | 0.9467 | 1.0000 | 0.0533 |  |  |  | 0.3883 | 30311.8907 |
| niah | qwen3_14b_no_thinking | 750 | 40 | 710 | 0.0533 | 0.9467 | 0.9750 | 0.0520 |  |  |  | 0.4272 | 30311.8427 |
| niah | qwen3_14b_thinking | 750 | 39 | 711 | 0.0520 | 0.9480 | 0.9744 | 0.0507 |  |  |  | 0.5794 | 30311.7107 |
| niah | hunyuan_a13b | 750 | 36 | 714 | 0.0480 | 0.9520 | 0.1111 | 0.0053 |  |  |  | 0.4440 | 30311.0867 |
| niah | seed_oss_36b | 750 | 41 | 709 | 0.0547 | 0.9453 | 0.9512 | 0.0520 |  |  |  | 0.8124 | 30311.6627 |
| ruler | qwen35_9b | 900 | 299 | 601 | 0.3322 | 0.6678 | 0.7749 | 0.2574 | 0.7771 |  | 0.8689 | 4.5035 | 17759.0856 |
| ruler | qwen3_8b | 900 | 584 | 316 | 0.6489 | 0.3511 | 0.6128 | 0.3977 | 0.6151 |  | 0.7437 | 2.2940 | 17768.9633 |
| ruler | qwen35_27b | 900 | 475 | 425 | 0.5278 | 0.4722 | 0.5832 | 0.3078 | 0.5931 |  | 0.5512 | 11.5526 | 17766.9978 |
| ruler | qwen35_35b_a3b | 900 | 85 | 815 | 0.0944 | 0.9056 | 0.5819 | 0.0550 | 0.5819 |  | 0.6964 | 1.2762 | 17751.0744 |
| ruler | qwen35_122b_a10b | 900 | 66 | 834 | 0.0733 | 0.9267 | 0.6026 | 0.0442 | 0.6026 |  | 0.7500 | 0.3232 | 17750.3600 |
| ruler | qwen3_14b_no_thinking | 900 | 54 | 846 | 0.0600 | 0.9400 | 0.5983 | 0.0359 | 0.6106 |  | 0.7941 | 0.3236 | 17749.7811 |
| ruler | qwen3_14b_thinking | 900 | 47 | 853 | 0.0522 | 0.9478 | 0.8036 | 0.0420 | 0.8036 |  | 0.7250 | 1.2247 | 17749.3944 |
| ruler | hunyuan_a13b | 900 | 33 | 867 | 0.0367 | 0.9633 | 0.3173 | 0.0116 | 0.4501 |  | 0.3636 | 0.2308 | 17748.7278 |
| ruler | seed_oss_36b | 900 | 35 | 865 | 0.0389 | 0.9611 | 0.7868 | 0.0306 | 0.7868 |  | 0.7000 | 2.3951 | 17749.0589 |

## Benchmark Summary

| benchmark | n_total | n_eval | n_error | coverage | error_rate | score_mean | score_all | f1_mean | rouge_l_mean | exact_match_mean | latency_mean | prompt_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longbench | 15750 | 13348 | 2402 | 0.8475 | 0.1525 | 0.3442 | 0.2917 | 0.6410 | 0.1117 | 0.2994 | 9.7697 | 12022.9759 |
| niah | 6750 | 1849 | 4901 | 0.2739 | 0.7261 | 0.9746 | 0.2670 |  |  |  | 2.1599 | 30321.9204 |
| ruler | 8100 | 1678 | 6422 | 0.2072 | 0.7928 | 0.6340 | 0.1313 | 0.6423 |  | 0.7065 | 2.6804 | 17754.8270 |
