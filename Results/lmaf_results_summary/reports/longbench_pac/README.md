# LongBench / PAC Result Report

Generated from deduplicated JSONL results. Duplicate rows prefer successful records and newer top-up outputs.

## Row Counts

- longbench: 2700 rows
- pac: 5400 rows

## Files

- fig_longbench_category_heatmap: figures\longbench_category_heatmap.png
- fig_longbench_model_ranking: figures\longbench_model_ranking.png
- fig_longbench_task_heatmap: figures\longbench_task_heatmap.png
- fig_pac_A_length_heatmap: figures\pac_A_length_heatmap.png
- fig_pac_A_position_heatmap: figures\pac_A_position_heatmap.png
- fig_pac_B_density_in_domain: figures\pac_B_density_in_domain.png
- fig_pac_B_density_out_domain: figures\pac_B_density_out_domain.png
- fig_pac_B_density_random_noise: figures\pac_B_density_random_noise.png
- fig_pac_C_condition_heatmap: figures\pac_C_condition_heatmap.png
- fig_pac_C_model_ranking: figures\pac_C_model_ranking.png
- fig_pac_D_distance_heatmap: figures\pac_D_distance_heatmap.png
- fig_pac_D_hops_heatmap: figures\pac_D_hops_heatmap.png
- fig_pac_subset_heatmap: figures\pac_subset_heatmap.png
- longbench_by_category_model: tables\longbench_by_category_model.csv
- longbench_by_model: tables\longbench_by_model.csv
- longbench_by_task_model: tables\longbench_by_task_model.csv
- pac_A_position_details: tables\pac_A_position_details.csv
- pac_B_interference_details: tables\pac_B_interference_details.csv
- pac_C_overlap_details: tables\pac_C_overlap_details.csv
- pac_D_multihop_details: tables\pac_D_multihop_details.csv
- pac_by_subset_model: tables\pac_by_subset_model.csv

## Model Coverage

| experiment   | model                 |   rows |
|:-------------|:----------------------|-------:|
| longbench    | hunyuan_a13b          |    300 |
| longbench    | qwen35_122b_a10b      |    300 |
| longbench    | qwen35_27b            |    300 |
| longbench    | qwen35_35b_a3b        |    300 |
| longbench    | qwen35_9b             |    300 |
| longbench    | qwen3_14b_no_thinking |    300 |
| longbench    | qwen3_14b_thinking    |    300 |
| longbench    | qwen3_8b              |    300 |
| longbench    | seed_oss_36b          |    300 |
| pac          | hunyuan_a13b          |    600 |
| pac          | qwen35_122b_a10b      |    600 |
| pac          | qwen35_27b            |    600 |
| pac          | qwen35_35b_a3b        |    600 |
| pac          | qwen35_9b             |    600 |
| pac          | qwen3_14b_no_thinking |    600 |
| pac          | qwen3_14b_thinking    |    600 |
| pac          | qwen3_8b              |    600 |
| pac          | seed_oss_36b          |    600 |