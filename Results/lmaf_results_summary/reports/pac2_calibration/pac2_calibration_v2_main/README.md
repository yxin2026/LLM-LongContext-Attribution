# PAC-Test 2.0 Calibration Report

This report calibrates 32K high-similarity interference difficulty for PAC-B.

Recommended decoy_count: `128`

Reason: fallback: closest to useful mean accuracy while preserving model spread

## By Decoy Count

| decoy_count | models_observed | mean_accuracy | mean_score_all | min_accuracy | max_accuracy | min_eval_per_model | max_api_error_rate | strong_minus_weak | qwen35_9b | qwen35_35b_a3b | qwen35_122b_a10b | ideal_band_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 128 | 3 | 0.9333 | 0.9333 | 0.8 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 0.8 | 1.0 | 0 |
| 256 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 512 | 3 | 1.0 | 0.6 | 1.0 | 1.0 | 0 | 1.0 |  | 1.0 | 1.0 |  | 0 |
| 768 | 3 | 1.0 | 0.6 | 1.0 | 1.0 | 0 | 1.0 |  | 1.0 | 1.0 |  | 0 |

## By Model And Decoy Count

| model | decoy_count | n_total | n_eval | n_api_error | accuracy | score_all | decoy_capture_rate | omission_rate | near_miss_rate | api_error_rate | mean_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_122b_a10b | 0 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17.258 |
| qwen35_35b_a3b | 0 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.979 |
| qwen35_9b | 0 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.94 |
| qwen35_122b_a10b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.447 |
| qwen35_35b_a3b | 128 | 5 | 5 | 0 | 0.8 | 0.8 | 0.2 | 0.0 | 0.0 | 0.0 | 1.987 |
| qwen35_9b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.887 |
| qwen35_122b_a10b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.116 |
| qwen35_35b_a3b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.715 |
| qwen35_9b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.237 |
| qwen35_122b_a10b | 512 | 5 | 0 | 5 |  | 0.0 |  |  |  | 1.0 |  |
| qwen35_35b_a3b | 512 | 5 | 4 | 1 | 1.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.2 | 2.782 |
| qwen35_9b | 512 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.131 |
| qwen35_122b_a10b | 768 | 5 | 0 | 5 |  | 0.0 |  |  |  | 1.0 |  |
| qwen35_35b_a3b | 768 | 5 | 4 | 1 | 1.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.2 | 29.757 |
| qwen35_9b | 768 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.247 |
