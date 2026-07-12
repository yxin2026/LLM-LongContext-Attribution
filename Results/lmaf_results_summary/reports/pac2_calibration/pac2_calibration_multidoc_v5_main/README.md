# PAC-Test 2.0 Calibration Report

This report calibrates 32K high-similarity interference difficulty for PAC-B.

Recommended decoy_count: `64`

Reason: fallback: closest to useful mean accuracy while preserving model spread

## By Decoy Count

| decoy_count | models_observed | mean_accuracy | mean_field_accuracy | mean_score_all | min_accuracy | max_accuracy | min_eval_per_model | max_api_error_rate | strong_minus_weak | qwen35_9b | qwen35_35b_a3b | qwen35_122b_a10b | ideal_band_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 3 | 0.2667 | 0.6222 | 0.2667 | 0.2 | 0.4 | 5 | 0.0 | 0.2 | 0.2 | 0.2 | 0.4 | 0 |
| 64 | 3 | 0.4 | 0.7111 | 0.4 | 0.0 | 0.8 | 5 | 0.0 | 0.8 | 0.0 | 0.4 | 0.8 | 0 |
| 128 | 3 | 0.3333 | 0.8667 | 0.3333 | 0.2 | 0.4 | 5 | 0.0 | -0.2 | 0.4 | 0.4 | 0.2 | 0 |
| 192 | 3 | 0.5333 | 0.9111 | 0.5333 | 0.4 | 0.8 | 5 | 0.0 | 0.4 | 0.4 | 0.4 | 0.8 | 0 |

## By Model And Decoy Count

| model | decoy_count | n_total | n_eval | n_api_error | accuracy | mean_field_accuracy | score_all | decoy_capture_rate | partial_rate | omission_rate | near_miss_rate | api_error_rate | mean_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_122b_a10b | 32 | 5 | 5 | 0 | 0.4 | 0.8667 | 0.4 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 | 3.832 |
| qwen35_35b_a3b | 32 | 5 | 5 | 0 | 0.2 | 0.7333 | 0.2 | 0.4 | 0.4 | 0.0 | 0.0 | 0.0 | 17.131 |
| qwen35_9b | 32 | 5 | 5 | 0 | 0.2 | 0.2667 | 0.2 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 | 7.255 |
| qwen35_122b_a10b | 64 | 5 | 5 | 0 | 0.8 | 1.0 | 0.8 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 4.181 |
| qwen35_35b_a3b | 64 | 5 | 5 | 0 | 0.4 | 0.8 | 0.4 | 0.2 | 0.4 | 0.0 | 0.0 | 0.0 | 3.469 |
| qwen35_9b | 64 | 5 | 5 | 0 | 0.0 | 0.3333 | 0.0 | 0.8 | 0.2 | 0.0 | 0.0 | 0.0 | 5.128 |
| qwen35_122b_a10b | 128 | 5 | 5 | 0 | 0.2 | 1.0 | 0.2 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 3.615 |
| qwen35_35b_a3b | 128 | 5 | 5 | 0 | 0.4 | 0.8667 | 0.4 | 0.2 | 0.4 | 0.0 | 0.0 | 0.0 | 3.436 |
| qwen35_9b | 128 | 5 | 5 | 0 | 0.4 | 0.7333 | 0.4 | 0.2 | 0.4 | 0.0 | 0.0 | 0.0 | 5.357 |
| qwen35_122b_a10b | 192 | 5 | 5 | 0 | 0.8 | 1.0 | 0.8 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 3.521 |
| qwen35_35b_a3b | 192 | 5 | 5 | 0 | 0.4 | 0.9333 | 0.4 | 0.0 | 0.6 | 0.0 | 0.0 | 0.0 | 5.628 |
| qwen35_9b | 192 | 5 | 5 | 0 | 0.4 | 0.8 | 0.4 | 0.2 | 0.4 | 0.0 | 0.0 | 0.0 | 5.061 |
