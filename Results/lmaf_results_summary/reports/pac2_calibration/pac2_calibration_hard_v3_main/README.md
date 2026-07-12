# PAC-Test 2.0 Calibration Report

This report calibrates 32K high-similarity interference difficulty for PAC-B.

Recommended decoy_count: `None`

Reason: ceiling effect: all stable decoy counts are near 100% with almost no model spread

## By Decoy Count

| decoy_count | models_observed | mean_accuracy | mean_score_all | min_accuracy | max_accuracy | min_eval_per_model | max_api_error_rate | strong_minus_weak | qwen35_9b | qwen35_35b_a3b | qwen35_122b_a10b | ideal_band_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1.0 | 0.6667 | 1.0 | 1.0 | 1 | 0.8 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 128 | 3 | 1.0 | 0.8 | 1.0 | 1.0 | 2 | 0.6 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 256 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 384 | 3 | 1.0 | 0.8 | 1.0 | 1.0 | 2 | 0.6 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 512 | 3 | 1.0 | 0.9333 | 1.0 | 1.0 | 4 | 0.2 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |

## By Model And Decoy Count

| model | decoy_count | n_total | n_eval | n_api_error | accuracy | score_all | decoy_capture_rate | omission_rate | near_miss_rate | api_error_rate | mean_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_122b_a10b | 0 | 5 | 1 | 4 | 1.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.8 | 5.366 |
| qwen35_35b_a3b | 0 | 5 | 4 | 1 | 1.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.2 | 2.853 |
| qwen35_9b | 0 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.447 |
| qwen35_122b_a10b | 128 | 5 | 2 | 3 | 1.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.6 | 6.869 |
| qwen35_35b_a3b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.338 |
| qwen35_9b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.035 |
| qwen35_122b_a10b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.637 |
| qwen35_35b_a3b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 13.887 |
| qwen35_9b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.571 |
| qwen35_122b_a10b | 384 | 5 | 2 | 3 | 1.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.6 | 5.954 |
| qwen35_35b_a3b | 384 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 23.802 |
| qwen35_9b | 384 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.722 |
| qwen35_122b_a10b | 512 | 5 | 4 | 1 | 1.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.2 | 4.85 |
| qwen35_35b_a3b | 512 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 13.873 |
| qwen35_9b | 512 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.328 |
