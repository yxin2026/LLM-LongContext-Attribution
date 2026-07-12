# PAC-Test 2.0 Calibration Report

This report calibrates 32K high-similarity interference difficulty for PAC-B.

Recommended decoy_count: `None`

Reason: ceiling effect: all stable decoy counts are near 100% with almost no model spread

## By Decoy Count

| decoy_count | models_observed | mean_accuracy | mean_score_all | min_accuracy | max_accuracy | min_eval_per_model | max_api_error_rate | strong_minus_weak | qwen35_9b | qwen35_35b_a3b | qwen35_122b_a10b | ideal_band_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 256 | 3 | 1.0 | 0.8667 | 1.0 | 1.0 | 3 | 0.4 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 384 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |
| 448 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 |

## By Model And Decoy Count

| model | decoy_count | n_total | n_eval | n_api_error | accuracy | score_all | decoy_capture_rate | omission_rate | near_miss_rate | api_error_rate | mean_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_122b_a10b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.065 |
| qwen35_35b_a3b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.723 |
| qwen35_9b | 128 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.848 |
| qwen35_122b_a10b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.595 |
| qwen35_35b_a3b | 256 | 5 | 3 | 2 | 1.0 | 0.6 | 0.0 | 0.0 | 0.0 | 0.4 | 2.621 |
| qwen35_9b | 256 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.622 |
| qwen35_122b_a10b | 384 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.975 |
| qwen35_35b_a3b | 384 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.893 |
| qwen35_9b | 384 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.742 |
| qwen35_122b_a10b | 448 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.076 |
| qwen35_35b_a3b | 448 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.178 |
| qwen35_9b | 448 | 5 | 5 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.346 |
