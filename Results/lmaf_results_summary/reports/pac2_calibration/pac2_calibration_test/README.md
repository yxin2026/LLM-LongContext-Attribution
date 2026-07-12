# PAC-Test 2.0 Calibration Report

This report calibrates 32K high-similarity interference difficulty for PAC-B.

Recommended decoy_count: `0`

Reason: fallback: closest to useful mean accuracy while preserving model spread

## By Decoy Count

| decoy_count | models_observed | mean_accuracy | min_accuracy | max_accuracy | strong_minus_weak | qwen35_9b | qwen35_35b_a3b | qwen35_122b_a10b | ideal_band_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1.0 | 1.0 | 1.0 |  | 1.0 |  |  | 0 |

## By Model And Decoy Count

| model | decoy_count | n_total | n_eval | n_api_error | accuracy | score_all | decoy_capture_rate | omission_rate | near_miss_rate | api_error_rate | mean_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_9b | 0 | 3 | 3 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9.831 |
