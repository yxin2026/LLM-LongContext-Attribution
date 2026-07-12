# PAC-Test 2.0 Formal v5 Dataset

This folder contains the formal PAC-Test 2.0 datasets built after difficulty calibration.
The core difficulty anchor is `decoy_count=64`, selected from `pac2_calibration_multidoc_v5_main`.

## Subsets

| subset | samples | model scope | purpose |
| --- | ---: | --- | --- |
| PAC-A_position | 75 | all_9 | Position effect under calibrated high-similarity interference. |
| PAC-B_interference | 72 | all_9 | Interference threshold curve around the calibrated critical point. |
| PAC-C_binding_capacity | 72 | representative_6 | Entity/property binding capacity with K entities and Q queried aliases. |
| PAC-D_multihop_false_chain | 72 | representative_6 | Multihop chain tracking under false-chain interference. |

Total unique samples: `291`
Planned API calls: `2187`

## Files

- `data/PAC-A_position/samples.jsonl`
- `data/PAC-B_interference/samples.jsonl`
- `data/PAC-C_binding_capacity/samples.jsonl`
- `data/PAC-D_multihop_false_chain/samples.jsonl`
- `data/all_samples.jsonl`
- `manifest.json`

PAC-A and PAC-B run all 9 models. PAC-C and PAC-D run the representative 6-model panel.
Use exact-match accuracy plus mean field accuracy, partial rate, and decoy capture rate.

## Run

One-hour quality pilot, recommended before the full run:

```bat
python PAC\run_pac2_pilot.py --run-id pac2_pilot_v5_quality --max-workers 4 --max-in-flight 4 --request-delay-sec 12 --timeout 360 --retry 2 --max-tokens 192
```

The pilot runs 58 unique samples across PAC-A/B/C/D with the three anchor models:

```text
qwen35_9b
qwen35_35b_a3b
qwen35_122b_a10b
```

Default pilot size is 174 API calls. To make it faster but noisier:

```bat
python PAC\run_pac2_pilot.py --run-id pac2_pilot_v5_quality_fast --samples-per-condition 1 --max-workers 4 --max-in-flight 4 --request-delay-sec 8 --timeout 360 --retry 2 --max-tokens 192
```

Summarize pilot results:

```bat
python PAC\run_pac2_pilot.py --run-id pac2_pilot_v5_quality --summarize-only
```

## PAC-D v2.1 Strong-Model Pilot

PAC-D v2.1 is a harder multihop false-chain benchmark for strong models. It upgrades PAC-D with a green-verifier gate:

- follow 4/5/6 handoff hops;
- each LINK must have a matching VERIFY line;
- the verifier must be `GATE=green`, `S=valid`, `REVIEW=approved`, and before cutoff;
- reject false chains with red/draft/missing/after-cutoff verifier, wrong epoch, wrong batch, secondary channel, pending review, or inactive alias;
- output `FINAL_V|LAST_VERIFY_SIG|TICKET_V`.

Default pilot size:

```text
3 hop settings x 2 false-chain settings x 2 samples = 12 samples
5 strong/representative models = 60 API calls
```

Run:

```bat
python PAC\run_pac_d_v21_pilot.py --run-id pac_d_v21_pilot --max-workers 3 --max-in-flight 3 --request-delay-sec 12 --timeout 420 --retry 2 --max-tokens 192
```

Safer for 122B rate limits:

```bat
python PAC\run_pac_d_v21_pilot.py --run-id pac_d_v21_pilot --max-workers 2 --max-in-flight 2 --request-delay-sec 20 --timeout 420 --retry 2 --max-tokens 192
```

Summarize:

```bat
python PAC\run_pac_d_v21_pilot.py --run-id pac_d_v21_pilot --summarize-only
```

Balanced run, recommended first:

```bat
python PAC\run_pac2_formal.py --run-id pac2_formal_v5_main --max-workers 3 --max-in-flight 3 --request-delay-sec 10 --timeout 360 --retry 2 --max-tokens 192
```

PAC v2.1 queue full run, recommended for the final run:

```bat
python PAC\run_pac_v21_queue.py --run-id pac_v21_full_queue --slots-per-key 1 --queue-max-attempts 4 --rate-limit-cooldown-sec 90 --transient-cooldown-sec 25 --timeout 420 --retry 1 --max-tokens 192
```

This runner uses an API-key release queue: every API key owns one active slot by default, pulls one task, releases itself after completion, then immediately pulls the next task. It does not run fixed batches.

Use multiple API keys with:

```bat
set SILICONFLOW_API_KEYS=key1,key2,key3
```

If you want a more aggressive run and each key has enough quota:

```bat
python PAC\run_pac_v21_queue.py --run-id pac_v21_full_queue --slots-per-key 2 --queue-max-attempts 4 --rate-limit-cooldown-sec 90 --transient-cooldown-sec 25 --timeout 420 --retry 1 --max-tokens 192
```

Safer run if the API starts returning connection or rate-limit errors:

```bat
python PAC\run_pac2_formal.py --run-id pac2_formal_v5_main --max-workers 2 --max-in-flight 2 --request-delay-sec 15 --timeout 360 --retry 2 --max-tokens 192
```

Resume uses the same command. Successful rows are skipped; retryable API errors are retried.

Summarize after or during a run:

```bat
python PAC\run_pac2_formal.py --run-id pac2_formal_v5_main --summarize-only
```
