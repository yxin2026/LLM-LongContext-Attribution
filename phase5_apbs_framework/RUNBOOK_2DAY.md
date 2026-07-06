# Two-Day Runbook

## Day 1 Morning: Implementation Smoke

Run these first on the GPU host:

```powershell
pip install -r requirements.txt
python scripts/check_environment.py --model Qwen/Qwen3.5-9B
python scripts/smoke_patch.py
python scripts/make_niah_dataset.py --lengths 4096 --positions 50 --samples-per-cell 2 --output data/smoke_4k.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/smoke_4k.jsonl --method baseline --output results/raw/smoke_baseline.jsonl --limit 2
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/smoke_4k.jsonl --method ntk --output results/raw/smoke_ntk.jsonl --target-length 4096 --limit 2
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/smoke_4k.jsonl --method apbs --output results/raw/smoke_apbs.jsonl --target-length 4096 --gamma 0.3 --limit 2
```

Pass condition:

- all three modes complete without shape/device errors
- output JSONL contains `prediction` and `correct`
- APBS log says `patched_rotary_modules` is greater than zero

## Day 1 Afternoon: Main 16K Proof Run

```powershell
python scripts/make_niah_dataset.py --lengths 16384 --positions 10,50,90 --samples-per-cell 50 --output data/niah_16k_main.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_16k_main.jsonl --method baseline --output results/raw/16k_baseline.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_16k_main.jsonl --method ntk --output results/raw/16k_ntk.jsonl --target-length 16384
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_16k_main.jsonl --method apbs --output results/raw/16k_apbs_g03.jsonl --target-length 16384 --gamma 0.3
python scripts/analyze_apbs_results.py --inputs "results/raw/*.jsonl" --output-dir results/analysis
```

Decision point:

- If APBS@50 is clearly above baseline and NTK, continue to Day 2.
- If APBS hurts head/tail badly, reduce gamma to `0.1`.
- If APBS has no middle lift, try `gamma=0.5`, but report the outcome honestly as negative/weak.

## Day 2 Morning: Gamma Sensitivity

```powershell
python scripts/make_niah_dataset.py --lengths 16384 --positions 50 --samples-per-cell 50 --output data/niah_16k_gamma.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g01_mid.jsonl --target-length 16384 --gamma 0.1
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g05_mid.jsonl --target-length 16384 --gamma 0.5
```

## Day 2 Afternoon: Optional 32K Sanity Check

```powershell
python scripts/make_niah_dataset.py --lengths 32768 --positions 10,50,90 --samples-per-cell 15 --output data/niah_32k_small.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_32k_small.jsonl --method baseline --output results/raw/32k_baseline.jsonl
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_32k_small.jsonl --method ntk --output results/raw/32k_ntk.jsonl --target-length 32768
python scripts/run_apbs_niah.py --model Qwen/Qwen3.5-9B --dataset data/niah_32k_small.jsonl --method apbs --output results/raw/32k_apbs_g03.jsonl --target-length 32768 --gamma 0.3
python scripts/analyze_apbs_results.py --inputs "results/raw/*.jsonl" --output-dir results/analysis
```

## Final Wording

Use this conclusion only if the metrics pass:

> APBS provides controlled intervention evidence on Qwen 9B at 16K: position-aware RoPE base compensation improves middle-position NIAH retrieval beyond both baseline RoPE and global NTK, while reducing the U-shaped position gap.

Avoid claiming cross-model or cross-architecture validity from this two-day experiment.
