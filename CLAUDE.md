# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Chinese long-context benchmark suite ("PAC-Test") plus two NIAH variants, all targeting **local Ollama models** via the OpenAI-compatible API. The PAC-Test side is the primary deliverable; the NIAH scripts are companion diagnostics (a separate person owns the *synthetic* NIAH; this repo's owner owns PAC-Test).

## Common commands

All scripts assume `ollama serve` is running at `http://localhost:11434/v1`.

```powershell
# PAC-Test eval (supports checkpoint resume — re-running same model+subset skips done samples)
python evaluate.py --model qwen2.5:7b --subset all
python evaluate.py --model deepseek-r1:7b --subset A,B --samples 100   # quick smoke
python evaluate.py --list-models                                        # see what Ollama has

# Multi-model sweep (calls evaluate.py per model in MODELS list)
python run_all_models.py

# Cross-model comparison table + chart (reads results/*_summary.json)
python compare_models.py

# NIAH variants
python niah_test.py --model qwen2.5:7b              # synthetic needle (sci-fi sentence)
python niah_knowledge_test.py --model qwen2.5:7b    # real-fact needle from facts_library.json

# NIAH heatmaps in gkamradt visual style → niah_plots/
python plot_niah.py --models qwen2.5:7b deepseek-r1:7b

# (Re)generate the dataset (writes data/subset_{A,B,C,D}.jsonl + PAC-Test_complete.jsonl)
python build_dataset_v3.py       # token-strict, uses the real Qwen2.5 tokenizer
```

Single-sample debugging: `--samples 1` on `evaluate.py` runs exactly one item per subset.

There is no test suite, no linter config, and no `requirements.txt`. Dependencies in practice: `requests`, `tqdm`, `numpy`, `pandas`, `matplotlib`.

## Architecture: the four moving parts

**1. Knowledge base → dataset → eval → plots** is a strict one-way pipeline:

```
core_knowledge/{facts_library,entity_pairs,fact_chains}.json
        ↓  build_dataset_v3.py
data/subset_{A,B,C,D}.jsonl
        ↓  evaluate.py  (per model, per subset, with checkpoints)
results/{model}_{ts}_{details.jsonl,summary.json,chart.png}
        ↓  compare_models.py / plot_niah.py
results/comparison_chart.png  |  niah_plots/*.png
```

The `core_knowledge/` JSONs are hand-curated and seed everything downstream. **If you regenerate the dataset, do not modify `core_knowledge/` — only `build_dataset_v3.py`.**

**2. The four PAC-Test subsets each isolate one failure mode** (see `data/README.md` for full spec):

| Subset | Variable swept                    | Grid (v3, token-strict)                                  | Samples            |
|--------|-----------------------------------|----------------------------------------------------------|--------------------|
| A      | needle **position** in context    | 4 lengths(4/8/16/32K) × 5 pos(10/25/50/75/90%) × 50       | 1000               |
| B      | distractor **dilution** + type    | 5 density(0/25/50/75/90%) × 3 types × 3 lengths(8/16/32K) × 40 | 1800          |
| C      | confusable-entity **overlap**     | 4 sim levels × 3 distances(near/med/far) × 50            | 600                |
| D      | **multi-hop** chain length + gap  | 3 hops(2/3/4) × 3 gaps × 4 lengths(8/16/32/64K) × 40      | **1160**           |

Counts are the **actual v3 output** (total 4560), not the design targets. Subset D (1160 = 440×2-hop + 360×3-hop + 360×4-hop) is the one place feasibility shaping bites: `core_knowledge/fact_chains.json` holds 15 chains (7×3-hop + 8×4-hop, balanced across the 4 chain types). The builder skips any (hops, gap, length) cell where `max_needle_tokens + (hops-1)*gap + 400 > length`, so far/large-gap cells at short lengths are intentionally absent (e.g. 4-hop exists only at 8K×near … up through 64K). It still lands short of the design's nominal 1440 *by design*, not by a KB gap. **Subset D's hop count is gated entirely by `fact_chains.json` — to add 5-hop or more chains, extend that file (keep `hops[i].target == hops[i+1].entity` and `answer == hops[-1].target`); do not touch the build script.**

Each sample is one JSONL line with `{sample_id, subset, domain, total_length, total_length_unit, context, question, answer, ...subset-specific fields}`. The subset-specific fields (`position_ratio`, `dilution_type`+`noise_density`, `similarity_level`+`distance_level`, `chain_type`+`num_hops`+`distance_level`, etc.) drive the breakdown analysis in `evaluate.py:breakdown()` and the comparison plots. All 5 design domains are present and balanced (`computer_science / medicine / law / finance / education`, ~50 facts each, 252 total).

**3. Two NIAH scripts share structure but test different things:**

- `niah_test.py` — synthetic needle (`量子隼-7型...`), unrelated Chinese corpus filler. Tests pure retrieval.
- `niah_knowledge_test.py` — real fact from `facts_library.json` as needle, **same-domain** filler scraped from `subset_A.jsonl`. Tests parametric-memory interference.

Both follow the gkamradt depth × context-length grid and write per-cell JSONs to `niah_results/` and `niah_knowledge_results/` respectively. `plot_niah.py` consumes both and renders heatmaps matching the exact gkamradt visual style (downward arrow + "Top/Bottom of Document" labels + goal box).

**4. Per-model context grids live in dicts, not flags.**
Both NIAH scripts and `plot_niah.py` carry `MODEL_MAX_CTX` and `MODEL_CTX_LENGTHS` dicts keyed by Ollama model name. When you add a new model, add entries to **both** dicts in `niah_knowledge_test.py` and `niah_test.py`. The Ollama `num_ctx` option is set from `MODEL_MAX_CTX` — without it, Ollama silently truncates to its default 2K-4K window.

## Conventions that bite

**Tokens vs. characters.** Chinese text on Qwen2.5 tokenizes at ~**1.71 chars/token** (`CHARS_PER_TOKEN = 1.71`, measured: 4096 chars = 2390 tokens). This is a load-bearing constant in `niah_*.py` and `plot_niah.py`.

- `build_dataset_v3.py` is **token-strict**: it counts/truncates with the real Qwen2.5-7B-Instruct tokenizer (`n_tokens()`), so the `total_length` field in `data/*.jsonl` is in *tokens* (and each line carries `total_length_unit: "tokens"` to make this explicit). No char↔token conversion is needed for the current dataset. (Older v1/v2 backups in `data_v*_backup/` used a char-as-1-token approximation that under-counted — multiply by 1.71 if you ever compare against those.)
- NIAH scripts take `--context-lengths-tokens` (real tokens) and internally multiply by 1.71 to derive char-level haystack size.
- Plot scripts always display tokens; old char-based result files are auto-converted via `chars_to_tokens()`.

**Thinking-model output.** DeepSeek-R1 and Qwen3.5 emit `<think>...</think>` blocks. Every scoring path (`evaluate.py:strip_thinking`, `niah_*.py:strip_thinking`) strips them before scoring. Keep `--max-tokens ≥ 1024` for these models or the answer gets clipped inside the thinking block.

**Checkpoint files in `results/`.** `evaluate.py` writes `{safe_model}_subset{X}_checkpoint.jsonl` while running, appends per-sample. On successful completion of all requested subsets it deletes them. **If a run is interrupted, do not delete the checkpoints** — re-running the same command resumes from them. Stale checkpoints from a previous schema can silently re-score with old logic; if you change scoring, delete checkpoints first.

**Model-name safety.** Colons in Ollama model IDs (e.g. `qwen2.5:7b`) are munged via `re.sub(r"[:/\\]", "_", ...)` for filenames. `compare_models.py:extract_model_name` reverses this for display by replacing `_` back to `:` — which is wrong for models with underscores in their actual name; if you add such a model, fix that function.

**Result file naming is parsed downstream.** `plot_niah.py` glob-matches `_tok{N}_dep{M}_results.json` (new) and falls back to `_len{N}_depth{M}_results.json` (old, char-based). If you rename, update the loaders.

## Working directory

Every script resolves its paths from `Path(__file__).parent` (`build_dataset_v3.py` uses `BASE_DIR = Path(__file__).parent` → `data/` + `core_knowledge/`), so all of them are location-independent and can be run from anywhere. (The old `build_dataset_fast.py`, which hard-coded a relative `base_dir` and had to be launched from the parent dir, has been removed.)

## Design provenance & the design↔implementation gap

The spec this repo implements lives **one level up**, in sibling folders of `PAC-Test-Dataset/` (not under it):

- `../实验设计/` — the blueprint. `测试集选型.docx` is the authoritative PAC-Test spec (the four subsets, sample templates, variable grids — `build_dataset_v3.py`'s docstring cites it by name). `模型选型.docx` lists the **intended 10-model lineup**; `测试集整理.xlsx` lists dataset provenance; `实验设计.xlsx` is the empty results-table skeleton (a fixed-16K cross-model table + a public-benchmark table + a scale-gradient table — `make_16k_table.py` fills the 16K one). `论文整理_V2.xlsx` has per-paper AI summaries of 13 core papers. (`实验方法整理.docx` is a 0-byte stub; `论文整理.docx` is an empty header superseded by the xlsx.)
- `../资料参考/` — 9 reference papers, foldered by mechanism: **核心实验** = *Lost in the Middle* (TACL'24, the U-shape/"lost-in-the-middle" basis) + *Loong* (multi-doc QA); **基准测试** = *U-NIAH* + *LongBench Pro*; **实验设计** = *DENIAHL*; **机制归因** = two *HoPE* variants + *BFloat16-breaks-RoPE / AnchorAttention*; **注意力可视化** = *CCA-Attention*.
- `../过程思考整理/` — `2.pdf` is a 14-page **forward roadmap** (a ChatGPT report) that goes well beyond what's built: controllable **coreference/anaphora** test sets, **attention-path tracing** (rollout/flow from answer token back to the anchor span), and **three training-free memory-augmentation methods** (structured MEMORY-block re-presentation / RAG-oracle / inference-time ALiBi-like attention-bias injection), plus a stats plan (McNemar / paired bootstrap / BH-FDR). `上下文记忆文献调研.docx` is RULER/Loong reading notes.

**What the current code actually implements vs. the design — known gaps:**

1. **Models: 3 local, not 10 cloud.** `run_all_models.py` + the NIAH `MODEL_*` dicts cover only `qwen2.5:7b`, `deepseek-r1:7b`, `qwen3.5:9b` (whatever Ollama has locally). The design's 10-model sweep (Qwen2.5 7/14/32/72B, Llama-3.1 8/70B, DeepSeek-V3, Mistral-7B, GLM-4-9B, Yi-1.5-34B) is aspirational — only `qwen2.5:7b` overlaps; `deepseek-r1:7b`≠DeepSeek-V3 and `qwen3.5:9b` isn't in the design at all. The scale-gradient and SWA-vs-full-attention comparisons the design motivates **cannot** be produced from the current model set.
2. **Public benchmarks not wired up.** The design wants LongBench (generalization) + NIAH + RULER. Only **NIAH** exists here (`niah_test.py`, `niah_knowledge_test.py`); **LongBench and RULER are not implemented**, so the public-benchmark row of `实验设计.xlsx` is unfillable from this repo.
3. **PAC dataset matches the spec.** Grids, positions, noise types, similarity levels, domains all match `测试集选型.docx`. Subset D now includes 4-hop (4-hop chains were added to `fact_chains.json`); its 1160 total reflects intended feasibility skips, not a missing-data gap.

**Scope decision (locked):** this repo does **PAC-Test only**. LongBench/RULER, the full 10-model sweep, and the entire `2.pdf` roadmap (coreference subset, attention rollout/flow attribution, the three training-free memory-augmentation methods) are **out of scope** — do not build them unless explicitly reopened.

**Scoring (primary = NIAH-faithful LLM-as-judge).** `evaluate.py` ports gkamradt NIAH's `OpenAIEvaluator`: a fixed **judge model scores each answer 1–10** on an accuracy rubric (1 完全无关 … 10 完全一致), via Ollama's native `/api/chat`. The original judges with GPT-4; offline we substitute a **fixed local judge** (`--judge-model`, default `gemma4:31b`) held constant across all evaluated models for cross-model comparability. `gemma4:31b` is chosen as a **neutral cross-family judge**: it shares no model family with any of the three PAC candidates (`qwen2.5:7b` / `deepseek-r1:7b` / `qwen3.5:9b`), so none of them is self-judged or judged by a same-family model (a Qwen judge would give same-family Qwen candidates a homophily edge). If you point `--judge-model` at a candidate or a same-family model, note that bias. Per-sample record carries `score` (1–10), `score_norm` (=score/10, the primary metric everywhere — `breakdown`, report, plots, summary `mean_score`/`acc10`), `score_src` (`judge`|`fallback`), plus **`em`/`contains` retained as secondary diagnostics**. Why the switch: EM under-counts verbose-but-correct answers, and Contains false-positives on subsets C/D where the gold is a substring of same-context distractors (e.g. 张冠李戴 misattribution scores Contains=True but judge=3). Knobs: `--no-judge` reverts to pure EM/Contains; `parse_judge_score` tolerates `[[N]]`/`[N]`/`评分：N`/trailing-number because local models don't reliably emit the `[[N]]` format; on a true parse/call failure it falls back to a 10/7/1 EM·Contains tier (`score_src="fallback"`). **Changing scoring invalidates `results/` checkpoints — delete them first** (stale checkpoints re-load old per-sample records without `score`). The NIAH scripts (`niah_*.py`) keep their own separate 10/7/1 substring scorer — that's fine there because their synthetic needle can't collide with filler. The design docs only specify "精确匹配 / 抽取式短答案 / 准确率" conceptually; the 1–10 judge is this repo's faithfulness to the gkamradt NIAH GitHub project, chosen per the owner's "github 上 NIAH 项目啥样咱们就啥样".

⚠ **Downstream not yet migrated:** `compare_models.py`, `plot_pac.py`, and `make_16k_table.py` still read `em`/`contains` from `*_summary.json`. They keep working (those keys still exist) but **do not yet surface the new `score_norm`/`mean_score`** — migrate them when cross-model NIAH-score tables/plots are needed.
