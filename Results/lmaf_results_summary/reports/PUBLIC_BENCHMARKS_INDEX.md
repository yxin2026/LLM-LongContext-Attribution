# Public Benchmark Report Index

## Recommended Main Version

Use `public_benchmarks_summary_budget_core` as the paper-facing version.

Reason: it uses the balanced `budget_core_main` run, with more even per-model sample budgets and better NIAH/RULER coverage than the older large-run folders.

Main artifacts:

- `public_benchmarks_summary_budget_core/summary_report.md`
- `public_benchmarks_summary_budget_core/summary_tables.xlsx`
- `public_benchmarks_summary_budget_core/tables/`
- `public_benchmarks_summary_budget_core/figures/`
- `public_benchmarks_summary_budget_core/code_snapshot/`

Included raw roots:

- LongBench: `results/raw/budget_core/budget_core_main/longbench`
- NIAH: `results/raw/budget_core/budget_core_main/niah`
- RULER: `results/raw/budget_core/budget_core_main/ruler`

## Supplementary Large-Run Version

Use `public_benchmarks_summary` as supplementary material or for checking older, larger framework-v2 runs.

Reason: it contains larger historical runs, especially LongBench, but NIAH/RULER have low successful-call coverage due to earlier quota/rate-limit failures.

Main artifacts:

- `public_benchmarks_summary/summary_report.md`
- `public_benchmarks_summary/summary_tables.xlsx`
- `public_benchmarks_summary/tables/`
- `public_benchmarks_summary/figures/`
- `public_benchmarks_summary/code_snapshot/`

Included raw roots:

- LongBench: `results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/longbench`
- NIAH: `results/raw/niah_batch/framework_v2_without_fast16k/framework_v2_extra`
- RULER: `results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/ruler`

## Reproduction Commands

Generate the recommended paper-facing version:

```bat
python scripts\summarize_public_benchmarks.py --output results\reports\public_benchmarks_summary_budget_core --longbench-root results\raw\budget_core\budget_core_main\longbench --niah-root results\raw\budget_core\budget_core_main\niah --ruler-root results\raw\budget_core\budget_core_main\ruler
C:\Users\GET-DATA402\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe scripts\build_public_benchmarks_workbook.mjs --report-dir results\reports\public_benchmarks_summary_budget_core
```

Generate the supplementary large-run version:

```bat
python scripts\summarize_public_benchmarks.py --output results\reports\public_benchmarks_summary
C:\Users\GET-DATA402\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe scripts\build_public_benchmarks_workbook.mjs --report-dir results\reports\public_benchmarks_summary
```

## Reporting Note

Scores in the generated reports are computed over successful API calls by default. API quota, rate-limit, and connection failures are not treated as model ability failures; they are reported separately through `coverage` and `error_rate`.
