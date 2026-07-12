from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.longbench import longbench_metric_kind
from lmaf.data.pac import adapt_external_pac_sample, score_pac_sample
from lmaf.data.ruler_adapter import score_ruler_sample
from lmaf.eval.metrics import best_contains
from lmaf.eval.scorers import score_niah, score_qa, score_summarization
from lmaf.inference.client import create_inference_client, resolve_provider_model
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS, append_jsonl, collect_jsonl, iter_jsonl_paths, read_jsonl, utc_timestamp
from lmaf.utils.token_count import TokenCounter


@dataclass(frozen=True)
class ModelPlan:
    alias: str
    api_model: str
    max_model_len: int
    enable_thinking: bool = False


@dataclass(frozen=True)
class WorkItem:
    experiment: str
    dataset_label: str
    model: ModelPlan
    output: Path
    sample: dict[str, Any]
    prompt: str
    prompt_tokens: int
    max_tokens: int
    score_fn: str


FRAMEWORK_MODELS: dict[str, ModelPlan] = {
    "qwen35_9b": ModelPlan("qwen35_9b", "Qwen/Qwen3.5-9B", 65536),
    "qwen3_8b": ModelPlan("qwen3_8b", "Qwen/Qwen3-8B", 32768),
    "qwen35_27b": ModelPlan("qwen35_27b", "Qwen/Qwen3.5-27B", 32768),
    "qwen35_35b_a3b": ModelPlan("qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", 65536),
    "qwen35_122b_a10b": ModelPlan("qwen35_122b_a10b", "Qwen/Qwen3.5-122B-A10B", 32768),
    "hunyuan_a13b": ModelPlan("hunyuan_a13b", "tencent/Hunyuan-A13B-Instruct", 65536),
    "seed_oss_36b": ModelPlan("seed_oss_36b", "ByteDance-Seed/Seed-OSS-36B-Instruct", 32768),
    "qwen3_14b_no_thinking": ModelPlan("qwen3_14b_no_thinking", "Qwen/Qwen3-14B", 32768),
    "qwen3_14b_thinking": ModelPlan("qwen3_14b_thinking", "Qwen/Qwen3-14B", 32768, enable_thinking=True),
}

PAC_SUBSETS = {
    "A": ("A_position", "subset_A.jsonl"),
    "B": ("B_interference", "subset_B.jsonl"),
    "C": ("C_overlap", "subset_C.jsonl"),
    "D": ("D_multihop", "subset_D.jsonl"),
}

FILE_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)
THREAD_LOCAL = threading.local()


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)
    experiments = parse_csv(args.experiments)
    api_keys = resolve_api_keys(args)
    counter = TokenCounter(args.tokenizer)
    work = build_work_items(args, models, experiments, counter)
    write_plan(args, work)
    if args.dry_run:
        print_plan(work)
        return
    if not work:
        print("No unfinished retryable work found.")
        return
    if args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")

    random.shuffle(work)
    run_parallel(args, work, api_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast parallel resume runner. It only calls API for missing/retryable samples."
    )
    parser.add_argument("--experiments", default="niah,longbench,ruler,pac", help="Comma list: niah,longbench,ruler,pac")
    parser.add_argument("--models", default=",".join(FRAMEWORK_MODELS), help="Comma-separated model aliases.")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true", help="Force thinking on for every selected model.")
    parser.add_argument("--stop-after", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", default="results/logs/unfinished_fast_plan.json")

    parser.add_argument("--niah-data", default="data/generated/niah_batch/framework_v2_without_fast16k")
    parser.add_argument("--niah-output-dir", default="results/raw/niah_batch/framework_v2_without_fast16k/framework_v2_extra")

    parser.add_argument("--longbench-data", default="data/processed/longbench_ruler_batch/framework_v2/longbench")
    parser.add_argument(
        "--longbench-output-dir",
        default="results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/longbench",
    )
    parser.add_argument(
        "--longbench-tasks",
        default="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news",
    )
    parser.add_argument("--longbench-truncate", choices=["none", "middle"], default="none")

    parser.add_argument("--ruler-data", default="data/processed/longbench_ruler_batch/framework_v2/ruler")
    parser.add_argument(
        "--ruler-output-dir",
        default="results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/ruler",
    )

    parser.add_argument(
        "--pac-source-data",
        default=os.getenv("PAC_TEST_DATA_DIR", r"D:\Workspace\llm-longcontext-attribution\上下文机制探究\PAC-Test-Dataset\data"),
    )
    parser.add_argument("--pac-output-dir", default="results/raw/pac_batch/pac_main")
    parser.add_argument("--pac-subsets", default="A,B,C,D")
    return parser.parse_args()


def resolve_models(value: str) -> list[ModelPlan]:
    out: list[ModelPlan] = []
    for name in parse_csv(value):
        if name not in FRAMEWORK_MODELS:
            raise SystemExit(f"Unknown model alias for fast runner: {name}")
        out.append(FRAMEWORK_MODELS[name])
    if not out:
        raise SystemExit("--models must include at least one model alias")
    return out


def resolve_api_keys(args: argparse.Namespace) -> list[str]:
    if args.api_key:
        return [args.api_key]
    keys = os.getenv("SILICONFLOW_API_KEYS")
    if keys:
        return [key.strip() for key in keys.split(",") if key.strip()]
    key = os.getenv("SILICONFLOW_API_KEY")
    return [key] if key else []


def build_work_items(
    args: argparse.Namespace,
    models: list[ModelPlan],
    experiments: list[str],
    counter: TokenCounter,
) -> list[WorkItem]:
    work: list[WorkItem] = []
    if "niah" in experiments:
        samples = collect_jsonl(ROOT / args.niah_data)
        work.extend(build_niah_work(args, models, samples))
    if "longbench" in experiments:
        samples = load_longbench_samples(ROOT / args.longbench_data, set(parse_csv(args.longbench_tasks)))
        work.extend(build_longbench_work(args, models, samples, counter))
    if "ruler" in experiments:
        samples = collect_jsonl(ROOT / args.ruler_data)
        work.extend(build_ruler_work(args, models, samples, counter))
    if "pac" in experiments:
        work.extend(build_pac_work(args, models, counter))
    if args.stop_after is not None:
        work = work[: args.stop_after]
    return work


def build_niah_work(args: argparse.Namespace, models: list[ModelPlan], samples: list[dict[str, Any]]) -> list[WorkItem]:
    work: list[WorkItem] = []
    for model in models:
        output = ROOT / args.niah_output_dir / f"{model.alias}.jsonl"
        completed = completed_ids(output)
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            target_len = int(sample.get("length_tokens_target") or 0)
            if target_len > model.max_model_len:
                append_terminal_skip(output, sample, args, model, target_len, "skipped_by_model_length")
                completed.add(sample_id)
                continue
            work.append(
                WorkItem(
                    experiment="niah",
                    dataset_label="niah_without_fast16k",
                    model=model,
                    output=output,
                    sample=sample,
                    prompt=str(sample["prompt"]),
                    prompt_tokens=int(sample.get("length_tokens_actual") or target_len),
                    max_tokens=args.max_tokens,
                    score_fn="niah",
                )
            )
    return work


def build_longbench_work(
    args: argparse.Namespace,
    models: list[ModelPlan],
    samples: list[dict[str, Any]],
    counter: TokenCounter,
) -> list[WorkItem]:
    work: list[WorkItem] = []
    for model in models:
        output = ROOT / args.longbench_output_dir / f"{model.alias}.jsonl"
        completed = completed_ids(output)
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            prompt = str(sample["prompt"])
            prompt_tokens = counter.count(prompt)
            if prompt_tokens > model.max_model_len:
                if args.longbench_truncate == "middle":
                    prompt = middle_truncate(prompt, model.max_model_len, counter)
                    prompt_tokens = counter.count(prompt)
                    sample = dict(sample)
                    sample["truncation"] = "middle"
                else:
                    append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                    completed.add(sample_id)
                    continue
            work.append(
                WorkItem(
                    experiment="longbench",
                    dataset_label="longbench_framework_v2",
                    model=model,
                    output=output,
                    sample=sample,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    max_tokens=args.max_tokens,
                    score_fn="longbench",
                )
            )
    return work


def build_ruler_work(
    args: argparse.Namespace,
    models: list[ModelPlan],
    samples: list[dict[str, Any]],
    counter: TokenCounter,
) -> list[WorkItem]:
    work: list[WorkItem] = []
    for model in models:
        output = ROOT / args.ruler_output_dir / f"{model.alias}.jsonl"
        completed = completed_ids(output)
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            prompt = str(sample["prompt"])
            prompt_tokens = counter.count(prompt)
            if prompt_tokens > model.max_model_len:
                append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                completed.add(sample_id)
                continue
            work.append(
                WorkItem(
                    experiment="ruler",
                    dataset_label="ruler_framework_v2",
                    model=model,
                    output=output,
                    sample=sample,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    max_tokens=args.max_tokens,
                    score_fn="ruler",
                )
            )
    return work


def build_pac_work(args: argparse.Namespace, models: list[ModelPlan], counter: TokenCounter) -> list[WorkItem]:
    work: list[WorkItem] = []
    for subset_key in parse_csv(args.pac_subsets):
        subset_key = subset_key.upper()
        if subset_key not in PAC_SUBSETS:
            raise SystemExit(f"Unsupported PAC subset: {subset_key}")
        subtask, filename = PAC_SUBSETS[subset_key]
        source = pac_source_for_subset(args.pac_source_data, filename)
        samples = [adapt_external_pac_sample(row, counter=counter, count_tokens=False) for row in read_jsonl(source)]
        samples = [sample for sample in samples if sample.get("subtask") == subtask]
        for model in models:
            output = ROOT / args.pac_output_dir / subset_key / f"{model.alias}.jsonl"
            completed = completed_ids(output)
            for sample in samples:
                sample_id = str(sample["sample_id"])
                if sample_id in completed:
                    continue
                prompt = str(sample["prompt"])
                prompt_tokens = prompt_tokens_from_sample(sample, prompt, counter)
                if prompt_tokens > model.max_model_len:
                    append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                    completed.add(sample_id)
                    continue
                work.append(
                    WorkItem(
                        experiment="pac",
                        dataset_label=f"pac_{subset_key}_{subtask}",
                        model=model,
                        output=output,
                        sample=sample,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        max_tokens=args.max_tokens,
                        score_fn="pac",
                    )
                )
    return work


def completed_ids(output: Path) -> set[str]:
    completed: set[str] = set()
    if not output.exists():
        return completed
    for row in read_jsonl(output):
        sample_id = row.get("sample_id")
        if sample_id not in (None, "") and row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS):
            completed.add(str(sample_id))
    return completed


def append_terminal_skip(
    output: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    model: ModelPlan,
    prompt_tokens: int,
    reason: str,
) -> None:
    row = dict(sample)
    row.update(
        {
            "model": model.alias,
            "provider": args.provider,
            "api_model": resolve_provider_model(args.provider, model.alias),
            "prediction": "",
            "latency_sec": 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "timestamp": utc_timestamp(),
            "score": 0.0,
            "metric": reason,
            "error": reason,
            "error_type": reason,
        }
    )
    append_with_lock(output, row)


def run_parallel(args: argparse.Namespace, work: list[WorkItem], api_keys: list[str]) -> None:
    started = time.perf_counter()
    total = len(work)
    done = 0
    errors = 0
    by_dataset = Counter(item.dataset_label for item in work)
    print("Fast unfinished runner plan:")
    for key, count in sorted(by_dataset.items()):
        print(f"  {key}: {count} API calls")
    print(f"Total pending API calls: {total}")
    print(f"Workers: {args.max_workers}")
    print("Tip: stop the old sequential batch processes before running this for maximum speed.")

    in_flight_limit = args.max_in_flight or max(args.max_workers * 2, args.max_workers)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        pending = []
        iterator = iter(work)
        while True:
            while len(pending) < in_flight_limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(run_one, args, item, choose_key(api_keys)))
            if not pending:
                break
            finished, pending_set = wait(pending, return_when=FIRST_COMPLETED)
            pending = list(pending_set)
            for future in finished:
                ok, label = future.result()
                done += 1
                errors += 0 if ok else 1
                if done % args.max_workers == 0 or done == total:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    rate = done / elapsed
                    eta = (total - done) / rate if rate else 0
                    print(
                        f"[{done}/{total}] ok={done - errors} errors={errors} "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}m last={label}",
                        flush=True,
                    )


def run_one(args: argparse.Namespace, item: WorkItem, api_key: str | None) -> tuple[bool, str]:
    client = get_thread_client(args, item.model, api_key)
    result = client.generate(
        prompt=item.prompt,
        request_id=str(item.sample["sample_id"]),
        max_tokens=item.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    row = dict(item.sample)
    row.update(
        {
            "model": item.model.alias,
            "provider": args.provider,
            "api_model": client.served_model_name,
            "prompt": item.prompt,
            "prediction": result.response_text,
            "latency_sec": result.latency_sec,
            "prompt_tokens": result.prompt_tokens or item.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "timestamp": utc_timestamp(),
            "error": result.error,
        }
    )
    if result.error:
        row.update({"score": 0.0, "metric": "request_error", "error_type": "request_error"})
    else:
        row.update(score_item(item, result.response_text))
    append_with_lock(item.output, row)
    return result.error is None, f"{item.dataset_label}/{item.model.alias}"


def get_thread_client(args: argparse.Namespace, model: ModelPlan, api_key: str | None):
    cache = getattr(THREAD_LOCAL, "clients", None)
    if cache is None:
        cache = {}
        THREAD_LOCAL.clients = cache
    key = (args.provider, model.alias, api_key or "", bool(model.enable_thinking or args.enable_thinking))
    if key not in cache:
        cache[key] = create_inference_client(
            provider=args.provider,
            model_name=model.alias,
            endpoint=args.endpoint,
            api_key=api_key,
            timeout=args.timeout,
            retry=args.retry,
            backoff=(1, 2, 4, 8),
            enable_thinking=model.enable_thinking or args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
    return cache[key]


def score_item(item: WorkItem, prediction: str) -> dict[str, Any]:
    if item.score_fn == "niah":
        answer = item.sample.get("answer")
        if isinstance(answer, list):
            score = float(all(best_contains(prediction, one) for one in answer))
            return {"score": score, "metric": "all_contains", "contains_answer": int(score)}
        return score_niah(prediction, str(answer))
    if item.score_fn == "longbench":
        if longbench_metric_kind(str(item.sample.get("task"))) == "summarization":
            return score_summarization(prediction, item.sample.get("answers", []))
        return score_qa(prediction, item.sample.get("answers", []))
    if item.score_fn == "ruler":
        return score_ruler_sample(item.sample, prediction)
    if item.score_fn == "pac":
        return score_pac_sample(item.sample, prediction)
    raise ValueError(f"Unknown score function: {item.score_fn}")


def append_with_lock(output: Path, row: dict[str, Any]) -> None:
    lock = FILE_LOCKS[output]
    with lock:
        append_jsonl(output, row)


def load_longbench_samples(input_dir: Path, tasks: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_jsonl_paths(input_dir):
        if path.stem in tasks:
            rows.extend(collect_jsonl(path))
    return rows


def pac_source_for_subset(source_data: str, filename: str) -> Path:
    source = Path(source_data)
    if source.is_file():
        return source
    return source / filename


def prompt_tokens_from_sample(sample: dict[str, Any], prompt: str, counter: TokenCounter) -> int:
    value = sample.get("length_tokens_actual") or sample.get("length_tokens_target") or sample.get("total_length")
    if value not in (None, ""):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    return counter.count(prompt)


def middle_truncate(prompt: str, max_tokens: int, counter: TokenCounter) -> str:
    marker = "\n\n[... middle truncated ...]\n\n"
    marker_tokens = counter.count(marker)
    keep = max(1, max_tokens - marker_tokens)
    left = keep // 2
    right = keep - left
    tokens = counter.encode(prompt)
    return counter.decode(tokens[:left]) + marker + counter.decode(tokens[-right:])


def choose_key(api_keys: list[str]) -> str | None:
    if not api_keys:
        return None
    return random.choice(api_keys)


def write_plan(args: argparse.Namespace, work: list[WorkItem]) -> None:
    path = ROOT / args.plan_output
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter((item.experiment, item.dataset_label, item.model.alias) for item in work)
    rows = [
        {"experiment": exp, "dataset": dataset, "model": model, "pending_api_calls": count}
        for (exp, dataset, model), count in sorted(summary.items())
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote fast-run plan to {path}")


def print_plan(work: list[WorkItem]) -> None:
    summary = Counter((item.experiment, item.dataset_label, item.model.alias) for item in work)
    for (experiment, dataset, model), count in sorted(summary.items()):
        print(f"{experiment:10s} {dataset:28s} {model:24s} {count}")
    print(f"Total pending API calls: {sum(summary.values())}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
