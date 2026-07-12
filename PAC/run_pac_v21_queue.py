from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAC_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pac2_formal as formal
import run_pac_d_v21_pilot as d21
from lmaf.data.pac2 import score_pac2_sample
from lmaf.inference.client import resolve_provider_model
from lmaf.utils.io import append_jsonl, read_jsonl, utc_timestamp


SUBSET_FILES = {
    "A": ("PAC-A_position", PAC_ROOT / "data" / "PAC-A_position" / "samples.jsonl"),
    "B": ("PAC-B_interference", PAC_ROOT / "data" / "PAC-B_interference" / "samples.jsonl"),
    "C": ("PAC-C_binding_capacity", PAC_ROOT / "data" / "PAC-C_binding_capacity" / "samples.jsonl"),
    "D21": ("PAC-D_multihop_false_chain", PAC_ROOT / "data" / "PAC-D_v2_1_hard" / "samples.jsonl"),
}

FILE_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)
COUNTER_LOCK = threading.Lock()
GLOBAL_REQUEST_LOCK = threading.Lock()
GLOBAL_LAST_REQUEST_AT = 0.0
MODEL_COOLDOWN_LOCK = threading.Lock()
MODEL_AVAILABLE_AT: dict[tuple[str, str], float] = defaultdict(float)
CONNECTION_GUARD_LOCK = threading.Lock()
CONNECTION_ERROR_TIMES: list[float] = []
CONNECTION_AVAILABLE_AT = 0.0


def main() -> None:
    args = parse_args()
    samples = load_selected_samples_from_args(args)
    selected_models = formal.resolve_requested_models(args.models)
    manifest = {"subsets": {name: {} for name in sorted({row["formal_subset"] for row in samples})}}
    work = formal.build_work(args, manifest, samples, selected_models)
    if args.stop_after is not None:
        work = work[: args.stop_after]
    write_plan(args, work)
    print_plan(args, work)

    if args.dry_run:
        formal.summarize(args)
        return
    if args.summarize_only:
        formal.summarize(args)
        return
    api_keys = resolve_api_keys(args)
    if work and args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if not work:
        print("No pending PAC v2.1 queue work.")
        formal.summarize(args)
        return
    run_queue(args, work, api_keys)
    formal.summarize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue-based PAC v2.1 full runner. API keys pull tasks dynamically until the queue is empty."
    )
    parser.add_argument("--run-id", default="pac_v21_full_queue")
    parser.add_argument("--subsets", default="A,B,C,D21", help="Comma list: A,B,C,D21.")
    parser.add_argument("--models", default="auto", help="'auto' uses sample model scopes, or comma model aliases.")
    parser.add_argument("--d21-samples-per-condition", type=int, default=2)
    parser.add_argument("--d21-seed", type=int, default=91021)
    parser.add_argument("--force-generate-d21", action="store_true")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-keys", default=None, help="Comma-separated API keys. Overrides SILICONFLOW_API_KEYS.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--retry", type=int, default=1, help="Per-request SDK retry count. Queue retries are controlled separately.")
    parser.add_argument("--queue-max-attempts", type=int, default=4)
    parser.add_argument("--rate-limit-cooldown-sec", type=float, default=90.0)
    parser.add_argument("--transient-cooldown-sec", type=float, default=25.0)
    parser.add_argument(
        "--connection-cooldown-sec",
        type=float,
        default=180.0,
        help="Cooldown for APIConnectionError/timeout before retrying the same sample.",
    )
    parser.add_argument(
        "--connection-max-attempts",
        type=int,
        default=2,
        help="Queue attempts for connection errors before writing a retryable error row for a later run.",
    )
    parser.add_argument(
        "--connection-burst-threshold",
        type=int,
        default=3,
        help="Pause globally after this many connection errors in the burst window.",
    )
    parser.add_argument("--connection-burst-window-sec", type=float, default=90.0)
    parser.add_argument("--global-connection-cooldown-sec", type=float, default=180.0)
    parser.add_argument(
        "--model-rate-limit-cooldown-sec",
        type=float,
        default=150.0,
        help="When one model/key hits TPM 429, pause only that model/key while other models keep running.",
    )
    parser.add_argument(
        "--global-request-delay-sec",
        type=float,
        default=0.0,
        help="Global minimum gap between API request starts across all workers. Use this for account-level TPM limits.",
    )
    parser.add_argument("--per-key-delay-sec", type=float, default=0.0)
    parser.add_argument("--slots-per-key", type=int, default=1, help="Default 1 means each API key has one active request slot.")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260707)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "raw" / "pac_v21_queue"))
    parser.add_argument("--report-root", default=str(ROOT / "results" / "reports" / "pac_v21_queue"))
    parser.add_argument("--plan-output", default=str(ROOT / "results" / "logs" / "pac_v21_queue_plan.json"))
    return parser.parse_args()


def load_selected_samples(value: str) -> list[dict[str, Any]]:
    raise RuntimeError("load_selected_samples(args) should be used")


def load_selected_samples_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in formal.parse_csv(args.subsets):
        normalized = key.upper()
        if normalized == "D":
            normalized = "D21"
        if normalized not in SUBSET_FILES:
            raise SystemExit(f"Unknown subset {key}. Use A,B,C,D21.")
        _subset, path = SUBSET_FILES[normalized]
        if normalized == "D21":
            ensure_d21_samples(args)
        if not path.exists():
            raise SystemExit(f"Subset file not found: {path}")
        rows.extend(read_jsonl(path))
    return rows


def ensure_d21_samples(args: argparse.Namespace) -> None:
    expected = 6 * args.d21_samples_per_condition
    path = SUBSET_FILES["D21"][1]
    regenerate = args.force_generate_d21 or not path.exists()
    if not regenerate:
        rows = list(read_jsonl(path))
        regenerate = len(rows) != expected
    if regenerate:
        rows = d21.generate_dataset(args.d21_samples_per_condition, args.d21_seed)
        d21.write_jsonl(path, rows)
        d21.write_json(d21.SUMMARY_PATH, d21.summarize_samples(rows))
        print(f"Prepared PAC-D v2.1 samples: {len(rows)} rows at {path}")


def resolve_api_keys(args: argparse.Namespace) -> list[str]:
    if args.api_keys:
        return [key.strip() for key in args.api_keys.split(",") if key.strip()]
    if args.api_key:
        return [args.api_key]
    keys = os.getenv("SILICONFLOW_API_KEYS")
    if keys:
        return [key.strip() for key in keys.split(",") if key.strip()]
    key = os.getenv("SILICONFLOW_API_KEY")
    return [key] if key else []


def run_queue(args: argparse.Namespace, work: list[formal.WorkItem], api_keys: list[str]) -> None:
    rng = random.Random(args.shuffle_seed)
    work = list(work)
    rng.shuffle(work)
    task_queue: queue.PriorityQueue[tuple[float, int, formal.WorkItem]] = queue.PriorityQueue()
    sequence = itertools.count()
    for item in work:
        task_queue.put((0.0, next(sequence), item))

    key_slots = build_key_slots(args, api_keys)
    attempts: dict[tuple[str, str, str], int] = defaultdict(int)
    stats = {
        "done": 0,
        "ok": 0,
        "errors": 0,
        "requeued": 0,
        "started_at": time.perf_counter(),
        "total": len(work),
    }
    print(
        f"PAC v2.1 queue started: tasks={len(work)}, workers={len(key_slots)}, "
        f"slots_per_key={args.slots_per_key}, global_request_delay_sec={args.global_request_delay_sec}"
    )
    threads: list[threading.Thread] = []
    for worker_index, api_key in enumerate(key_slots, start=1):
        thread = threading.Thread(
            target=worker_loop,
            args=(args, worker_index, api_key, task_queue, sequence, attempts, stats),
            daemon=True,
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print(
        f"PAC v2.1 queue finished: ok={stats['ok']} final_errors={stats['errors']} "
        f"requeued={stats['requeued']}"
    )


def build_key_slots(args: argparse.Namespace, api_keys: list[str]) -> list[str | None]:
    if args.provider == "siliconflow":
        if args.slots_per_key <= 0:
            raise SystemExit("--slots-per-key must be positive")
        return [key for key in api_keys for _ in range(args.slots_per_key)]
    workers = max(1, args.slots_per_key)
    return [api_keys[0] if api_keys else None for _ in range(workers)]


def worker_loop(
    args: argparse.Namespace,
    worker_index: int,
    api_key: str | None,
    task_queue: queue.PriorityQueue[tuple[float, int, formal.WorkItem]],
    sequence: itertools.count,
    attempts: dict[tuple[str, str, str], int],
    stats: dict[str, Any],
) -> None:
    last_request_at = 0.0
    while True:
        try:
            available_at, _seq, item = task_queue.get(timeout=2.0)
        except queue.Empty:
            return
        wait_until_available = available_at - time.perf_counter()
        if wait_until_available > 0:
            time.sleep(wait_until_available)
        now = time.perf_counter()
        wait_for = last_request_at + args.per_key_delay_sec - now
        if wait_for > 0:
            time.sleep(wait_for)
        connection_wait_until = get_connection_available_at()
        if connection_wait_until > time.perf_counter():
            task_queue.put((connection_wait_until, next(sequence), item))
            task_queue.task_done()
            time.sleep(0.05)
            continue
        model_wait_until = get_model_available_at(item, api_key)
        if model_wait_until > time.perf_counter():
            task_queue.put((model_wait_until, next(sequence), item))
            task_queue.task_done()
            time.sleep(0.05)
            continue
        throttle_global_request_starts(args.global_request_delay_sec)
        model_wait_until = get_model_available_at(item, api_key)
        if model_wait_until > time.perf_counter():
            task_queue.put((model_wait_until, next(sequence), item))
            task_queue.task_done()
            time.sleep(0.05)
            continue
        last_request_at = time.perf_counter()

        key = item_key(item)
        attempts[key] += 1
        result_row = call_model(args, item, api_key)
        error = str(result_row.get("error") or "")
        if error and is_transient_error(error):
            max_attempts = max_attempts_for_error(args, error)
            if attempts[key] < max_attempts:
                if is_rate_limit_error(error):
                    cooldown = args.rate_limit_cooldown_sec
                    model_available_at = set_model_cooldown(args, item, api_key)
                    retry_at = max(time.perf_counter() + cooldown, model_available_at)
                    cooldown_label = max(cooldown, model_available_at - time.perf_counter())
                elif is_connection_error(error):
                    cooldown = args.connection_cooldown_sec
                    connection_available_at = note_connection_error(args)
                    retry_at = max(time.perf_counter() + cooldown, connection_available_at)
                    cooldown_label = max(cooldown, connection_available_at - time.perf_counter())
                else:
                    cooldown = args.transient_cooldown_sec
                    retry_at = time.perf_counter() + cooldown
                    cooldown_label = cooldown
                with COUNTER_LOCK:
                    stats["requeued"] += 1
                    print(
                        f"[worker {worker_index}] transient error, requeue after {cooldown_label:.0f}s "
                        f"(attempt {attempts[key]}/{max_attempts}): "
                        f"{item.subset}/{item.model.alias}/{item.sample['sample_id']} :: {short_error(error)}",
                        flush=True,
                    )
                task_queue.put((retry_at, next(sequence), item))
                task_queue.task_done()
                continue
            if is_connection_error(error):
                result_row["error_type"] = "connection_error_deferred"
                result_row["metric"] = "connection_error_deferred"

        append_with_lock(item.output, result_row)
        ok = not error
        with COUNTER_LOCK:
            stats["done"] += 1
            stats["ok"] += 1 if ok else 0
            stats["errors"] += 0 if ok else 1
            done = int(stats["done"])
            if done % max(1, args.progress_every) == 0 or done == stats["total"]:
                elapsed = max(1e-6, time.perf_counter() - float(stats["started_at"]))
                rate = done / elapsed
                eta = (stats["total"] - done) / rate if rate else 0
                print(
                    f"[{done}/{stats['total']}] ok={stats['ok']} final_errors={stats['errors']} "
                    f"requeued={stats['requeued']} rate={rate:.3f}/s eta={eta/60:.1f}m "
                    f"last={item.subset}/{item.model.alias}",
                    flush=True,
                )
        task_queue.task_done()


def call_model(args: argparse.Namespace, item: formal.WorkItem, api_key: str | None) -> dict[str, Any]:
    client = formal.get_thread_client(args, item.model, api_key)
    result = client.generate(
        prompt=str(item.sample["prompt"]),
        request_id=f"{item.subset}/{item.model.alias}/{item.sample['sample_id']}",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    row = dict(item.sample)
    row.update(
        {
            "model": item.model.alias,
            "provider": args.provider,
            "api_model": client.served_model_name,
            "prediction": result.response_text,
            "latency_sec": result.latency_sec,
            "prompt_tokens": result.prompt_tokens or item.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "timestamp": utc_timestamp(),
            "error": result.error,
        }
    )
    if result.error:
        row.update({"score": 0.0, "field_accuracy": 0.0, "metric": "request_error", "error_type": "request_error"})
    else:
        row.update(score_pac2_sample(item.sample, result.response_text))
    return row


def throttle_global_request_starts(delay_sec: float) -> None:
    global GLOBAL_LAST_REQUEST_AT
    if delay_sec <= 0:
        return
    with GLOBAL_REQUEST_LOCK:
        now = time.perf_counter()
        wait_for = GLOBAL_LAST_REQUEST_AT + delay_sec - now
        if wait_for > 0:
            time.sleep(wait_for)
        GLOBAL_LAST_REQUEST_AT = time.perf_counter()


def model_cooldown_key(item: formal.WorkItem, api_key: str | None) -> tuple[str, str]:
    return (item.model.alias, api_key or "")


def get_model_available_at(item: formal.WorkItem, api_key: str | None) -> float:
    with MODEL_COOLDOWN_LOCK:
        return MODEL_AVAILABLE_AT[model_cooldown_key(item, api_key)]


def set_model_cooldown(args: argparse.Namespace, item: formal.WorkItem, api_key: str | None) -> float:
    available_at = time.perf_counter() + max(0.0, args.model_rate_limit_cooldown_sec)
    with MODEL_COOLDOWN_LOCK:
        key = model_cooldown_key(item, api_key)
        MODEL_AVAILABLE_AT[key] = max(MODEL_AVAILABLE_AT[key], available_at)
        return MODEL_AVAILABLE_AT[key]


def get_connection_available_at() -> float:
    with CONNECTION_GUARD_LOCK:
        return CONNECTION_AVAILABLE_AT


def note_connection_error(args: argparse.Namespace) -> float:
    global CONNECTION_AVAILABLE_AT
    now = time.perf_counter()
    with CONNECTION_GUARD_LOCK:
        cutoff = now - max(1.0, args.connection_burst_window_sec)
        CONNECTION_ERROR_TIMES[:] = [item for item in CONNECTION_ERROR_TIMES if item >= cutoff]
        CONNECTION_ERROR_TIMES.append(now)
        if len(CONNECTION_ERROR_TIMES) >= max(1, args.connection_burst_threshold):
            CONNECTION_AVAILABLE_AT = max(
                CONNECTION_AVAILABLE_AT,
                now + max(0.0, args.global_connection_cooldown_sec),
            )
            CONNECTION_ERROR_TIMES.clear()
        return CONNECTION_AVAILABLE_AT


def item_key(item: formal.WorkItem) -> tuple[str, str, str]:
    return (item.subset, item.model.alias, str(item.sample["sample_id"]))


def is_transient_error(error: str) -> bool:
    lowered = error.lower()
    needles = [
        "ratelimiterror",
        "rate limit",
        "tpm limit",
        "429",
        "apiconnectionerror",
        "connection error",
        "timeout",
        "timed out",
        "502",
        "503",
        "504",
    ]
    return any(needle in lowered for needle in needles)


def max_attempts_for_error(args: argparse.Namespace, error: str) -> int:
    if is_connection_error(error):
        return max(1, args.connection_max_attempts)
    return max(1, args.queue_max_attempts)


def is_connection_error(error: str) -> bool:
    lowered = error.lower()
    needles = [
        "apiconnectionerror",
        "connection error",
        "connection aborted",
        "remote end closed connection",
        "timeout",
        "timed out",
    ]
    return any(needle in lowered for needle in needles)


def is_rate_limit_error(error: str) -> bool:
    lowered = error.lower()
    return "ratelimiterror" in lowered or "rate limit" in lowered or "tpm limit" in lowered or "429" in lowered


def short_error(error: str) -> str:
    return error.replace("\n", " ")[:180]


def append_with_lock(path: Path, row: dict[str, Any]) -> None:
    lock = FILE_LOCKS[path]
    with lock:
        append_jsonl(path, row)


def write_plan(args: argparse.Namespace, work: list[formal.WorkItem]) -> None:
    path = Path(args.plan_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((item.subset, item.model.alias) for item in work)
    payload = {
        "run_id": args.run_id,
        "queue_mode": "api_key_release_pool",
        "total_pending_api_calls": len(work),
        "subsets": formal.parse_csv(args.subsets),
        "models": args.models,
        "rate_control": {
            "slots_per_key": args.slots_per_key,
            "global_request_delay_sec": args.global_request_delay_sec,
            "per_key_delay_sec": args.per_key_delay_sec,
            "queue_max_attempts": args.queue_max_attempts,
            "connection_max_attempts": args.connection_max_attempts,
            "connection_cooldown_sec": args.connection_cooldown_sec,
            "connection_burst_threshold": args.connection_burst_threshold,
            "connection_burst_window_sec": args.connection_burst_window_sec,
            "global_connection_cooldown_sec": args.global_connection_cooldown_sec,
            "rate_limit_cooldown_sec": args.rate_limit_cooldown_sec,
            "model_rate_limit_cooldown_sec": args.model_rate_limit_cooldown_sec,
            "transient_cooldown_sec": args.transient_cooldown_sec,
            "timeout": args.timeout,
            "sdk_retry": args.retry,
        },
        "pending_by_subset_model": [
            {"subset": subset, "model": model, "pending_api_calls": count}
            for (subset, model), count in sorted(counts.items())
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def print_plan(args: argparse.Namespace, work: list[formal.WorkItem]) -> None:
    counts = Counter((item.subset, item.model.alias) for item in work)
    print("PAC v2.1 queue pending work:")
    for (subset, model), count in sorted(counts.items()):
        print(f"  {subset:<30} {model:<24} {count}")
    print(f"Total pending API calls: {len(work)}")
    print("Queue mode: each API key/slot pulls the next task immediately after it finishes.")


if __name__ == "__main__":
    main()
