from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
RAW_FILE = "improve_raw_results.jsonl"

FILE_LOCK = threading.Lock()
COUNTER_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    guard_output_dir(out_dir)
    plan_path = Path(args.plan) if args.plan else out_dir / "improve_plan.jsonl"
    plan_rows = load_plan(plan_path)
    plan_rows = filter_plan(plan_rows, args)
    raw_path = out_dir / RAW_FILE
    completed = completed_keys(raw_path)
    work = [row for row in plan_rows if task_key(row) not in completed]
    if args.stop_after is not None:
        work = work[: args.stop_after]

    write_queue_plan(out_dir, args, plan_path, plan_rows, work, completed)
    print_plan(work, plan_rows, completed)

    if args.dry_run:
        summarize(out_dir, args)
        return
    if args.summarize_only:
        summarize(out_dir, args)
        return
    api_keys = resolve_api_keys(args)
    if work and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if not work:
        print("No pending improve work.")
        summarize(out_dir, args)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    run_queue(args, out_dir, raw_path, work, api_keys)
    summarize(out_dir, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PAC v2.1 Improve intervention prompts with an API-key release queue and resume."
    )
    parser.add_argument("--run-id", default="pac_v21_improve_540")
    parser.add_argument("--plan", default=None, help="Defaults to results/improve/<run-id>/improve_plan.jsonl.")
    parser.add_argument("--out-dir", default=None, help="Defaults to results/improve/<run-id>.")
    parser.add_argument("--models", default="auto", help="Comma aliases or auto.")
    parser.add_argument("--interventions", default="auto", help="Comma interventions or auto.")
    parser.add_argument("--subsets", default="auto", help="Comma A,B,C or auto.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-keys", default=None)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--retry", type=int, default=1, help="Per queue attempt API retries.")
    parser.add_argument("--queue-max-attempts", type=int, default=4)
    parser.add_argument("--rate-limit-cooldown-sec", type=float, default=180.0)
    parser.add_argument("--transient-cooldown-sec", type=float, default=45.0)
    parser.add_argument("--per-key-delay-sec", type=float, default=20.0)
    parser.add_argument("--slots-per-key", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override plan max_tokens_recommended.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--disable-extra-body", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=20260708)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser.parse_args()


def guard_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    improve_root = (ROOT / "results" / "improve").resolve()
    if improve_root not in [resolved, *resolved.parents]:
        raise SystemExit(f"Refusing to write outside results/improve: {resolved}")


def load_plan(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Improve plan not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ["sample_id", "intervention", "model_alias", "pac_subset", "intervention_prompt"]:
                if key not in row:
                    raise SystemExit(f"Missing {key} in {path}:{line_no}")
            rows.append(row)
    return rows


def filter_plan(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    model_filter = None if args.models.strip().lower() == "auto" else set(parse_csv(args.models))
    intervention_filter = None if args.interventions.strip().lower() == "auto" else set(parse_csv(args.interventions))
    subset_filter = None if args.subsets.strip().lower() == "auto" else set(parse_csv(args.subsets))
    out = []
    for row in rows:
        if model_filter is not None and row["model_alias"] not in model_filter:
            continue
        if intervention_filter is not None and row["intervention"] not in intervention_filter:
            continue
        if subset_filter is not None and row["pac_subset"] not in subset_filter:
            continue
        out.append(row)
    return out


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_api_keys(args: argparse.Namespace) -> list[str]:
    if args.api_keys:
        return [key.strip() for key in args.api_keys.split(",") if key.strip()]
    if args.api_key:
        return [args.api_key.strip()]
    keys = os.getenv("SILICONFLOW_API_KEYS")
    if keys:
        return [key.strip() for key in keys.split(",") if key.strip()]
    key = os.getenv("SILICONFLOW_API_KEY")
    return [key] if key else []


def task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["sample_id"]), str(row["intervention"]), str(row["model_alias"]))


def completed_keys(raw_path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not raw_path.exists():
        return completed
    for row in read_jsonl(raw_path):
        if row.get("error") in (None, ""):
            completed.add(task_key(row))
    return completed


def run_queue(args: argparse.Namespace, out_dir: Path, raw_path: Path, work: list[dict[str, Any]], api_keys: list[str]) -> None:
    rng = random.Random(args.shuffle_seed)
    work = list(work)
    rng.shuffle(work)
    task_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    for row in work:
        task_queue.put(row)

    key_slots = build_key_slots(api_keys, args.slots_per_key)
    attempts: dict[tuple[str, str, str], int] = defaultdict(int)
    stats = {
        "total": len(work),
        "done": 0,
        "ok": 0,
        "errors": 0,
        "requeued": 0,
        "started_at": time.perf_counter(),
    }
    print(
        f"PAC improve queue started: tasks={len(work)}, workers={len(key_slots)}, "
        f"slots_per_key={args.slots_per_key}, per_key_delay_sec={args.per_key_delay_sec}"
    )
    threads = []
    for worker_index, (api_key, key_slot) in enumerate(key_slots, start=1):
        thread = threading.Thread(
            target=worker_loop,
            args=(args, worker_index, api_key, key_slot, task_queue, attempts, stats, raw_path),
            daemon=True,
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print(
        f"PAC improve queue finished: ok={stats['ok']} final_errors={stats['errors']} "
        f"requeued={stats['requeued']}"
    )


def build_key_slots(api_keys: list[str], slots_per_key: int) -> list[tuple[str, str]]:
    if slots_per_key <= 0:
        raise SystemExit("--slots-per-key must be positive")
    out = []
    for key_index, api_key in enumerate(api_keys):
        for _ in range(slots_per_key):
            out.append((api_key, f"key_{key_index}"))
    return out


def worker_loop(
    args: argparse.Namespace,
    worker_index: int,
    api_key: str,
    key_slot: str,
    task_queue: queue.Queue[dict[str, Any]],
    attempts: dict[tuple[str, str, str], int],
    stats: dict[str, Any],
    raw_path: Path,
) -> None:
    last_request_at = 0.0
    while True:
        try:
            row = task_queue.get(timeout=2.0)
        except queue.Empty:
            return
        wait_for = last_request_at + args.per_key_delay_sec - time.perf_counter()
        if wait_for > 0:
            time.sleep(wait_for)
        last_request_at = time.perf_counter()

        key = task_key(row)
        attempts[key] += 1
        print(
            f"[START] {row['sample_id']} {row['model_alias']} {row['intervention']} {key_slot} "
            f"attempt={attempts[key]}",
            flush=True,
        )
        result_row = run_one(args, row, api_key, key_slot, attempts[key])
        error_type = str(result_row.get("error_type") or "")
        error_text = str(result_row.get("error") or "")
        transient = error_type in {"rate_limited", "timeout", "connection_error", "api_error"} and is_transient_error(error_text)
        if transient and attempts[key] < args.queue_max_attempts:
            cooldown = args.rate_limit_cooldown_sec if error_type == "rate_limited" else args.transient_cooldown_sec
            with COUNTER_LOCK:
                stats["requeued"] += 1
                print(
                    f"[REQUEUE] {row['sample_id']} {row['model_alias']} {row['intervention']} "
                    f"{key_slot} {error_type} after={cooldown:.0f}s {short_error(error_text)}",
                    flush=True,
                )
            time.sleep(cooldown)
            task_queue.put(row)
            task_queue.task_done()
            continue

        append_jsonl_locked(raw_path, result_row)
        ok = result_row.get("error") in (None, "")
        log_result(result_row, ok)
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
                    f"requeued={stats['requeued']} rate={rate:.3f}/s eta={eta/60:.1f}m",
                    flush=True,
                )
        task_queue.task_done()


def run_one(args: argparse.Namespace, plan: dict[str, Any], api_key: str, key_slot: str, attempt_count: int) -> dict[str, Any]:
    max_tokens = args.max_tokens if args.max_tokens is not None else int(plan.get("max_tokens_recommended") or 512)
    started = time.perf_counter()
    response_text = ""
    error = ""
    error_type = "none"
    extra_body_used = False
    extra_body_fallback = False
    completion_tokens = None
    prompt_tokens = None

    for retry_index in range(max(1, args.retry + 1)):
        try:
            response_text, usage, extra_body_used, extra_body_fallback = call_chat_completion(
                args=args,
                plan=plan,
                api_key=api_key,
                key_slot=key_slot,
                max_tokens=max_tokens,
                disable_extra_body=args.disable_extra_body,
            )
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            error = ""
            error_type = "none"
            break
        except Exception as exc:
            error = repr(exc)
            error_type = classify_error(error)
            if retry_index < args.retry and error_type in {"rate_limited", "timeout", "connection_error", "api_error"}:
                time.sleep(min(8, 2 * (retry_index + 1)))
                continue
            break

    latency = time.perf_counter() - started
    if error:
        scored = score_error(plan, error_type)
    else:
        scored = score_prediction(plan, response_text)
        if scored["error_type"] == "empty_output":
            error_type = "empty_output"
        elif scored["error_type"] == "parse_error":
            error_type = "parse_error"
    result = {
        "sample_id": plan["sample_id"],
        "pac_subset": plan["pac_subset"],
        "formal_subset": plan.get("formal_subset"),
        "intervention": plan["intervention"],
        "model_id": plan["model_id"],
        "model_alias": plan["model_alias"],
        "prompt_version": plan.get("prompt_version"),
        "api_key_slot": key_slot,
        "baseline_answer": plan.get("baseline_answer"),
        "intervention_answer": scored.get("prediction_parsed") or response_text,
        "prediction_raw": response_text,
        "prediction_parsed": scored.get("prediction_parsed"),
        "gold_answer": plan.get("gold_answer"),
        "gold_answer_json": plan.get("gold_answer_json"),
        "correct_baseline": int(plan.get("correct_baseline") or 0),
        "correct_intervention": int(scored.get("correct") or 0),
        "field_accuracy_baseline": float(plan.get("field_accuracy_baseline") or 0.0),
        "field_accuracy_intervention": float(scored.get("field_accuracy") or 0.0),
        "all_correct_baseline": int(plan.get("correct_baseline") or 0),
        "all_correct_intervention": int(scored.get("all_correct") or scored.get("correct") or 0),
        "decoy_capture_baseline": int(plan.get("decoy_capture_baseline") or 0),
        "decoy_capture_intervention": int(scored.get("decoy_capture") or 0),
        "binding_error_baseline": int(plan.get("binding_error_baseline") or 0),
        "binding_error_intervention": int(scored.get("binding_error") or 0),
        "omission_rate_intervention": float(scored.get("omission_rate") or 0.0),
        "baseline_error_type": plan.get("baseline_error_type"),
        "intervention_error_type": scored.get("error_type") if not error else error_type,
        "latency": round(latency, 4),
        "latency_sec": round(latency, 4),
        "error": error,
        "error_type": error_type if error else scored.get("error_type"),
        "error_message": error,
        "attempt_count": attempt_count,
        "queue_retry_count": attempt_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_url": args.base_url,
        "extra_body_used": extra_body_used,
        "extra_body_fallback": extra_body_fallback,
        "timestamp": utc_timestamp(),
        "common_across_available_models": plan.get("common_across_available_models"),
        "position": plan.get("position"),
        "decoy_count": plan.get("decoy_count"),
        "binding_load_K": plan.get("binding_load_K"),
        "query_count_Q": plan.get("query_count_Q"),
        "target_alias": plan.get("target_alias"),
        "target_entity": plan.get("target_entity"),
        "target_attribute": plan.get("target_attribute"),
        "target_qualifier": plan.get("target_qualifier"),
        "profile_id": plan.get("profile_id"),
        "query_aliases": plan.get("query_aliases"),
        "intervention_prompt_sha256": plan.get("intervention_prompt_sha256"),
    }
    result.update(scored)
    return result


def call_chat_completion(
    args: argparse.Namespace,
    plan: dict[str, Any],
    api_key: str,
    key_slot: str,
    max_tokens: int,
    disable_extra_body: bool,
) -> tuple[str, Any, bool, bool]:
    client = get_thread_client(args.base_url, api_key, args.timeout)
    messages = [
        {
            "role": "system",
            "content": "You are a precise evaluator. Follow the requested output format exactly.",
        },
        {"role": "user", "content": plan["intervention_prompt"]},
    ]
    kwargs = {
        "model": plan["model_id"],
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "extra_headers": {"X-Request-ID": f"pac-improve/{plan['intervention']}/{plan['model_alias']}/{plan['sample_id']}"},
    }
    if not disable_extra_body:
        kwargs["extra_body"] = {"enable_thinking": False}
    try:
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        return response.choices[0].message.content or "", usage, not disable_extra_body, False
    except Exception as exc:
        message = repr(exc).lower()
        if not disable_extra_body and ("extra_body" in message or "enable_thinking" in message or "unrecognized" in message):
            kwargs.pop("extra_body", None)
            response = client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            return response.choices[0].message.content or "", usage, False, True
        raise


def get_thread_client(base_url: str, api_key: str, timeout: float):
    cache = getattr(THREAD_LOCAL, "clients", None)
    if cache is None:
        cache = {}
        THREAD_LOCAL.clients = cache
    key = (base_url, api_key, timeout)
    if key not in cache:
        from openai import OpenAI

        cache[key] = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)
    return cache[key]


def score_error(plan: dict[str, Any], error_type: str) -> dict[str, Any]:
    return {
        "prediction_parsed": "",
        "correct": 0,
        "field_accuracy": 0.0,
        "all_correct": 0,
        "decoy_capture": 0,
        "binding_error": 0,
        "omission_rate": 1.0,
        "false_chain_hit": 0,
        "error_type": error_type,
    }


def score_prediction(plan: dict[str, Any], prediction: str) -> dict[str, Any]:
    intervention = str(plan.get("intervention"))
    if not prediction.strip():
        return {
            "prediction_parsed": "",
            "correct": 0,
            "field_accuracy": 0.0,
            "all_correct": 0,
            "decoy_capture": 0,
            "binding_error": 0,
            "omission_rate": 1.0,
            "false_chain_hit": 0,
            "error_type": "empty_output",
        }
    if str(plan.get("pac_subset")) == "C" or intervention in {"binding_table", "json_only_binding"}:
        return score_binding_json(plan, prediction)
    return score_triplet(plan, prediction)


def score_triplet(plan: dict[str, Any], prediction: str) -> dict[str, Any]:
    expected = [str(item) for item in plan.get("gold_expected_answers") or split_answer(plan.get("gold_answer", ""))]
    answer = "|".join(expected)
    parsed = extract_final_answer(prediction)
    parsed_norm = normalize_answer(parsed)
    score_text = parsed if parsed_norm else prediction
    score_norm = normalize_answer(score_text)
    correct = int(normalize_answer(answer) in score_norm)
    hits = sum(1 for item in expected if normalize_answer(item) in score_norm)
    field_accuracy = hits / len(expected) if expected else 0.0
    decoy_hit = find_decoy_hit(plan, score_text)
    if correct:
        error_type = "correct"
    elif decoy_hit:
        error_type = "decoy_value_capture"
    elif hits:
        error_type = "partial_triplet"
    elif not parsed_norm:
        error_type = "empty_output"
    else:
        error_type = "other_error"
    return {
        "prediction_parsed": parsed,
        "correct": correct,
        "field_accuracy": field_accuracy,
        "all_correct": correct,
        "decoy_capture": int(bool(decoy_hit)),
        "binding_error": int(error_type in {"decoy_value_capture", "partial_triplet"}),
        "omission_rate": 0.0 if parsed_norm else 1.0,
        "false_chain_hit": 0,
        "error_type": error_type,
        "decoy_hit_value": decoy_hit or "",
    }


def score_binding_json(plan: dict[str, Any], prediction: str) -> dict[str, Any]:
    gold = plan.get("gold_answer_json") or {}
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except Exception:
            gold = {}
    parsed_obj, parsed_text = extract_answer_json(prediction)
    if parsed_obj is None:
        return {
            "prediction_parsed": parsed_text,
            "correct": 0,
            "field_accuracy": 0.0,
            "all_correct": 0,
            "decoy_capture": int(bool(find_decoy_hit(plan, prediction))),
            "binding_error": int(bool(find_decoy_hit(plan, prediction))),
            "omission_rate": 1.0,
            "false_chain_hit": 0,
            "error_type": "parse_error",
        }
    total = len(gold)
    hits = 0
    omissions = 0
    wrong_bound = 0
    decoy_hit = False
    decoy_values = set(str(v) for v in plan.get("decoy_values") or [])
    expected_values = set(str(v) for v in gold.values())
    for alias, gold_value in gold.items():
        pred_value = parsed_obj.get(alias)
        if pred_value is None:
            omissions += 1
            continue
        pred_value = str(pred_value).strip()
        if normalize_answer(pred_value) == normalize_answer(str(gold_value)):
            hits += 1
        elif pred_value in decoy_values or normalize_answer(pred_value) in {normalize_answer(v) for v in decoy_values}:
            decoy_hit = True
            wrong_bound += 1
        elif pred_value in expected_values or normalize_answer(pred_value) in {normalize_answer(v) for v in expected_values}:
            wrong_bound += 1
        else:
            wrong_bound += 1
    field_accuracy = hits / total if total else 0.0
    all_correct = int(total > 0 and hits == total)
    if all_correct:
        error_type = "correct"
    elif decoy_hit:
        error_type = "decoy_value_capture"
    elif omissions == total:
        error_type = "omission"
    elif wrong_bound:
        error_type = "binding_error"
    else:
        error_type = "other_error"
    return {
        "prediction_parsed": json.dumps(parsed_obj, ensure_ascii=False, sort_keys=True),
        "correct": all_correct,
        "field_accuracy": field_accuracy,
        "all_correct": all_correct,
        "decoy_capture": int(decoy_hit),
        "binding_error": int(wrong_bound > 0),
        "omission_rate": omissions / total if total else 0.0,
        "false_chain_hit": 0,
        "error_type": error_type,
    }


def extract_final_answer(prediction: str) -> str:
    for line in reversed(prediction.splitlines()):
        if "FINAL_ANSWER" in line:
            return line.split(":", 1)[-1].strip()
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return prediction.strip()


def extract_answer_json(prediction: str) -> tuple[dict[str, Any] | None, str]:
    target = ""
    for line in reversed(prediction.splitlines()):
        if "ANSWER_JSON" in line:
            target = line.split(":", 1)[-1].strip()
            break
    if not target:
        match = re.search(r"\{.*\}", prediction, flags=re.S)
        target = match.group(0).strip() if match else ""
    if not target:
        return None, ""
    try:
        parsed = json.loads(target)
        if isinstance(parsed, dict):
            return parsed, target
    except Exception:
        pass
    return None, target


def find_decoy_hit(plan: dict[str, Any], prediction: str) -> str | None:
    pred_norm = normalize_answer(prediction)
    for value in plan.get("decoy_values") or []:
        value = str(value)
        if value and normalize_answer(value) in pred_norm:
            return value
    return None


def split_answer(answer: str) -> list[str]:
    return [part.strip() for part in str(answer).split("|") if part.strip()]


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().upper())


def classify_error(error: str) -> str:
    lowered = error.lower()
    if any(token in lowered for token in ["429", "ratelimit", "rate limit", "tpm limit", "rpm limit", "insufficient quota", "rate_limit"]):
        return "rate_limited"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "connection" in lowered or "apiconnectionerror" in lowered or "network" in lowered:
        return "connection_error"
    return "api_error"


def is_transient_error(error: str) -> bool:
    lowered = error.lower()
    return any(
        token in lowered
        for token in [
            "429",
            "ratelimit",
            "rate limit",
            "tpm limit",
            "rpm limit",
            "timeout",
            "timed out",
            "connection",
            "apiconnectionerror",
            "502",
            "503",
            "504",
            "server error",
        ]
    )


def log_result(row: dict[str, Any], ok: bool) -> None:
    if ok:
        print(
            f"[OK] {row['sample_id']} {row['model_alias']} {row['intervention']} "
            f"score={row.get('correct_intervention')} field={float(row.get('field_accuracy_intervention') or 0):.3f} "
            f"latency={float(row.get('latency_sec') or 0):.1f}s",
            flush=True,
        )
        return
    label = str(row.get("error_type") or "ERROR").upper()
    if label == "RATE_LIMITED":
        label = "RATE_LIMIT"
    print(
        f"[{label}] {row['sample_id']} {row['model_alias']} {row['intervention']} "
        f"{row.get('api_key_slot')} {short_error(str(row.get('error') or ''))}",
        flush=True,
    )


def append_jsonl_locked(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FILE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = task_key(row)
        rank = (1 if row.get("error") in (None, "") else 0, idx)
        old = best.get(key)
        if old is None or rank >= (1 if old[1].get("error") in (None, "") else 0, old[0]):
            best[key] = (idx, row)
    return [row for _idx, row in best.values()]


def summarize(out_dir: Path, args: argparse.Namespace) -> None:
    raw_path = out_dir / RAW_FILE
    rows = dedupe_rows(read_jsonl(raw_path))
    write_jsonl(out_dir / "improve_raw_results.deduped.jsonl", rows)
    summary_rows = summarize_by_model(rows, args.bootstrap_iters)
    paired_rows = paired_delta_rows(rows)
    error_rows = api_error_rows(rows)
    transition_rows = transition_summary(rows)
    write_csv(out_dir / "improve_summary_by_model.csv", summary_rows)
    write_csv(out_dir / "improve_paired_delta.csv", paired_rows)
    write_csv(out_dir / "api_errors.csv", error_rows)
    write_csv(out_dir / "improve_error_transitions.csv", transition_rows)
    write_figures(out_dir, summary_rows)
    write_report_text(out_dir, summary_rows, transition_rows)
    print(f"Wrote improve summary to {out_dir}")


def summarize_by_model(rows: list[dict[str, Any]], bootstrap_iters: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("model_alias")), str(row.get("intervention")), str(row.get("pac_subset")))].append(row)
    out = []
    for (model, intervention, subset), items in sorted(buckets.items()):
        n = len(items)
        valid = [row for row in items if row.get("error") in (None, "")]
        baseline_correct = [int(row.get("correct_baseline") or 0) for row in items]
        intervention_correct = [int(row.get("correct_intervention") or 0) for row in items]
        deltas = [i - b for b, i in zip(baseline_correct, intervention_correct)]
        baseline_accuracy = mean_safe(baseline_correct)
        intervention_accuracy = mean_safe(intervention_correct)
        absolute_gain = intervention_accuracy - baseline_accuracy
        ci_low, ci_high = bootstrap_ci(deltas, bootstrap_iters, seed=stable_seed(model, intervention, subset))
        base_decoy = mean_safe([int(row.get("decoy_capture_baseline") or 0) for row in items])
        int_decoy = mean_safe([int(row.get("decoy_capture_intervention") or 0) for row in items])
        base_binding = mean_safe([int(row.get("binding_error_baseline") or 0) for row in items])
        int_binding = mean_safe([int(row.get("binding_error_intervention") or 0) for row in items])
        out.append(
            {
                "model_alias": model,
                "intervention": intervention,
                "pac_subset": subset,
                "n": n,
                "n_valid_intervention": len(valid),
                "baseline_accuracy": round(baseline_accuracy, 4),
                "intervention_accuracy": round(intervention_accuracy, 4),
                "absolute_gain": round(absolute_gain, 4),
                "absolute_gain_ci95_low": round(ci_low, 4),
                "absolute_gain_ci95_high": round(ci_high, 4),
                "relative_gain": round(absolute_gain / baseline_accuracy, 4) if baseline_accuracy else "",
                "baseline_decoy_capture_rate": round(base_decoy, 4),
                "intervention_decoy_capture_rate": round(int_decoy, 4),
                "decoy_capture_reduction": round(base_decoy - int_decoy, 4),
                "baseline_binding_error_rate": round(base_binding, 4),
                "intervention_binding_error_rate": round(int_binding, 4),
                "binding_error_reduction": round(base_binding - int_binding, 4),
                "api_error_rate": round(sum(1 for row in items if row.get("error") not in (None, "")) / n, 4) if n else "",
                "mean_latency_sec": round(mean_safe([float(row.get("latency_sec") or 0) for row in valid]), 3) if valid else "",
            }
        )
    return out


def paired_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda r: (r.get("model_alias"), r.get("pac_subset"), r.get("intervention"), r.get("sample_id"))):
        baseline_correct = int(row.get("correct_baseline") or 0)
        intervention_correct = int(row.get("correct_intervention") or 0)
        out.append(
            {
                "model_alias": row.get("model_alias"),
                "pac_subset": row.get("pac_subset"),
                "intervention": row.get("intervention"),
                "sample_id": row.get("sample_id"),
                "baseline_correct": baseline_correct,
                "intervention_correct": intervention_correct,
                "delta_correct": intervention_correct - baseline_correct,
                "baseline_error_type": row.get("baseline_error_type"),
                "intervention_error_type": row.get("intervention_error_type") or row.get("error_type"),
            }
        )
    return out


def api_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("error") in (None, ""):
            continue
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "condition": row.get("intervention"),
                "pac_subset": row.get("pac_subset"),
                "model_alias": row.get("model_alias"),
                "api_model": row.get("model_id"),
                "api_key_slot": row.get("api_key_slot"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message") or row.get("error"),
                "attempt_count": row.get("attempt_count"),
                "latency_sec": row.get("latency_sec"),
            }
        )
    return out


def transition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], int] = Counter()
    for row in rows:
        transition = classify_transition(row)
        buckets[(str(row.get("model_alias")), str(row.get("intervention")), str(row.get("pac_subset")), transition)] += 1
    return [
        {
            "model_alias": model,
            "intervention": intervention,
            "pac_subset": subset,
            "transition": transition,
            "count": count,
        }
        for (model, intervention, subset, transition), count in sorted(buckets.items())
    ]


def classify_transition(row: dict[str, Any]) -> str:
    base_correct = int(row.get("correct_baseline") or 0)
    int_correct = int(row.get("correct_intervention") or 0)
    base_error = str(row.get("baseline_error_type") or "")
    if not base_correct and int_correct:
        if base_error == "decoy_value_capture":
            return "decoy_capture_to_correct"
        if base_error in {"partial_triplet", "binding_error", "near_miss_value"}:
            return "binding_error_to_correct"
        return "baseline_wrong_to_intervention_correct"
    if base_correct and not int_correct:
        return "baseline_correct_to_intervention_wrong"
    if not base_correct and not int_correct:
        return "wrong_to_still_wrong"
    return "correct_to_still_correct"


def write_figures(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = summary_rows
    plot_gain_bar(
        out_dir / "segment_anchor_gain_by_model.png",
        [row for row in rows if row["intervention"] == "segment_anchor"],
        "Segment Anchor Accuracy Gain",
        "absolute_gain",
        plt,
    )
    plot_gain_bar(
        out_dir / "evidence_first_decoy_reduction.png",
        [row for row in rows if row["intervention"] == "evidence_first"],
        "Evidence-first Decoy Capture Reduction",
        "decoy_capture_reduction",
        plt,
    )
    plot_gain_bar(
        out_dir / "binding_table_error_reduction.png",
        [row for row in rows if row["intervention"] == "binding_table"],
        "Binding Table Binding Error Reduction",
        "binding_error_reduction",
        plt,
    )
    plot_heatmap(out_dir / "improve_overall_gain_heatmap.png", rows, plt)


def plot_gain_bar(path: Path, rows: list[dict[str, Any]], title: str, value_key: str, plt: Any) -> None:
    if not rows:
        return
    labels = [str(row["model_alias"]) for row in rows]
    values = [float(row.get(value_key) or 0) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#4C78A8")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(value_key)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(path: Path, rows: list[dict[str, Any]], plt: Any) -> None:
    models = sorted({str(row["model_alias"]) for row in rows})
    interventions = sorted({str(row["intervention"]) for row in rows})
    data = [[0.0 for _ in interventions] for _ in models]
    for i, model in enumerate(models):
        for j, intervention in enumerate(interventions):
            vals = [float(row.get("absolute_gain") or 0) for row in rows if row["model_alias"] == model and row["intervention"] == intervention]
            data[i][j] = mean_safe(vals)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(interventions)), interventions, rotation=25, ha="right")
    ax.set_yticks(range(len(models)), models)
    ax.set_title("Overall Improve Accuracy Gain")
    for i, _model in enumerate(models):
        for j, _intervention in enumerate(interventions):
            ax.text(j, i, f"{data[i][j]:.3f}", ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report_text(out_dir: Path, summary_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PAC Improve Results Draft",
        "",
        "Baseline was not rerun. Each intervention result is paired to an existing PAC v2.1 baseline row by sample_id, model_alias, and intervention subset.",
        "",
        "The three interventions map to behavior-level failure modes: Segment Anchor targets position/structure salience, Evidence-first targets high-similarity decoy capture, and Binding Table targets entity-attribute binding instability. These results should not be described as proving low-level RoPE, attention, or MoE routing mechanisms.",
        "",
        "## Summary",
        "",
    ]
    if not summary_rows:
        lines.append("No completed intervention rows yet.")
    else:
        for row in summary_rows:
            lines.append(
                f"- {row['model_alias']} / {row['intervention']} / PAC-{row['pac_subset']}: "
                f"baseline={row['baseline_accuracy']}, intervention={row['intervention_accuracy']}, "
                f"gain={row['absolute_gain']} (95% CI {row['absolute_gain_ci95_low']} to {row['absolute_gain_ci95_high']}), "
                f"decoy reduction={row['decoy_capture_reduction']}, binding reduction={row['binding_error_reduction']}."
            )
    out_dir.joinpath("improve_report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_queue_plan(out_dir: Path, args: argparse.Namespace, plan_path: Path, plan_rows: list[dict[str, Any]], work: list[dict[str, Any]], completed: set[tuple[str, str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["pac_subset"], row["intervention"], row["model_alias"]) for row in work)
    payload = {
        "run_id": args.run_id,
        "created_at": utc_timestamp(),
        "plan_path": str(plan_path),
        "raw_output": str(out_dir / RAW_FILE),
        "queue_mode": "api_key_release_pool",
        "total_plan_rows_after_filter": len(plan_rows),
        "completed_success_rows": len(completed),
        "pending_api_calls": len(work),
        "rate_control": {
            "slots_per_key": args.slots_per_key,
            "per_key_delay_sec": args.per_key_delay_sec,
            "queue_max_attempts": args.queue_max_attempts,
            "rate_limit_cooldown_sec": args.rate_limit_cooldown_sec,
            "transient_cooldown_sec": args.transient_cooldown_sec,
            "timeout": args.timeout,
            "retry": args.retry,
        },
        "pending_by_subset_intervention_model": [
            {"pac_subset": subset, "intervention": intervention, "model_alias": model, "pending": count}
            for (subset, intervention, model), count in sorted(counts.items())
        ],
    }
    (out_dir / "improve_queue_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_plan(work: list[dict[str, Any]], plan_rows: list[dict[str, Any]], completed: set[tuple[str, str, str]]) -> None:
    counts = Counter((row["pac_subset"], row["intervention"], row["model_alias"]) for row in work)
    print("PAC improve pending work:")
    for (subset, intervention, model), count in sorted(counts.items()):
        print(f"  PAC-{subset:<2} {intervention:<16} {model:<20} {count}")
    print(f"Total plan rows after filter: {len(plan_rows)}")
    print(f"Completed success rows: {len(completed)}")
    print(f"Pending API calls: {len(work)}")
    print("Queue mode: each API key/slot pulls the next task immediately after it finishes.")


def mean_safe(values: list[float | int]) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return mean(vals) if vals else 0.0


def bootstrap_ci(deltas: list[int], n_iters: int, seed: int) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, n_iters)):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(mean_safe(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def stable_seed(*parts: str) -> int:
    text = "|".join(parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 2_000_000_000


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def short_error(error: str) -> str:
    return error.replace("\n", " ")[:180]


if __name__ == "__main__":
    main()
