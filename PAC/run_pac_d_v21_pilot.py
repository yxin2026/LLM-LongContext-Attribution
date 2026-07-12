from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAC_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pac2_formal as formal
from lmaf.utils.token_count import TokenCounter


VERSION = "PAC-D_v2.1_green_verifier_multihop"
SUBSET = "PAC-D_multihop_false_chain"
DATA_DIR = PAC_ROOT / "data" / "PAC-D_v2_1_hard"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"
SUMMARY_PATH = DATA_DIR / "summary.json"
LENGTH = 32000
CUTOFF = "23:50"
DEFAULT_MODELS = "qwen35_35b_a3b,qwen35_122b_a10b,seed_oss_36b,qwen3_14b_no_thinking,qwen3_14b_thinking"

FILLER = [
    "Audit paragraphs mix old chain evidence, false links, verifier notes, and final-code tickets.",
    "A link is usable only when the matching verifier is green, valid, approved, and before cutoff.",
    "False chains may have valid-looking links but fail the verifier gate or use a wrong epoch.",
    "The final answer requires the reached node, the last verifier signature, and the active ticket.",
    "Nearby node names, batches, epochs, and alias records are deliberately confusable.",
]


def main() -> None:
    args = parse_args()
    if args.force_generate or not SAMPLES_PATH.exists():
        rows = generate_dataset(args.samples_per_condition, args.seed)
        write_jsonl(SAMPLES_PATH, rows)
        write_json(SUMMARY_PATH, summarize_samples(rows))
        print(f"Wrote {len(rows)} PAC-D v2.1 samples to {SAMPLES_PATH}")
    samples = list(read_jsonl(SAMPLES_PATH))
    if args.stop_after_samples is not None:
        samples = samples[: args.stop_after_samples]

    selected_models = formal.resolve_requested_models(args.models)
    work = formal.build_work(args, make_manifest(), samples, selected_models)
    formal.write_plan(args, make_manifest(), [SUBSET], work)
    formal.print_plan(work)
    print(f"PAC-D v2.1 unique samples: {len(samples)}")
    print(f"PAC-D v2.1 estimated calls: {len(work)}")

    if args.dry_run:
        formal.summarize(args)
        return
    if args.summarize_only:
        formal.summarize(args)
        return
    api_keys = formal.resolve_api_keys(args)
    if work and args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if work:
        random.Random(args.shuffle_seed).shuffle(work)
        formal.run_parallel(args, work, api_keys)
    formal.summarize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAC-D v2.1 hard pilot for strong-model multihop false-chain testing.")
    parser.add_argument("--run-id", default="pac_d_v21_pilot")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--samples-per-condition", type=int, default=2)
    parser.add_argument("--seed", type=int, default=91021)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--stop-after-samples", type=int, default=None)
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-in-flight", type=int, default=3)
    parser.add_argument("--request-delay-sec", type=float, default=12.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260707)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "raw" / "pac_d_v21_pilot"))
    parser.add_argument("--report-root", default=str(ROOT / "results" / "reports" / "pac_d_v21_pilot"))
    parser.add_argument("--plan-output", default=str(ROOT / "results" / "logs" / "pac_d_v21_pilot_plan.json"))
    return parser.parse_args()


def generate_dataset(samples_per_condition: int, seed: int) -> list[dict[str, Any]]:
    counter = TokenCounter()
    rows: list[dict[str, Any]] = []
    for hop_count in [4, 5, 6]:
        for false_chain_count in [16, 32]:
            for local_index in range(samples_per_condition):
                sample_index = hop_count * 1000 + false_chain_count * 100 + local_index
                rows.append(
                    generate_sample(
                        hop_count=hop_count,
                        false_chain_count=false_chain_count,
                        seed=seed,
                        sample_index=sample_index,
                        counter=counter,
                    )
                )
    return rows


def generate_sample(
    hop_count: int,
    false_chain_count: int,
    seed: int,
    sample_index: int,
    counter: TokenCounter,
) -> dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + sample_index)
    batch = f"B{40 + sample_index % 70:03d}-night-{1 + sample_index % 7:02d}"
    epoch = f"EPOCH-{rng.randrange(20, 99)}"
    alias = f"D21-ALIAS-{sample_index % 10000:04d}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10, 99)}"
    nodes = [f"D21-NODE-{sample_index % 10000:04d}-{idx:02d}" for idx in range(hop_count + 1)]
    used_values: set[str] = set()
    final_code = value(rng, used_values)
    ticket_code = value(rng, used_values)
    true_links: list[str] = []
    true_verifiers: list[str] = []
    signatures: list[str] = []
    cutoff = to_minutes(CUTOFF)

    for idx in range(hop_count):
        link_id = f"L{sample_index}-{idx:02d}-{rng.randrange(100, 999)}"
        sig = value(rng, used_values)
        signatures.append(sig)
        true_links.append(
            link_line(
                link_id,
                idx + 1,
                before_cutoff(cutoff, idx),
                nodes[idx],
                nodes[idx + 1],
                batch,
                epoch,
                "valid",
                "approved",
                "primary",
            )
        )
        true_verifiers.append(
            verify_line(link_id, before_cutoff(cutoff, idx + 5), "green", sig, "valid", "approved")
        )

    last_signature = signatures[-1]
    final_line = final_line_for(sample_index, before_cutoff(cutoff, hop_count + 1), nodes[-1], batch, epoch, final_code)
    ticket_line = ticket_line_for(sample_index, alias, nodes[-1], batch, epoch, ticket_code)
    answer = f"{final_code}|{last_signature}|{ticket_code}"

    alias_lines = [
        f"M|A={alias}|NODE={nodes[0]}|AS=active|EPOCH={epoch}",
        f"M|A={alias}|NODE={near_node(nodes[0], 7)}|AS=inactive|EPOCH={epoch}",
        f"M|A={alias}-OLD|NODE={nodes[0]}|AS=inactive|EPOCH={epoch}",
    ]
    false_lines, decoy_values = make_false_lines(
        rng, sample_index, nodes, batch, epoch, false_chain_count, used_values, cutoff
    )
    rng.shuffle(false_lines)
    rng.shuffle(alias_lines)
    focus_lines = interleave(true_links, true_verifiers) + [final_line, ticket_line]
    pre_block = join([lines(alias_lines), lines(false_lines[: len(false_lines) // 2])])
    post_block = lines(false_lines[len(false_lines) // 2 :])
    focus_block = lines(focus_lines)
    instruction = (
        "Use only the long context. Resolve the active alias to the start node. Follow exactly the requested "
        "number of handoff steps. A LINK is usable only if it matches the current FROM node, STEP, Q, EPOCH, "
        "T before cutoff, S=valid, REVIEW=approved, CHANNEL=primary, and also has a matching VERIFY line "
        "with the same LID, GATE=green, S=valid, REVIEW=approved, and T before cutoff. Ignore valid-looking "
        "links with red/missing/draft verifiers, wrong epoch, wrong batch, secondary channel, pending review, "
        "after-cutoff time, inactive aliases, and near-node false chains. After the last hop, output "
        "FINAL_V|LAST_VERIFY_SIG|TICKET_V for the reached node."
    )
    question = (
        f"[Question]\nStarting from alias A={alias}, use Q={batch}, {epoch}, cutoff T<{CUTOFF}, "
        f"and follow exactly {hop_count} verified green handoff steps. What is FINAL_V|LAST_VERIFY_SIG|TICKET_V "
        "at the reached node? Output only the three values separated by |.\n[Answer]"
    )
    prompt = compose_prompt(LENGTH, 50, instruction, pre_block, focus_block, post_block, question, counter, seed + sample_index, (answer, final_code, ticket_code, last_signature))
    return {
        "experiment": "pac2",
        "formal_subset": SUBSET,
        "subtask": "D_v2_1_green_verifier_multihop",
        "pac2_axis": "multihop_false_chain_v2_1",
        "difficulty_version": VERSION,
        "pac_version": "PAC v2.1",
        "sample_id": f"pac_d_v21_H{hop_count}_F{false_chain_count:02d}_{sample_index % 100:04d}",
        "length_tokens_target": LENGTH,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": 50,
        "position_percent_actual": position_percent(prompt, focus_lines[0], counter),
        "hop_count": hop_count,
        "false_chain_count": false_chain_count,
        "prompt": prompt,
        "answer": answer,
        "expected_answers": [final_code, last_signature, ticket_code],
        "final_code": final_code,
        "last_verify_signature": last_signature,
        "ticket_code": ticket_code,
        "start_alias": alias,
        "expected_path": nodes,
        "target_qualifier": batch,
        "epoch": epoch,
        "decoy_values": decoy_values,
        "distractor_answers": decoy_values,
        "models_to_run": formal.parse_csv(DEFAULT_MODELS),
        "model_scope": "strong_5",
        "metric_hint": "exact_final_code_signature_ticket",
        "error": None,
    }


def make_false_lines(
    rng: random.Random,
    sample_index: int,
    nodes: list[str],
    batch: str,
    epoch: str,
    false_chain_count: int,
    used_values: set[str],
    cutoff: int,
) -> tuple[list[str], list[str]]:
    out: list[str] = []
    decoys: list[str] = []
    hop_count = len(nodes) - 1
    for idx in range(false_chain_count):
        step = 1 + (idx % hop_count)
        from_node = nodes[step - 1]
        to_node = near_node(nodes[step], idx + 1)
        link_id = f"FL{sample_index}-{idx:02d}-{rng.randrange(100, 999)}"
        mode = idx % 9
        q = batch if mode != 2 else near_batch(batch, idx)
        ep = epoch if mode != 3 else f"{epoch}-ALT"
        status = "valid" if mode != 4 else "draft"
        review = "approved" if mode != 5 else "pending"
        channel = "primary" if mode != 6 else "secondary"
        t = after_cutoff(cutoff, idx) if mode == 7 else before_cutoff(cutoff, idx + 11)
        out.append(link_line(link_id, step, t, from_node, to_node, q, ep, status, review, channel))
        sig = value(rng, used_values)
        decoys.append(sig)
        gate = "green"
        vstatus = "valid"
        vrevew = "approved"
        vt = before_cutoff(cutoff, idx + 13)
        if mode == 0:
            gate = "red"
        elif mode == 1:
            vstatus = "draft"
        elif mode == 8:
            vt = after_cutoff(cutoff, idx)
        out.append(verify_line(link_id, vt, gate, sig, vstatus, vrevew))
        false_final = value(rng, used_values)
        false_ticket = value(rng, used_values)
        decoys.extend([false_final, false_ticket])
        out.append(final_line_for(sample_index + idx + 10000, before_cutoff(cutoff, idx), to_node, batch, epoch, false_final))
        out.append(ticket_line_for(sample_index + idx + 10000, f"FALSE-A-{idx}", to_node, batch, epoch, false_ticket))
    return out, decoys


def compose_prompt(
    length: int,
    position: int,
    instruction: str,
    pre_block: str,
    focus_block: str,
    post_block: str,
    question: str,
    counter: TokenCounter,
    seed: int,
    forbidden: tuple[str, ...],
) -> str:
    prefix = join(["[Task]", instruction, "[Long Context]", pre_block])
    desired_start = int(length * position / 100)
    before_budget = max(0, desired_start - counter.count(prefix) - 4)
    before = join([prefix, filler(before_budget, seed, counter, forbidden)])
    used = counter.count(join([before, focus_block, post_block, question]))
    tail_budget = max(0, length - used - 4)
    return join([before, focus_block, post_block, filler(tail_budget, seed + 991, counter, forbidden), question])


def filler(target_tokens: int, seed: int, counter: TokenCounter, forbidden: tuple[str, ...]) -> str:
    if target_tokens <= 0:
        return ""
    choices = [item for item in FILLER if not any(term and term in item for term in forbidden)] or FILLER

    def build(n: int) -> str:
        return "\n".join(f"BG{idx + 1:04d}: {choices[(seed + idx) % len(choices)]}" for idx in range(n))

    lo, hi = 0, max(1, target_tokens // 10)
    while counter.count(build(hi)) < target_tokens and hi < max(100, target_tokens * 2):
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(build(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return build(lo)


def link_line(lid: str, step: int, t: str, src: str, dst: str, q: str, epoch: str, status: str, review: str, channel: str) -> str:
    return f"LINK|LID={lid}|STEP={step}|T={t}|FROM={src}|TO={dst}|Q={q}|EPOCH={epoch}|S={status}|REVIEW={review}|CHANNEL={channel}"


def verify_line(lid: str, t: str, gate: str, sig: str, status: str, review: str) -> str:
    return f"VERIFY|LID={lid}|T={t}|GATE={gate}|SIG={sig}|S={status}|REVIEW={review}"


def final_line_for(idx: int, t: str, node: str, batch: str, epoch: str, code: str) -> str:
    return f"FINAL|ID=FIN{idx}|T={t}|NODE={node}|Q={batch}|EPOCH={epoch}|V={code}|S=valid|REVIEW=approved|CHANNEL=primary"


def ticket_line_for(idx: int, alias: str, node: str, batch: str, epoch: str, code: str) -> str:
    return f"TICKET|ID=TIC{idx}|A={alias}|NODE={node}|Q={batch}|EPOCH={epoch}|V={code}|S=valid|REVIEW=approved"


def interleave(left: list[str], right: list[str]) -> list[str]:
    out: list[str] = []
    for a, b in zip(left, right):
        out.extend([a, b])
    return out


def value(rng: random.Random, used: set[str]) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(1000):
        item = "-".join("".join(rng.choice(alphabet) for _ in range(3)) for _ in range(3))
        if item not in used:
            used.add(item)
            return item
    raise RuntimeError("could not create unique value")


def near_node(node: str, offset: int) -> str:
    return f"{node}-ALT{offset:02d}"


def near_batch(batch: str, offset: int) -> str:
    match = re.match(r"B(\d+)-([^-]+)-(\d+)", batch)
    if not match:
        return f"{batch}-ALT{offset:02d}"
    return f"B{int(match.group(1)) + 1:03d}-{match.group(2)}-{int(match.group(3)):02d}"


def to_minutes(text: str) -> int:
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute)


def before_cutoff(cutoff: int, offset: int) -> str:
    minute = max(0, cutoff - 1 - (offset % 19))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def after_cutoff(cutoff: int, offset: int) -> str:
    minute = min(23 * 60 + 59, cutoff + 1 + (offset % 8))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def lines(items: list[str]) -> str:
    return "\n".join(items)


def join(items: list[str]) -> str:
    return "\n\n".join(item for item in items if item)


def position_percent(prompt: str, needle: str, counter: TokenCounter) -> float:
    return round(counter.count(prompt[: prompt.index(needle)]) * 100 / max(1, counter.count(prompt)), 2)


def make_manifest() -> dict[str, Any]:
    return {"subsets": {SUBSET: {"subset": SUBSET}}}


def summarize_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "n_samples": len(rows),
        "length_tokens_min": min(row["length_tokens_actual"] for row in rows),
        "length_tokens_max": max(row["length_tokens_actual"] for row in rows),
        "length_tokens_mean": round(mean(row["length_tokens_actual"] for row in rows), 2),
        "position_actual_min": min(row["position_percent_actual"] for row in rows),
        "position_actual_max": max(row["position_percent_actual"] for row in rows),
        "conditions": sorted({f"H{row['hop_count']}_F{row['false_chain_count']}" for row in rows}),
        "models_to_run": formal.parse_csv(DEFAULT_MODELS),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
