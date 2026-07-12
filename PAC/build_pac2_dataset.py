from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.pac2 import generate_pac2_b_calibration
from lmaf.utils.token_count import TokenCounter


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
LENGTH = 32000
SEED = 73007
CALIBRATED_DECOY = 64
CUTOFF_TIME = "23:50"

ALL_MODELS = [
    "qwen35_9b",
    "qwen3_8b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "seed_oss_36b",
    "hunyuan_a13b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
]

REPRESENTATIVE_MODELS = [
    "qwen35_9b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "seed_oss_36b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
]

ATTRIBUTES = [
    "night-release-code",
    "night-seal-code",
    "exception-register-code",
    "cold-chain-handoff-code",
    "inbound-review-code",
    "night-review-code",
]

FACILITY_PREFIXES = [
    "Huadong-Yiyao-Lenglian-Zhongxin",
    "Huanan-Yiyao-Lenglian-Zhongxin",
    "Huadong-Yiliao-Lenglian-Zhongxin",
    "Huadong-Yiyao-Lengcang-Zhongxin",
    "Huazhong-Yiyao-Lenglian-Zhongxin",
]

FILLER_SENTENCES = [
    "Audit notes describe storage, transfer, route confirmation, and cold-chain checks.",
    "The archive mixes active aliases, inactive aliases, revised facts, and near-duplicate entities.",
    "Only approved primary-channel valid evidence before cutoff can support the final answer.",
    "Nearby batch IDs, entity numbers, field names, and false links are intentionally confusable.",
    "The requested answer requires preserving bindings across distant evidence lines.",
]


def main() -> None:
    counter = TokenCounter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subsets = {
        "PAC-A_position": build_pac_a(counter),
        "PAC-B_interference": build_pac_b(counter),
        "PAC-C_binding_capacity": build_pac_c(counter),
        "PAC-D_multihop_false_chain": build_pac_d(counter),
    }
    all_rows: list[dict[str, Any]] = []
    for subset, rows in subsets.items():
        subset_dir = DATA_DIR / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(subset_dir / "samples.jsonl", rows)
        write_json(subset_dir / "summary.json", summarize_rows(rows))
        all_rows.extend(rows)
    write_jsonl(DATA_DIR / "all_samples.jsonl", all_rows)
    manifest = build_manifest(subsets)
    write_json(OUT_DIR / "manifest.json", manifest)
    write_readme(manifest)
    print_summary(manifest)


def build_pac_a(counter: TokenCounter) -> list[dict[str, Any]]:
    rows = []
    for position in [10, 25, 50, 75, 90]:
        for local_index in range(15):
            sample_index = position * 100 + local_index
            row = generate_pac2_b_calibration(
                length=LENGTH,
                position=position,
                decoy_count=CALIBRATED_DECOY,
                seed=SEED + 101,
                sample_index=sample_index,
                counter=counter,
            )
            row.update(
                {
                    "sample_id": f"pac2_A_position_v5_pos{position:02d}_decoy64_{local_index:04d}",
                    "formal_subset": "PAC-A_position",
                    "pac2_axis": "position_effect",
                    "position_bin": position,
                    "calibrated_decoy_count": CALIBRATED_DECOY,
                    "models_to_run": ALL_MODELS,
                    "model_scope": "all_9",
                    "formal_note": "v5 multi-document triad task; calibrated decoy=64; position varies.",
                }
            )
            rows.append(row)
    return rows


def build_pac_b(counter: TokenCounter) -> list[dict[str, Any]]:
    rows = []
    for decoy_count in [0, 16, 32, 64, 128, 192]:
        for local_index in range(12):
            sample_index = decoy_count * 100 + local_index
            row = generate_pac2_b_calibration(
                length=LENGTH,
                position=50,
                decoy_count=decoy_count,
                seed=SEED + 202,
                sample_index=sample_index,
                counter=counter,
            )
            row.update(
                {
                    "sample_id": f"pac2_B_interference_v5_decoy{decoy_count:03d}_{local_index:04d}",
                    "formal_subset": "PAC-B_interference",
                    "pac2_axis": "interference_threshold",
                    "position_bin": 50,
                    "calibrated_decoy_count": CALIBRATED_DECOY,
                    "models_to_run": ALL_MODELS,
                    "model_scope": "all_9",
                    "formal_note": "v5 multi-document triad task; position fixed at 50; decoy count varies.",
                }
            )
            rows.append(row)
    return rows


def build_pac_c(counter: TokenCounter) -> list[dict[str, Any]]:
    rows = []
    for binding_k in [16, 32, 64]:
        for query_count in [3, 5, 8]:
            for local_index in range(8):
                rows.append(
                    generate_binding_capacity_sample(
                        length=LENGTH,
                        binding_k=binding_k,
                        query_count=query_count,
                        decoy_count=CALIBRATED_DECOY,
                        seed=SEED + 303,
                        sample_index=binding_k * 1000 + query_count * 100 + local_index,
                        counter=counter,
                    )
                )
    return rows


def build_pac_d(counter: TokenCounter) -> list[dict[str, Any]]:
    rows = []
    for hop_count in [2, 3, 4]:
        for false_chain_count in [4, 8, 16]:
            for local_index in range(8):
                rows.append(
                    generate_multihop_false_chain_sample(
                        length=LENGTH,
                        hop_count=hop_count,
                        false_chain_count=false_chain_count,
                        decoy_count=CALIBRATED_DECOY,
                        seed=SEED + 404,
                        sample_index=hop_count * 1000 + false_chain_count * 100 + local_index,
                        counter=counter,
                    )
                )
    return rows


def generate_binding_capacity_sample(
    length: int,
    binding_k: int,
    query_count: int,
    decoy_count: int,
    seed: int,
    sample_index: int,
    counter: TokenCounter,
) -> dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + sample_index)
    batch = _make_qualifier(sample_index)
    profile_id = f"PROFILE-C-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(100, 999)}"
    field = ATTRIBUTES[0]
    aliases: list[str] = []
    entities: list[str] = []
    values: list[str] = []
    used_values: set[str] = set()
    for idx in range(binding_k):
        entity = f"{FACILITY_PREFIXES[idx % len(FACILITY_PREFIXES)]}-{100 + idx:03d}"
        alias = f"C-ALIAS-{sample_index % 10000:04d}-{idx:03d}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10, 99)}"
        value = _unique_value(rng, used_values)
        aliases.append(alias)
        entities.append(entity)
        values.append(value)

    query_indices = sorted(rng.sample(range(binding_k), query_count))
    query_aliases = [aliases[idx] for idx in query_indices]
    expected_answers = [values[idx] for idx in query_indices]
    answer = "|".join(expected_answers)

    alias_lines = []
    evidence_lines_by_index: list[str] = []
    for idx, (alias, entity, value) in enumerate(zip(aliases, entities, values)):
        alias_lines.append(_alias_line(alias, entity, "active"))
        if idx % 5 == 0:
            alias_lines.append(_alias_line(alias, _near_entity(entity, idx + 3), "inactive"))
        evidence_lines_by_index.append(
            _evidence_line(
                doc_id=f"C{sample_index}-{idx:03d}",
                timestamp=_latest_before_cutoff(_time_to_minutes(CUTOFF_TIME), idx),
                entity=entity,
                attribute=field,
                qualifier=batch,
                value=value,
                status="valid",
                review="approved",
                channel="primary",
                source="capacity-ledger",
            )
        )

    decoys = _binding_decoys(rng, entities, values, batch, field, decoy_count, used_values)
    profile_lines = [
        _profile_line(profile_id, 1, field, "approved"),
        _profile_line(profile_id, 1, ATTRIBUTES[1], "draft"),
        _profile_line(f"{profile_id}-ALT", 1, ATTRIBUTES[2], "approved"),
    ]
    focus_lines = [evidence_lines_by_index[idx] for idx in query_indices]
    non_focus_evidence = [
        line for idx, line in enumerate(evidence_lines_by_index) if idx not in set(query_indices)
    ]
    rng.shuffle(alias_lines)
    rng.shuffle(non_focus_evidence)
    rng.shuffle(decoys)

    pre_block = _join(
        [
            _format_lines(alias_lines[: len(alias_lines) // 2]),
            _format_lines(profile_lines),
            _format_lines(non_focus_evidence[: len(non_focus_evidence) // 2]),
            _format_lines([item["fact"] for item in decoys[: decoy_count // 2]]),
        ]
    )
    post_block = _join(
        [
            _format_lines(non_focus_evidence[len(non_focus_evidence) // 2 :]),
            _format_lines([item["fact"] for item in decoys[decoy_count // 2 :]]),
            _format_lines(alias_lines[len(alias_lines) // 2 :]),
        ]
    )
    focus_block = _format_lines(focus_lines)
    instruction = (
        "Use only the long context. Resolve active aliases, use only the approved profile field, "
        "and select valid approved primary-channel evidence before the cutoff. Preserve alias-to-entity-to-value "
        "binding exactly. Reject inactive aliases, near entities, wrong batches, after-cutoff memos, "
        "pending reviews, secondary channels, and draft profile lines."
    )
    question = (
        f"[Question]\nFor approved PID={profile_id}, batch Q={batch}, cutoff T<{CUTOFF_TIME}, "
        f"return the {field} V for these aliases in exactly this order: {', '.join(query_aliases)}. "
        "Output only the V values separated by |.\n[Answer]"
    )
    prompt = _compose_prompt(length, 50, instruction, pre_block, focus_block, post_block, question, counter, seed + sample_index, (answer, *expected_answers))
    return {
        "experiment": "pac2",
        "formal_subset": "PAC-C_binding_capacity",
        "subtask": "C_binding_capacity",
        "pac2_axis": "binding_capacity",
        "difficulty_version": "pac2_C_binding_capacity_v5",
        "sample_id": f"pac2_C_binding_v5_K{binding_k:02d}_Q{query_count:02d}_{sample_index % 100:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": 50,
        "position_percent_actual": _position_percent(prompt, focus_lines[0], counter),
        "binding_k": binding_k,
        "query_count": query_count,
        "decoy_count": decoy_count,
        "calibrated_decoy_count": CALIBRATED_DECOY,
        "prompt": prompt,
        "answer": answer,
        "expected_answers": expected_answers,
        "query_aliases": query_aliases,
        "target_field": field,
        "target_qualifier": batch,
        "profile_id": profile_id,
        "decoy_values": [item["value"] for item in decoys],
        "distractor_answers": [item["value"] for item in decoys],
        "models_to_run": REPRESENTATIVE_MODELS,
        "model_scope": "representative_6",
        "metric_hint": "exact_ordered_values_and_mean_field_accuracy",
        "error": None,
    }


def generate_multihop_false_chain_sample(
    length: int,
    hop_count: int,
    false_chain_count: int,
    decoy_count: int,
    seed: int,
    sample_index: int,
    counter: TokenCounter,
) -> dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + sample_index)
    batch = _make_qualifier(sample_index)
    alias = f"D-ALIAS-{sample_index % 10000:04d}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10, 99)}"
    nodes = [f"D-NODE-{sample_index % 10000:04d}-{idx:02d}" for idx in range(hop_count + 1)]
    answer = _unique_value(rng, set())
    cutoff_minutes = _time_to_minutes(CUTOFF_TIME)
    alias_lines = [_alias_line(alias, nodes[0], "active"), _alias_line(alias, _near_node(nodes[0], 7), "inactive")]
    true_links = [
        _link_line(
            link_id=f"T{sample_index}-{idx:02d}",
            timestamp=_latest_before_cutoff(cutoff_minutes, idx),
            src=nodes[idx],
            dst=nodes[idx + 1],
            status="valid",
            review="approved",
            channel="primary",
        )
        for idx in range(hop_count)
    ]
    final_fact = _evidence_line(
        doc_id=f"DANS{sample_index}",
        timestamp=_latest_before_cutoff(cutoff_minutes, hop_count + 3),
        entity=nodes[-1],
        attribute="final-release-code",
        qualifier=batch,
        value=answer,
        status="valid",
        review="approved",
        channel="primary",
        source="chain-ledger",
    )
    used_values = {answer}
    false_lines = []
    decoy_values = []
    for i in range(false_chain_count):
        branch_from = nodes[i % hop_count]
        branch_to = _near_node(nodes[min(i % hop_count + 1, hop_count)], i + 1)
        mode = i % 6
        status = "valid" if mode not in (1, 4) else _invalid_status(i)
        review = "approved" if mode != 2 else "pending"
        channel = "primary" if mode != 3 else "secondary"
        timestamp = _after_cutoff_time(cutoff_minutes, i) if mode == 5 else _latest_before_cutoff(cutoff_minutes, i + 5)
        false_lines.append(_link_line(f"F{sample_index}-{i:02d}", timestamp, branch_from, branch_to, status, review, channel))
        false_value = _unique_value(rng, used_values)
        decoy_values.append(false_value)
        false_lines.append(
            _evidence_line(
                doc_id=f"FANS{sample_index}-{i:02d}",
                timestamp=_latest_before_cutoff(cutoff_minutes, i + 7),
                entity=branch_to,
                attribute="final-release-code",
                qualifier=batch,
                value=false_value,
                status="valid",
                review="approved",
                channel="primary",
                source="false-chain-ledger",
            )
        )
    for i in range(decoy_count):
        false_value = _unique_value(rng, used_values)
        decoy_values.append(false_value)
        false_lines.append(
            _evidence_line(
                doc_id=f"DNOISE{sample_index}-{i:03d}",
                timestamp=_latest_before_cutoff(cutoff_minutes, i),
                entity=_near_node(nodes[-1], i + 11),
                attribute="final-release-code",
                qualifier=_near_qualifier(batch, i),
                value=false_value,
                status="valid",
                review="approved",
                channel="primary",
                source="noise-ledger",
            )
        )
    rng.shuffle(false_lines)
    focus_block = _format_lines(true_links + [final_fact])
    pre_block = _join([_format_lines(alias_lines), _format_lines(false_lines[: len(false_lines) // 2])])
    post_block = _format_lines(false_lines[len(false_lines) // 2 :])
    instruction = (
        "Use only the long context. Resolve the active alias to the start node. Follow exactly the requested "
        "number of valid approved primary-channel handoff links before cutoff. Ignore inactive aliases, invalid links, "
        "pending reviews, secondary channels, after-cutoff links, near-node false chains, and wrong-batch final codes. "
        "After the final hop, report the valid approved primary final-release-code for the reached node."
    )
    question = (
        f"[Question]\nStarting from alias A={alias}, follow {hop_count} handoff hops using batch Q={batch} "
        f"and cutoff T<{CUTOFF_TIME}. What is the final-release-code V at the reached node? Output only V.\n[Answer]"
    )
    prompt = _compose_prompt(length, 50, instruction, pre_block, focus_block, post_block, question, counter, seed + sample_index, (answer,))
    return {
        "experiment": "pac2",
        "formal_subset": "PAC-D_multihop_false_chain",
        "subtask": "D_multihop_false_chain",
        "pac2_axis": "multihop_false_chain",
        "difficulty_version": "pac2_D_multihop_false_chain_v5",
        "sample_id": f"pac2_D_multihop_v5_H{hop_count}_F{false_chain_count:02d}_{sample_index % 100:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": 50,
        "position_percent_actual": _position_percent(prompt, true_links[0], counter),
        "hop_count": hop_count,
        "false_chain_count": false_chain_count,
        "decoy_count": decoy_count,
        "calibrated_decoy_count": CALIBRATED_DECOY,
        "prompt": prompt,
        "answer": answer,
        "expected_answers": [answer],
        "start_alias": alias,
        "expected_path": nodes,
        "target_qualifier": batch,
        "decoy_values": decoy_values,
        "distractor_answers": decoy_values,
        "models_to_run": REPRESENTATIVE_MODELS,
        "model_scope": "representative_6",
        "metric_hint": "final_answer_exact_with_false_chain_capture",
        "error": None,
    }


def _binding_decoys(rng: random.Random, entities: list[str], values: list[str], batch: str, field: str, decoy_count: int, used_values: set[str]) -> list[dict[str, str]]:
    decoys = []
    cutoff_minutes = _time_to_minutes(CUTOFF_TIME)
    for i in range(decoy_count):
        entity = entities[i % len(entities)]
        base = values[i % len(values)]
        mode = i % 8
        if mode == 0:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, field, batch, "valid", "approved", "secondary", _latest_before_cutoff(cutoff_minutes, i), "secondary_channel"
        elif mode == 1:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, field, batch, "valid", "pending", "primary", _latest_before_cutoff(cutoff_minutes, i), "pending_review"
        elif mode == 2:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, field, batch, "valid", "approved", "primary", _after_cutoff_time(cutoff_minutes, i), "after_cutoff"
        elif mode == 3:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = _near_entity(entity, i), field, batch, "valid", "approved", "primary", _latest_before_cutoff(cutoff_minutes, i), "near_entity"
        elif mode == 4:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, ATTRIBUTES[(i + 1) % len(ATTRIBUTES)], batch, "valid", "approved", "primary", _latest_before_cutoff(cutoff_minutes, i), "wrong_field"
        elif mode == 5:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, field, _near_qualifier(batch, i), "valid", "approved", "primary", _latest_before_cutoff(cutoff_minutes, i), "wrong_batch"
        elif mode == 6:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = entity, field, batch, _invalid_status(i), "approved", "primary", _latest_before_cutoff(cutoff_minutes, i), "invalid_status"
        else:
            d_entity, d_field, d_batch, status, review, channel, timestamp, reason = _prefix_variant(entity, i), field, batch, "valid", "approved", "primary", _latest_before_cutoff(cutoff_minutes, i), "prefix_variant"
        value = _unique_variant(base, rng, used_values, i)
        used_values.add(value)
        decoys.append(
            {
                "value": value,
                "reason": reason,
                "fact": _evidence_line(f"CDEC{i:04d}", timestamp, d_entity, d_field, d_batch, value, status, review, channel, "capacity-decoy"),
            }
        )
    return decoys


def _compose_prompt(length: int, position: int, instruction: str, pre_block: str, focus_block: str, post_block: str, question: str, counter: TokenCounter, seed: int, forbidden: tuple[str, ...]) -> str:
    prefix = _join(["[Task]", instruction, "[Long Context]", pre_block])
    desired_start = int(length * position / 100)
    filler_before_budget = max(0, desired_start - counter.count(prefix) - 4)
    filler_before = _make_filler(filler_before_budget, seed, counter, forbidden)
    before = _join([prefix, filler_before])
    used = counter.count(_join([before, focus_block, post_block, question]))
    tail_budget = max(0, length - used - 4)
    tail = _make_filler(tail_budget, seed + 991, counter, forbidden)
    return _join([before, focus_block, post_block, tail, question])


def _make_filler(target_tokens: int, seed: int, counter: TokenCounter, forbidden: tuple[str, ...]) -> str:
    if target_tokens <= 0:
        return ""
    sentences = [s for s in FILLER_SENTENCES if not any(term and term in s for term in forbidden)] or FILLER_SENTENCES

    def text_for(n: int) -> str:
        return "\n".join(f"BG{idx + 1:04d}: {sentences[(seed + idx) % len(sentences)]}" for idx in range(n))

    lo, hi = 0, max(1, target_tokens // 10)
    while counter.count(text_for(hi)) < target_tokens and hi < max(100, target_tokens * 2):
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text_for(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text_for(lo)


def _alias_line(alias: str, entity: str, status: str) -> str:
    return f"M|A={alias}|E={entity}|AS={status}"


def _profile_line(profile_id: str, step: int, field: str, status: str) -> str:
    return f"P|PID={profile_id}|STEP={step}|F={field}|PSTATUS={status}"


def _evidence_line(doc_id: str, timestamp: str, entity: str, attribute: str, qualifier: str, value: str, status: str, review: str, channel: str, source: str) -> str:
    return f"DOC={doc_id}|T={timestamp}|E={entity}|F={attribute}|Q={qualifier}|V={value}|S={status}|REVIEW={review}|CHANNEL={channel}|SRC={source}"


def _link_line(link_id: str, timestamp: str, src: str, dst: str, status: str, review: str, channel: str) -> str:
    return f"LINK={link_id}|T={timestamp}|FROM={src}|REL=handoff|TO={dst}|S={status}|REVIEW={review}|CHANNEL={channel}"


def _format_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def _join(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _make_qualifier(sample_index: int) -> str:
    batch = 40 + (sample_index % 80)
    shift = ["night", "morning", "mid"][sample_index % 3]
    round_id = 1 + (sample_index % 7)
    return f"B{batch:03d}-{shift}-{round_id:02d}"


def _near_qualifier(qualifier: str, offset: int) -> str:
    match = re.match(r"B(\d+)-([^-]+)-(\d+)", qualifier)
    if not match:
        return f"{qualifier}-near{offset:02d}"
    batch = int(match.group(1))
    shift = match.group(2)
    round_id = int(match.group(3))
    shifts = ["night", "morning", "mid"]
    mode = offset % 4
    if mode == 0:
        batch += 1
    elif mode == 1:
        batch = max(1, batch - 1)
    elif mode == 2:
        shift = shifts[(shifts.index(shift) + 1) % len(shifts)] if shift in shifts else "night"
    else:
        round_id += 1
    return f"B{batch:03d}-{shift}-{round_id:02d}"


def _time_to_minutes(text: str) -> int:
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute)


def _latest_before_cutoff(cutoff_minutes: int, offset: int) -> str:
    minute = max(0, cutoff_minutes - 1 - (offset % 17))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _after_cutoff_time(cutoff_minutes: int, offset: int) -> str:
    minute = min(23 * 60 + 59, cutoff_minutes + 1 + (offset % 9))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _invalid_status(offset: int) -> str:
    return ["expired", "revoked", "draft"][offset % 3]


def _unique_value(rng: random.Random, used: set[str]) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(1000):
        value = "-".join("".join(rng.choice(alphabet) for _ in range(3)) for _ in range(3))
        if value not in used:
            used.add(value)
            return value
    raise RuntimeError("could not create unique value")


def _unique_variant(answer: str, rng: random.Random, used: set[str], offset: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    chars = list(answer)
    mutable = [idx for idx, ch in enumerate(chars) if ch != "-"]
    for attempt in range(200):
        candidate = chars[:]
        idx = mutable[(offset + attempt) % len(mutable)]
        replacement = rng.choice(alphabet)
        if replacement == candidate[idx]:
            replacement = alphabet[(alphabet.index(replacement) + 1) % len(alphabet)]
        candidate[idx] = replacement
        if attempt % 3 == 2:
            idx2 = mutable[(offset * 3 + attempt) % len(mutable)]
            candidate[idx2] = rng.choice(alphabet)
        value = "".join(candidate)
        if value not in used and value != answer:
            return value
    return _unique_value(rng, used)


def _near_entity(entity: str, offset: int) -> str:
    match = re.search(r"(\d+)$", entity)
    if not match:
        return f"{entity}-near{offset:03d}"
    number = int(match.group(1))
    width = len(match.group(1))
    delta = (offset % 9) - 4 or 5
    return re.sub(r"\d+$", f"{max(1, number + delta):0{width}d}", entity)


def _prefix_variant(entity: str, offset: int) -> str:
    match = re.search(r"(\d+)$", entity)
    suffix = match.group(1) if match else f"{offset:03d}"
    return f"{FACILITY_PREFIXES[(offset % (len(FACILITY_PREFIXES) - 1)) + 1]}-{suffix}"


def _near_node(node: str, offset: int) -> str:
    return f"{node}-ALT{offset:02d}"


def _position_percent(prompt: str, needle: str, counter: TokenCounter) -> float:
    idx = prompt.index(needle)
    return round(counter.count(prompt[:idx]) * 100 / max(1, counter.count(prompt)), 2)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_samples": len(rows),
        "subset": rows[0].get("formal_subset"),
        "model_scope": rows[0].get("model_scope"),
        "models_to_run": rows[0].get("models_to_run"),
        "length_tokens_min": min(row["length_tokens_actual"] for row in rows),
        "length_tokens_max": max(row["length_tokens_actual"] for row in rows),
        "length_tokens_mean": round(mean(row["length_tokens_actual"] for row in rows), 2),
        "position_actual_min": min(row["position_percent_actual"] for row in rows),
        "position_actual_max": max(row["position_percent_actual"] for row in rows),
    }


def build_manifest(subsets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    subset_summaries = {name: summarize_rows(rows) for name, rows in subsets.items()}
    api_calls = sum(len(rows) * len(rows[0]["models_to_run"]) for rows in subsets.values())
    return {
        "dataset_name": "PAC-Test 2.0 Formal v5",
        "difficulty_anchor": {
            "calibration_run": "pac2_calibration_multidoc_v5_main",
            "chosen_decoy_count": CALIBRATED_DECOY,
            "reason": "decoy=64 produced the clearest weak/mid/strong separation in calibration.",
        },
        "length_tokens_target": LENGTH,
        "all_models": ALL_MODELS,
        "representative_models": REPRESENTATIVE_MODELS,
        "subsets": subset_summaries,
        "total_unique_samples": sum(len(rows) for rows in subsets.values()),
        "planned_api_calls": api_calls,
    }


def write_readme(manifest: dict[str, Any]) -> None:
    lines = [
        "# PAC-Test 2.0 Formal v5 Dataset",
        "",
        "This folder contains the formal PAC-Test 2.0 datasets built after difficulty calibration.",
        "The core difficulty anchor is `decoy_count=64`, selected from `pac2_calibration_multidoc_v5_main`.",
        "",
        "## Subsets",
        "",
        "| subset | samples | model scope | purpose |",
        "| --- | ---: | --- | --- |",
    ]
    purposes = {
        "PAC-A_position": "Position effect under calibrated high-similarity interference.",
        "PAC-B_interference": "Interference threshold curve around the calibrated critical point.",
        "PAC-C_binding_capacity": "Entity/property binding capacity with K entities and Q queried aliases.",
        "PAC-D_multihop_false_chain": "Multihop chain tracking under false-chain interference.",
    }
    for name, summary in manifest["subsets"].items():
        lines.append(f"| {name} | {summary['n_samples']} | {summary['model_scope']} | {purposes[name]} |")
    lines.extend(
        [
            "",
            f"Total unique samples: `{manifest['total_unique_samples']}`",
            f"Planned API calls: `{manifest['planned_api_calls']}`",
            "",
            "## Files",
            "",
            "- `data/PAC-A_position/samples.jsonl`",
            "- `data/PAC-B_interference/samples.jsonl`",
            "- `data/PAC-C_binding_capacity/samples.jsonl`",
            "- `data/PAC-D_multihop_false_chain/samples.jsonl`",
            "- `data/all_samples.jsonl`",
            "- `manifest.json`",
            "",
            "PAC-A and PAC-B run all 9 models. PAC-C and PAC-D run the representative 6-model panel.",
            "Use exact-match accuracy plus mean field accuracy, partial rate, and decoy capture rate.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def print_summary(manifest: dict[str, Any]) -> None:
    print("PAC-Test 2.0 Formal v5 dataset built")
    print(f"total_unique_samples={manifest['total_unique_samples']}")
    print(f"planned_api_calls={manifest['planned_api_calls']}")
    for name, summary in manifest["subsets"].items():
        print(
            f"{name}: n={summary['n_samples']} len={summary['length_tokens_min']}-{summary['length_tokens_max']} "
            f"pos={summary['position_actual_min']}-{summary['position_actual_max']} scope={summary['model_scope']}"
        )


if __name__ == "__main__":
    main()
