from __future__ import annotations

import random
import re
from typing import Any

from lmaf.utils.token_count import TokenCounter


PAC2_B_INTERFERENCE_TYPE = "multidoc_profile_triad_high_similarity"
PAC2_DIFFICULTY_VERSION = "pac2_b_multidoc_triad_v5"

ATTRIBUTES = [
    "night-release-code",
    "night-seal-code",
    "night-review-code",
    "day-release-code",
    "night-release-id",
    "cold-chain-handoff-code",
    "inbound-review-code",
    "exception-register-code",
]

FACILITY_PREFIXES = [
    "Huadong-Yiyao-Lenglian-Zhongxin",
    "Huanan-Yiyao-Lenglian-Zhongxin",
    "Huadong-Yiliao-Lenglian-Zhongxin",
    "Huadong-Yiyao-Lengcang-Zhongxin",
    "Huazhong-Yiyao-Lenglian-Zhongxin",
]

FILLER_SENTENCES = [
    "Audit notes describe storage, transfer, shift logs, and routine cold-chain checks.",
    "The archive contains aliases, old values, revoked values, near-duplicate records, and policy exceptions.",
    "Only active aliases, approved profiles, primary-channel evidence, and valid records count for final answers.",
    "Nearby entity numbers, nearby batches, inactive statuses, and after-cutoff memos are deliberately misleading.",
    "The requested value is not inferable from names alone; it requires preserving several bindings.",
]


def generate_pac2_b_calibration(
    length: int = 32000,
    position: int = 50,
    decoy_count: int = 64,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Generate a hard PAC-Test 2.0 calibration sample.

    v5 borrows LongBench's useful pressure: separated evidence, profile rules,
    multi-field output, and realistic exclusion conditions. The task is no
    longer a single key-value lookup.
    """

    counter = counter or TokenCounter()
    rng = random.Random(seed * 1_000_003 + decoy_count * 10_007 + sample_index)
    entity_id = 20 + ((seed + sample_index * 7) % 760)
    target_entity = f"{FACILITY_PREFIXES[0]}-{entity_id:03d}"
    target_alias = f"ALIAS-{sample_index:03d}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10, 99)}"
    profile_id = f"PROFILE-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(100, 999)}"
    target_fields = [ATTRIBUTES[0], ATTRIBUTES[1], ATTRIBUTES[7]]
    target_qualifier = _make_qualifier(sample_index)
    cutoff_time = "23:50"
    target_times = ["22:58", "23:17", "23:41"]
    expected_answers: list[str] = []
    used_values: set[str] = set()
    for _field in target_fields:
        value = _make_value(rng)
        while value in used_values:
            value = _make_value(rng)
        used_values.add(value)
        expected_answers.append(value)
    answer = "|".join(expected_answers)

    target_records = [
        _evidence_line(
            doc_id=f"T{seed}-{sample_index:04d}-{idx + 1}",
            timestamp=target_times[idx],
            entity=target_entity,
            attribute=field,
            qualifier=target_qualifier,
            value=expected_answers[idx],
            status="valid",
            review="approved",
            channel="primary",
            source="route-ledger",
        )
        for idx, field in enumerate(target_fields)
    ]
    decoys = _make_decoys(
        rng=rng,
        sample_index=sample_index,
        target_entity=target_entity,
        target_alias=target_alias,
        target_qualifier=target_qualifier,
        target_fields=target_fields,
        expected_answers=expected_answers,
        decoy_count=decoy_count,
        cutoff_time=cutoff_time,
        used_values=used_values,
    )
    alias_lines = _make_alias_lines(
        rng=rng,
        sample_index=sample_index,
        target_alias=target_alias,
        target_entity=target_entity,
        decoy_count=decoy_count,
    )
    profile_lines = _make_profile_lines(rng, profile_id, target_fields)
    policy_lines = _make_policy_lines(cutoff_time)
    rng.shuffle(decoys)
    rng.shuffle(alias_lines)
    rng.shuffle(profile_lines)

    local_count = min(len(decoys), max(0, min(80, decoy_count // 2)))
    local_decoys = decoys[:local_count]
    remaining_decoys = decoys[local_count:]
    local_a = len(local_decoys) // 3
    local_b = (len(local_decoys) * 2) // 3
    target_prefix = _join(
        [
            _format_block(local_decoys[:local_a]),
            target_records[0],
            _format_block(local_decoys[local_a:local_b]),
        ]
    )
    target_fact = target_records[1]
    target_suffix = _join([_format_block(local_decoys[local_b:]), target_records[2]])
    split = len(remaining_decoys) // 2
    pre_decoys = remaining_decoys[:split]
    post_decoys = remaining_decoys[split:]

    alias_split = len(alias_lines) // 2
    profile_split = len(profile_lines) // 2
    pre_block = _join(
        [
            _format_aliases(alias_lines[:alias_split]),
            _format_profiles(profile_lines[:profile_split]),
            _format_policy(policy_lines[:2]),
            _format_block(pre_decoys),
        ]
    )
    post_block = _join(
        [
            _format_block(post_decoys),
            _format_profiles(profile_lines[profile_split:]),
            _format_aliases(alias_lines[alias_split:]),
            _format_policy(policy_lines[2:]),
        ]
    )

    instruction = (
        "Use only the long context. This is an audit-style multi-document task. "
        "Alias lines map A to E and are usable only when AS=active. Profile lines define "
        "the field order and are usable only when PSTATUS=approved. Evidence memos use "
        "DOC, T, E, F, Q, V, S, REVIEW, CHANNEL, and SRC. For each field named by the "
        "approved profile, select the latest memo before the cutoff whose S=valid, "
        "REVIEW=approved, CHANNEL=primary, E matches the active alias entity, and Q matches "
        "the requested batch. Reject revoked, expired, draft, pending-review, secondary-channel, "
        "after-cutoff, inactive-alias, wrong-profile, wrong-field, wrong-batch, and near-entity "
        "evidence. Output the selected V values in profile step order, separated by |, with no explanation."
    )
    question = (
        f"[Question]\nFor alias A={target_alias}, batch Q={target_qualifier}, approved profile "
        f"PID={profile_id}, and cutoff T<{cutoff_time}, what is the final three-part route triplet? "
        "Resolve the active alias, use the profile step order, and output only V1|V2|V3.\n[Answer]"
    )
    prompt = _compose_prompt(
        length=length,
        position=position,
        instruction=instruction,
        pre_block=pre_block,
        target_prefix=target_prefix,
        target_fact=target_fact,
        target_suffix=target_suffix,
        post_block=post_block,
        question=question,
        counter=counter,
        seed=seed + sample_index + decoy_count,
        forbidden=(target_alias, target_entity, target_qualifier, answer, *expected_answers),
    )
    actual_position = _position_percent(prompt, target_fact, counter)

    return {
        "experiment": "pac2",
        "subtask": "B_calibration",
        "pac2_axis": "interference_threshold",
        "difficulty_version": PAC2_DIFFICULTY_VERSION,
        "sample_id": (
            f"pac2_B_v5_len{length}_pos{position}_decoy{decoy_count}_"
            f"seed{seed}_{sample_index:04d}"
        ),
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": position,
        "position_percent_actual": actual_position,
        "decoy_count": decoy_count,
        "density": decoy_count,
        "interference_type": PAC2_B_INTERFERENCE_TYPE,
        "prompt": prompt,
        "answer": answer,
        "expected_answers": expected_answers,
        "answer_format": "V1|V2|V3",
        "target_alias": target_alias,
        "target_entity": target_entity,
        "target_attribute": "|".join(target_fields),
        "target_fields": target_fields,
        "target_qualifier": target_qualifier,
        "target_status": "valid",
        "target_time": "|".join(target_times),
        "cutoff_time": cutoff_time,
        "profile_id": profile_id,
        "target_fact": target_fact,
        "target_facts": target_records,
        "alias_lines": alias_lines,
        "profile_lines": profile_lines,
        "policy_lines": policy_lines,
        "distractor_answers": [item["value"] for item in decoys],
        "decoy_values": [item["value"] for item in decoys],
        "decoy_bindings": [
            {
                "entity": item["entity"],
                "attribute": item["attribute"],
                "qualifier": item["qualifier"],
                "value": item["value"],
                "status": item["status"],
                "timestamp": item["timestamp"],
                "review": item.get("review", ""),
                "channel": item.get("channel", ""),
                "confusion_type": item["confusion_type"],
            }
            for item in decoys
        ],
        "seed": seed,
        "error": None,
    }


def score_pac2_sample(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    answer = str(sample.get("answer") or "")
    expected = [str(item) for item in sample.get("expected_answers") or []]
    pred_norm = _normalize_answer(prediction)
    answer_norm = _normalize_answer(answer)
    decoys = [str(item) for item in sample.get("decoy_values") or sample.get("distractor_answers") or []]
    expected_hits = sum(1 for item in expected if _normalize_answer(item) in pred_norm)
    field_accuracy = expected_hits / len(expected) if expected else 0.0

    if not pred_norm:
        error_type = "omission"
    elif answer_norm and answer_norm in pred_norm:
        error_type = "correct"
    else:
        decoy_hit = next((value for value in decoys if _normalize_answer(value) in pred_norm), None)
        if decoy_hit:
            error_type = "decoy_value_capture"
        elif expected and expected_hits:
            error_type = "partial_triplet"
        elif _looks_like_near_miss(answer, prediction):
            error_type = "near_miss_value"
        elif any(token in prediction.lower() for token in ("not sure", "unknown", "cannot", "unspecified")):
            error_type = "omission"
        else:
            error_type = "other_error"

    return {
        "score": float(error_type == "correct"),
        "field_accuracy": round(field_accuracy, 4),
        "metric": "pac2_triplet_exact_with_decoy_error_type",
        "error_type": error_type,
        "decoy_captured": int(error_type == "decoy_value_capture"),
        "omitted": int(error_type == "omission"),
        "near_miss": int(error_type == "near_miss_value"),
    }


def _make_decoys(
    rng: random.Random,
    sample_index: int,
    target_entity: str,
    target_alias: str,
    target_qualifier: str,
    target_fields: list[str],
    expected_answers: list[str],
    decoy_count: int,
    cutoff_time: str,
    used_values: set[str],
) -> list[dict[str, str]]:
    decoys: list[dict[str, str]] = []
    cutoff_minutes = _time_to_minutes(cutoff_time)
    for i in range(decoy_count):
        field = target_fields[i % len(target_fields)]
        base_answer = expected_answers[i % len(expected_answers)]
        mode = i % 14
        if mode == 0:
            entity = target_entity
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _older_time(i)
            confusion = "same_binding_older_valid"
        elif mode == 1:
            entity = target_entity
            attribute = field
            qualifier = target_qualifier
            status = _invalid_status(i)
            review = "approved"
            channel = "primary"
            timestamp = _later_time(i)
            confusion = "same_binding_later_invalid"
        elif mode == 2:
            entity = target_entity
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _after_cutoff_time(cutoff_minutes, i)
            confusion = "same_binding_after_cutoff_valid"
        elif mode == 3:
            entity = _near_entity(target_entity, i)
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "near_entity_same_field_batch_latest_valid"
        elif mode == 4:
            entity = target_entity
            attribute = ATTRIBUTES[(i % (len(ATTRIBUTES) - 1)) + 1]
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "same_entity_near_field_latest_valid"
        elif mode == 5:
            entity = target_entity
            attribute = field
            qualifier = _near_qualifier(target_qualifier, i)
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "same_entity_same_field_near_batch_latest_valid"
        elif mode == 6:
            entity = _prefix_variant(target_entity, i)
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "prefix_variant_latest_valid"
        elif mode == 7:
            entity = target_entity.replace("Zhongxin", "Backup-Zhongxin")
            attribute = field
            qualifier = _near_qualifier(target_qualifier, i)
            status = _invalid_status(i)
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "backup_entity_invalid"
        elif mode == 8:
            entity = target_entity
            attribute = field
            qualifier = _near_qualifier(target_qualifier, i)
            status = _invalid_status(i)
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "near_batch_invalid"
        elif mode == 9:
            entity = target_entity
            attribute = f"{field}-verify"
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "attribute_suffix_latest_valid"
        elif mode == 10:
            entity = target_entity
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "pending"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "same_binding_pending_review"
        elif mode == 11:
            entity = target_entity
            attribute = field
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "secondary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "same_binding_secondary_channel"
        elif mode == 12:
            entity = target_entity
            attribute = target_fields[(i + 1) % len(target_fields)]
            qualifier = target_qualifier
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _latest_before_cutoff(cutoff_minutes, i)
            confusion = "profile_step_swap_latest_valid"
        else:
            entity = _near_entity(target_entity, sample_index + i * 3)
            attribute = field
            qualifier = _near_qualifier(target_qualifier, i)
            status = "valid"
            review = "approved"
            channel = "primary"
            timestamp = _older_time(i)
            confusion = f"alias_noise_for_{target_alias}"

        value = _unique_variant(base_answer, rng, used_values, i)
        used_values.add(value)
        decoys.append(
            {
                "entity": entity,
                "attribute": attribute,
                "qualifier": qualifier,
                "value": value,
                "status": status,
                "timestamp": timestamp,
                "review": review,
                "channel": channel,
                "confusion_type": confusion,
                "fact": _evidence_line(
                    doc_id=f"D{sample_index:04d}-{i:04d}",
                    timestamp=timestamp,
                    entity=entity,
                    attribute=attribute,
                    qualifier=qualifier,
                    value=value,
                    status=status,
                    review=review,
                    channel=channel,
                    source="decoy-ledger",
                ),
            }
        )
    return decoys


def _make_alias_lines(
    rng: random.Random,
    sample_index: int,
    target_alias: str,
    target_entity: str,
    decoy_count: int,
) -> list[str]:
    alias_lines = [_alias_line(target_alias, target_entity, "active")]
    n_alias_decoys = min(96, max(12, decoy_count // 4 if decoy_count else 12))
    for i in range(n_alias_decoys):
        if i % 4 == 0:
            alias = target_alias
            entity = _near_entity(target_entity, i + 11)
            status = "inactive"
        elif i % 4 == 1:
            alias = f"{target_alias}-OLD"
            entity = target_entity
            status = "inactive"
        elif i % 4 == 2:
            alias = f"ALIAS-{sample_index:03d}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10, 99)}"
            entity = _near_entity(target_entity, i)
            status = "active"
        else:
            alias = target_alias.replace("ALIAS", "A-LIAS")
            entity = _prefix_variant(target_entity, i)
            status = "active"
        alias_lines.append(_alias_line(alias, entity, status))
    return alias_lines


def _make_profile_lines(rng: random.Random, profile_id: str, target_fields: list[str]) -> list[str]:
    lines = [_profile_line(profile_id, idx + 1, field, "approved") for idx, field in enumerate(target_fields)]
    wrong_profile = f"{profile_id}-ALT"
    for idx, field in enumerate(reversed(target_fields), start=1):
        lines.append(_profile_line(wrong_profile, idx, field, "approved"))
    for idx, field in enumerate(target_fields, start=1):
        wrong_field = ATTRIBUTES[(ATTRIBUTES.index(field) + 3) % len(ATTRIBUTES)]
        lines.append(_profile_line(profile_id, idx, wrong_field, "draft"))
    for idx in range(6):
        lines.append(
            _profile_line(
                f"PROFILE-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(100, 999)}",
                1 + (idx % 3),
                ATTRIBUTES[(idx + 2) % len(ATTRIBUTES)],
                "approved",
            )
        )
    return lines


def _make_policy_lines(cutoff_time: str) -> list[str]:
    return [
        f"POLICY|RULE=latest-before-cutoff|CUT={cutoff_time}|OP=strictly-before",
        "POLICY|RULE=evidence-filter|S=valid|REVIEW=approved|CHANNEL=primary",
        "POLICY|RULE=profile-order|USE=approved PID only|ORDER=STEP_ASC",
        "POLICY|RULE=reject|ITEMS=inactive-alias,wrong-profile,wrong-field,wrong-batch,near-entity,after-cutoff,pending-review,secondary-channel,draft,revoked,expired",
    ]


def _alias_line(alias: str, entity: str, status: str) -> str:
    return f"M|A={alias}|E={entity}|AS={status}"


def _profile_line(profile_id: str, step: int, field: str, status: str) -> str:
    return f"P|PID={profile_id}|STEP={step}|F={field}|PSTATUS={status}"


def _evidence_line(
    doc_id: str,
    timestamp: str,
    entity: str,
    attribute: str,
    qualifier: str,
    value: str,
    status: str,
    review: str,
    channel: str,
    source: str,
) -> str:
    return (
        f"DOC={doc_id}|T={timestamp}|E={entity}|F={attribute}|Q={qualifier}|V={value}|"
        f"S={status}|REVIEW={review}|CHANNEL={channel}|SRC={source}"
    )


def _record_line(record_id: str, timestamp: str, entity: str, attribute: str, qualifier: str, value: str, status: str) -> str:
    return f"R={record_id}|T={timestamp}|E={entity}|F={attribute}|Q={qualifier}|V={value}|S={status}"


def _format_block(decoys: list[dict[str, str]]) -> str:
    return "\n".join(item["fact"] for item in decoys)


def _format_aliases(lines: list[str]) -> str:
    return "\n".join(lines)


def _format_profiles(lines: list[str]) -> str:
    return "\n".join(lines)


def _format_policy(lines: list[str]) -> str:
    return "\n".join(lines)


def _compose_prompt(
    length: int,
    position: int,
    instruction: str,
    pre_block: str,
    target_prefix: str,
    target_fact: str,
    target_suffix: str,
    post_block: str,
    question: str,
    counter: TokenCounter,
    seed: int,
    forbidden: tuple[str, ...],
) -> str:
    prefix = _join(["[Task]", instruction, "[Long Context]", pre_block])
    desired_start = int(length * position / 100)
    filler_before_budget = max(0, desired_start - counter.count(prefix) - counter.count(target_prefix) - 4)
    filler_before = _make_filler(filler_before_budget, seed=seed, counter=counter, forbidden=forbidden)
    before_target = _join([prefix, filler_before, target_prefix])
    used = counter.count(_join([before_target, target_fact, target_suffix, post_block, question]))
    tail_budget = max(0, length - used - 4)
    tail = _make_filler(tail_budget, seed=seed + 991, counter=counter, forbidden=forbidden)
    return _join([before_target, target_fact, target_suffix, post_block, tail, question])


def _make_filler(
    target_tokens: int,
    seed: int,
    counter: TokenCounter,
    forbidden: tuple[str, ...] = (),
) -> str:
    if target_tokens <= 0:
        return ""
    forbidden_terms = [item for item in forbidden if item]
    sentences = [
        sentence
        for sentence in FILLER_SENTENCES
        if not any(term in sentence for term in forbidden_terms)
    ] or FILLER_SENTENCES

    def text_for(n_sentences: int) -> str:
        return "\n".join(
            f"BG{idx + 1:04d}: {sentences[(seed + idx) % len(sentences)]}"
            for idx in range(n_sentences)
        )

    lo = 0
    hi = max(1, target_tokens // 10)
    while counter.count(text_for(hi)) < target_tokens and hi < max(100, target_tokens * 2):
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text_for(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text_for(lo)


def _make_value(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    part1 = "".join(rng.choice(alphabet) for _ in range(3))
    part2 = "".join(rng.choice(alphabet) for _ in range(3))
    part3 = "".join(rng.choice(alphabet) for _ in range(3))
    return f"{part1}-{part2}-{part3}"


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
        round_id = max(1, round_id + 1)
    return f"B{batch:03d}-{shift}-{round_id:02d}"


def _older_time(offset: int) -> str:
    hour = 1 + (offset * 3) % 20
    minute = (offset * 7) % 60
    if hour >= 23:
        hour = 22
    return f"{hour:02d}:{minute:02d}"


def _later_time(offset: int) -> str:
    minute = (20 + offset * 7) % 50
    return f"23:{minute:02d}"


def _latest_before_cutoff(cutoff_minutes: int, offset: int) -> str:
    minute = max(0, cutoff_minutes - 1 - (offset % 11))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _after_cutoff_time(cutoff_minutes: int, offset: int) -> str:
    minute = min(23 * 60 + 59, cutoff_minutes + 1 + (offset % 9))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _time_to_minutes(text: str) -> int:
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute)


def _invalid_status(offset: int) -> str:
    return ["expired", "revoked", "draft"][offset % 3]


def _unique_variant(answer: str, rng: random.Random, used: set[str], offset: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    chars = list(answer)
    mutable = [idx for idx, ch in enumerate(chars) if ch != "-"]
    for attempt in range(100):
        candidate = chars[:]
        idx = mutable[(offset + attempt) % len(mutable)]
        replacement = rng.choice(alphabet)
        if replacement == candidate[idx]:
            replacement = alphabet[(alphabet.index(replacement) + 1) % len(alphabet)]
        candidate[idx] = replacement
        if attempt % 3 == 2:
            idx2 = mutable[(offset * 3 + attempt) % len(mutable)]
            replacement2 = rng.choice(alphabet)
            if replacement2 == candidate[idx2]:
                replacement2 = alphabet[(alphabet.index(replacement2) + 2) % len(alphabet)]
            candidate[idx2] = replacement2
        value = "".join(candidate)
        if value not in used and value != answer:
            return value
    return _make_value(rng)


def _near_entity(target_entity: str, offset: int) -> str:
    match = re.search(r"(\d+)$", target_entity)
    if not match:
        return f"{target_entity}-near{offset:03d}"
    number = int(match.group(1))
    width = len(match.group(1))
    delta = (offset % 9) - 4
    if delta == 0:
        delta = 5
    new_number = max(1, number + delta)
    return re.sub(r"\d+$", f"{new_number:0{width}d}", target_entity)


def _prefix_variant(target_entity: str, offset: int) -> str:
    prefix = FACILITY_PREFIXES[(offset % (len(FACILITY_PREFIXES) - 1)) + 1]
    match = re.search(r"(\d+)$", target_entity)
    suffix = match.group(1) if match else f"{offset:03d}"
    return f"{prefix}-{suffix}"


def _position_percent(prompt: str, needle: str, counter: TokenCounter) -> float:
    idx = prompt.index(needle)
    prefix_tokens = counter.count(prompt[:idx])
    total_tokens = max(1, counter.count(prompt))
    return round(prefix_tokens * 100 / total_tokens, 2)


def _normalize_answer(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _looks_like_near_miss(answer: str, prediction: str) -> bool:
    expected_parts = [part for part in str(answer).split("|") if part]
    candidates = re.findall(r"[A-Za-z0-9]{2,}[-A-Za-z0-9]*", prediction)
    for answer_part in expected_parts or [answer]:
        answer_norm = _normalize_answer(answer_part)
        for candidate in candidates:
            cand_norm = _normalize_answer(candidate)
            if len(cand_norm) == len(answer_norm) and _hamming(cand_norm, answer_norm) <= 2:
                return True
    return False


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right))


def _join(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)
