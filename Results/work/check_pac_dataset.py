from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


FILES = ["subset_A.jsonl", "subset_B.jsonl", "subset_C.jsonl", "subset_D.jsonl", "PAC-Test_complete.jsonl"]
REQUIRED = ["sample_id", "subset", "total_length", "total_length_unit", "context", "question", "answer"]
CONTROL_FIELDS = [
    "domain",
    "position_ratio",
    "dilution_type",
    "noise_density",
    "similarity_level",
    "distance_level",
    "target_entity",
    "confusable_entity",
    "chain_type",
    "num_hops",
    "fact",
    "fact_chain",
]


def main() -> None:
    base = Path(sys.argv[1])
    print(f"base={base}")
    for fname in FILES:
        inspect_file(base / fname)


def inspect_file(path: Path) -> None:
    n = 0
    subsets: Counter[str] = Counter()
    lengths: Counter[object] = Counter()
    units: Counter[str] = Counter()
    ids: set[str] = set()
    dup = 0
    bad_required = 0
    key_union: set[str] = set()
    controls: dict[str, Counter[str]] = defaultdict(Counter)
    sample = None
    sample_has_prompt = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample is None:
                sample = row
            n += 1
            sid = str(row.get("sample_id"))
            if sid in ids:
                dup += 1
            ids.add(sid)
            subsets[str(row.get("subset"))] += 1
            lengths[row.get("total_length")] += 1
            units[str(row.get("total_length_unit"))] += 1
            key_union.update(row)
            sample_has_prompt = sample_has_prompt or "prompt" in row
            if any(req not in row or row.get(req) in (None, "") for req in REQUIRED):
                bad_required += 1
            for field in CONTROL_FIELDS:
                if field in row:
                    value = row[field]
                    if isinstance(value, list):
                        value = f"list[{len(value)}]"
                    controls[field][str(value)] += 1

    print(f"\n== {path.name} ==")
    print(f"rows={n} subsets={dict(subsets)} duplicate_ids={dup} bad_required_rows={bad_required}")
    print(f"total_length_unit={dict(units)}")
    print(f"lengths={dict(sorted(lengths.items(), key=lambda item: str(item[0])))}")
    print(f"has_prompt_field={sample_has_prompt}")
    print(f"keys={sorted(key_union)}")
    for field in CONTROL_FIELDS:
        if field in controls:
            values = dict(sorted(controls[field].items(), key=lambda item: item[0]))
            if len(values) > 12:
                values = dict(list(values.items())[:12])
                print(f"{field}={values} ...")
            else:
                print(f"{field}={values}")
    if sample:
        q = str(sample.get("question", ""))
        a = str(sample.get("answer", ""))
        print(f"sample_id={sample.get('sample_id')}")
        print(f"question_escape={q[:40].encode('unicode_escape').decode('ascii')}")
        print(f"answer_escape={a[:40].encode('unicode_escape').decode('ascii')}")


if __name__ == "__main__":
    main()
