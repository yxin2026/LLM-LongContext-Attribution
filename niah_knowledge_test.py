#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge-Aware Needle in a Haystack Test
NIAH variant using real domain facts from facts_library.json as needles
and domain-specific filler from subset_A.jsonl as haystack.

This tests parametric memory interference:
  - Needle = real fact the model may already know from training
  - Haystack = same-domain filler (adds semantic interference)
  - Key question: does position still matter when the model "knows" the answer?

Usage:
    python niah_knowledge_test.py --model qwen2.5:7b
    python niah_knowledge_test.py --model deepseek-r1:7b --facts-per-domain 3
"""

import json
import re
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "niah_knowledge_results"

DEFAULT_BASE_URL = "http://localhost:11434/v1"

CHARS_PER_TOKEN = 1.71   # measured: 4096 chars = 2390 tokens on Qwen2.5-7b

MODEL_MAX_CTX = {
    "qwen2.5:7b":     32768,
    "deepseek-r1:7b": 131072,
    "qwen3.5:9b":     262144,
}

# Per-model context length grids (tokens), 1K to max, ~12-15 evenly-spaced levels
MODEL_CTX_LENGTHS = {
    "qwen2.5:7b": [
        1000, 3000, 5000, 8000, 11000, 14000,
        17000, 20000, 23000, 26000, 29000, 32000,
    ],
    "deepseek-r1:7b": [
        1000, 9000, 18000, 27000, 36000, 45000, 55000,
        64000, 73000, 82000, 91000, 100000, 109000, 118000, 128000,
    ],
    "qwen3.5:9b": [
        1000, 14000, 27000, 41000, 55000, 68000, 82000, 96000,
        109000, 123000, 136000, 150000, 164000, 177000, 200000,
    ],
}

# =========================================================
# Prompt  (same structure as original NIAH)
# =========================================================

SYSTEM_PROMPT = (
    "你是一个乐于助人的AI助手，请根据给定文本回答用户的问题。"
    "保持回答简短直接，不要给出文档之外的信息，也不要重复你的发现。"
)

def build_question(entity, attribute):
    return f"{entity}的{attribute}是什么？"

def build_user_prompt(context, question):
    return (
        "-------\n" + context + "\n-------\n"
        "用户问题：\n---------\n"
        + question + "\n"
        "不要给出文档之外的信息，也不要重复你的发现。"
    )

# =========================================================
# Data loading
# =========================================================

def load_facts(facts_path):
    with open(facts_path, encoding="utf-8") as f:
        return json.load(f)

def build_domain_corpus(subset_a_path, facts_set, min_len=10):
    """Extract domain-specific filler sentences from subset_A.jsonl."""
    domain_corpus = defaultdict(set)
    with open(subset_a_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            domain = d["domain"]
            # Split context into sentences on Chinese period
            sentences = d["context"].replace("。", "。|").split("|")
            for s in sentences:
                s = s.strip()
                if s and len(s) >= min_len and s not in facts_set:
                    domain_corpus[domain].add(s)
    return {k: list(v) for k, v in domain_corpus.items()}

# =========================================================
# Haystack building
# =========================================================

def build_haystack(corpus_sentences, target_chars):
    """Repeat domain filler sentences to reach target_chars."""
    base = " ".join(corpus_sentences)
    repeat = (target_chars // len(base)) + 2
    return (base * repeat)[:target_chars]

def insert_needle(haystack, needle, depth_percent):
    """Insert needle at depth_percent, snap to sentence boundary."""
    if depth_percent >= 100:
        return haystack + " " + needle
    insertion_point = int(len(haystack) * depth_percent / 100)
    boundary = haystack.rfind("。", max(0, insertion_point - 400), insertion_point + 1)
    if boundary != -1:
        boundary += 1
    else:
        boundary = insertion_point
    return haystack[:boundary] + " " + needle + " " + haystack[boundary:]

# =========================================================
# API
# =========================================================

def strip_thinking(text):
    # handles <think>/<thinking> pairs plus orphan close (open swallowed) and
    # orphan open (clipped before close). answer follows the final close tag.
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    m = list(re.finditer(r"</think(?:ing)?>", text, flags=re.IGNORECASE))
    if m:
        text = text[m[-1].end():]
    m = re.search(r"<think(?:ing)?>", text, flags=re.IGNORECASE)
    if m:
        text = text[:m.start()]
    return text.strip()

def query_model(model, context, question, base_url, max_tokens=1024,
                timeout=300, num_ctx=None):
    payload = {
        "model":       model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(context, question)},
        ],
        "temperature": 0.0,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    headers = {"Authorization": "Bearer ollama", "Content-Type": "application/json"}
    t0 = time.time()
    for attempt in range(3):
        try:
            resp = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            latency = time.time() - t0
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return strip_thinking(text), latency
        except requests.exceptions.Timeout:
            if attempt == 2:
                return "[TIMEOUT]", float(timeout)
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                return f"[ERROR: {e}]", time.time() - t0
            time.sleep(2 ** attempt)
    return "[FAILED]", 0.0

# =========================================================
# Scoring
# =========================================================

def score_response(pred, answer):
    pred_norm = re.sub(r"[\s\W]", "", pred).lower()
    ans_norm  = re.sub(r"[\s\W]", "", answer).lower()
    if ans_norm in pred_norm:
        return 10
    for token in re.findall(r"[一-鿿0-9a-z.]+", answer.lower()):
        if len(token) >= 2 and token in pred_norm:
            return 7
    return 1

# =========================================================
# Heatmap (per domain + overall)
# =========================================================

def plot_heatmap(matrix, context_lengths, depths, title, out_path):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        return

    fig_w = max(8, len(context_lengths) * 1.4)
    fig_h = max(5, len(depths) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels([f"{c:,}" for c in context_lengths], fontsize=9)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d:.0f}%" for d in depths], fontsize=9)
    ax.set_xlabel("Context Length (chars)", fontsize=11)
    ax.set_ylabel("Needle Depth", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(len(depths)):
        for j in range(len(context_lengths)):
            v = matrix[i][j]
            color = "white" if v < 0.35 or v > 0.75 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {out_path}")

def build_matrix(results_for_model, context_lengths, depths):
    """Average scores per (depth, context_length) cell across all facts."""
    cells = defaultdict(list)
    for r in results_for_model:
        key = (r["depth_percent"], r["context_length"])
        cells[key].append(r["score"] / 10.0)
    matrix = np.zeros((len(depths), len(context_lengths)))
    for i, d in enumerate(depths):
        for j, cl in enumerate(context_lengths):
            vals = cells.get((d, cl), [])
            matrix[i][j] = np.mean(vals) if vals else 0.0
    return matrix

# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Aware NIAH Test (Chinese, Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",            default="qwen2.5:7b")
    parser.add_argument("--base-url",         default=DEFAULT_BASE_URL)
    parser.add_argument("--context-lengths-tokens", type=int, nargs="+",
                        default=None,
                        help="Context lengths in TOKENS. Defaults to per-model preset (1K to model max).")
    parser.add_argument("--depth-percents",   type=float, nargs="+",
                        default=list(np.linspace(0, 100, 15).round(1)))
    parser.add_argument("--facts-per-domain", type=int, default=5,
                        help="Number of facts sampled per domain")
    parser.add_argument("--domains",          nargs="+",
                        default=["finance", "computer_science", "medicine", "law", "education"])
    parser.add_argument("--max-tokens",       type=int, default=1024)
    parser.add_argument("--timeout",          type=int, default=300)
    parser.add_argument("--no-plot",          action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[:/\\]", "_", args.model)

    ctx_tokens = args.context_lengths_tokens or MODEL_CTX_LENGTHS.get(
        args.model, [1000, 3000, 5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 29000, 32000]
    )
    ctx_chars  = [int(t * CHARS_PER_TOKEN) for t in ctx_tokens]
    num_ctx    = MODEL_MAX_CTX.get(args.model, max(ctx_tokens))

    # Load knowledge base
    facts_path    = BASE_DIR / "core_knowledge" / "facts_library.json"
    subset_a_path = DATA_DIR / "subset_A.jsonl"
    all_facts = load_facts(facts_path)
    facts_set = {f["fact_text"] for f in all_facts}

    print(f"\nBuilding domain corpora from subset_A...")
    domain_corpus = build_domain_corpus(subset_a_path, facts_set)
    for dom, sents in domain_corpus.items():
        print(f"  {dom}: {len(sents)} unique filler sentences")

    # Sample facts per domain
    domain_facts = defaultdict(list)
    for f in all_facts:
        if f["domain"] in args.domains:
            domain_facts[f["domain"]].append(f)

    sampled = {}
    rng = np.random.default_rng(42)
    for dom in args.domains:
        pool = domain_facts[dom]
        n    = min(args.facts_per_domain, len(pool))
        idx  = rng.choice(len(pool), n, replace=False)
        sampled[dom] = [pool[i] for i in idx]

    n_facts = sum(len(v) for v in sampled.values())
    total_combos = n_facts * len(ctx_tokens) * len(args.depth_percents)
    print(f"\nModel   : {args.model}")
    print(f"Max ctx : {num_ctx:,} tokens")
    print(f"Test ctx: {[f'{t//1000}K' for t in ctx_tokens]}")
    print(f"Domains : {args.domains}")
    print(f"Facts   : {n_facts} total ({args.facts_per_domain}/domain)")
    print(f"Grid    : {len(ctx_tokens)} lengths x {len(args.depth_percents)} depths")
    print(f"Combos  : {total_combos}")
    print(f"Results : {RESULTS_DIR}\n")

    all_results = []

    for domain in args.domains:
        corpus    = domain_corpus.get(domain, [])
        facts_dom = sampled.get(domain, [])
        if not corpus or not facts_dom:
            print(f"Skipping {domain}: missing corpus or facts.")
            continue

        for fact in tqdm(facts_dom, desc=f"{domain}", ncols=90, leave=False):
            needle   = fact["fact_text"]
            question = build_question(fact["entity_name"], fact["attribute"])
            answer   = fact["value"]

            for ctx_tok, ctx_char in zip(ctx_tokens, ctx_chars):
                for depth_percent in args.depth_percents:
                    fname = (
                        f"{safe_model}_{domain}"
                        f"_fid{fact['id']}"
                        f"_tok{ctx_tok}"
                        f"_dep{depth_percent:.1f}"
                    )
                    result_file = RESULTS_DIR / f"{fname}.json"

                    if result_file.exists():
                        with open(result_file, encoding="utf-8") as f:
                            all_results.append(json.load(f))
                        continue

                    haystack = build_haystack(corpus, ctx_char - len(needle) - 200)
                    context  = insert_needle(haystack, needle, depth_percent)
                    context  = context[:ctx_char]

                    pred, latency = query_model(
                        args.model, context, question,
                        args.base_url, args.max_tokens, args.timeout,
                        num_ctx=num_ctx,
                    )
                    s = score_response(pred, answer)

                    result = {
                        "model":                  args.model,
                        "domain":                 domain,
                        "fact_id":                fact["id"],
                        "needle":                 needle,
                        "question":               question,
                        "answer":                 answer,
                        "context_length_tokens":  ctx_tok,
                        "context_length_chars":   ctx_char,
                        "depth_percent":          depth_percent,
                        "model_response":         pred,
                        "score":                  s,
                        "test_duration_seconds":  round(latency, 2),
                        "test_timestamp_utc":     datetime.now(timezone.utc).isoformat(),
                    }
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    all_results.append(result)

    # Summary
    total   = len(all_results)
    perfect = sum(1 for r in all_results if r["score"] == 10)
    partial = sum(1 for r in all_results if r["score"] == 7)
    miss    = sum(1 for r in all_results if r["score"] == 1)
    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Total   : {total}")
    print(f"  Score 10: {perfect} ({perfect/total:.1%})")
    print(f"  Score  7: {partial} ({partial/total:.1%})")
    print(f"  Score  1: {miss}  ({miss/total:.1%})")

    # Per-domain breakdown
    for dom in args.domains:
        dom_res = [r for r in all_results if r["domain"] == dom]
        if dom_res:
            p10 = sum(1 for r in dom_res if r["score"] == 10)
            print(f"  [{dom}] n={len(dom_res)}  Score10={p10/len(dom_res):.1%}")
    print(f"{'='*60}")

    if not args.no_plot:
        # Overall heatmap
        matrix = build_matrix(all_results, args.context_lengths, args.depth_percents)
        plot_heatmap(
            matrix, args.context_lengths, args.depth_percents,
            f"Knowledge NIAH (Overall) — {args.model}",
            RESULTS_DIR / f"{safe_model}_overall_heatmap.png",
        )
        # Per-domain heatmaps
        for dom in args.domains:
            dom_res = [r for r in all_results if r["domain"] == dom]
            if not dom_res:
                continue
            mat = build_matrix(dom_res, args.context_lengths, args.depth_percents)
            plot_heatmap(
                mat, args.context_lengths, args.depth_percents,
                f"Knowledge NIAH [{dom}] — {args.model}",
                RESULTS_DIR / f"{safe_model}_{dom}_heatmap.png",
            )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
