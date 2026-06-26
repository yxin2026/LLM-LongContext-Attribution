#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Needle in a Haystack Test
Follows gkamradt/LLMTest_NeedleInAHaystack methodology.
Adapted for Chinese text and local Ollama models.

Usage:
    python niah_test.py --model qwen2.5:7b
    python niah_test.py --model deepseek-r1:7b
    python niah_test.py --model qwen3.5:9b
"""

import json
import re
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "niah_results"

DEFAULT_BASE_URL = "http://localhost:11434/v1"

CHARS_PER_TOKEN = 1.71   # measured: 4096 chars = 2390 tokens on Qwen2.5-7b

# Per-model max context (tokens), used to set Ollama num_ctx
MODEL_MAX_CTX = {
    "qwen2.5:7b":    32768,
    "deepseek-r1:7b": 131072,
    "qwen3.5:9b":    262144,
}

# Per-model test context lengths (tokens)
# Approximately linear spacing following original NIAH grid
MODEL_CTX_LENGTHS = {
    "qwen2.5:7b": [
        1000, 3000, 5000, 8000, 11000, 14000, 17000, 20000,
        23000, 26000, 29000, 32000,
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

NEEDLE   = "量子隼-7型实验飞船于2089年完成了首次超光速测试，飞行距离达到47.3光年。"
QUESTION = "量子隼-7型实验飞船的首次超光速测试飞行距离是多少？"
ANSWER   = "47.3光年"

SYSTEM_PROMPT = (
    "你是一个乐于助人的AI助手，请根据给定文本回答用户的问题。"
    "保持回答简短直接，不要给出文档之外的信息，也不要重复你的发现。"
)

def build_user_prompt(context, question):
    return (
        "-------\n" + context + "\n-------\n"
        "用户问题：\n---------\n"
        + question + "\n"
        "不要给出文档之外的信息，也不要重复你的发现。"
    )

_CORPUS = [
    "近年来，人工智能领域的研究取得了显著进展，深度学习模型在多项任务上超越了人类表现。",
    "多项研究表明，大规模预训练模型在自然语言处理场景中表现出色，显著降低了标注成本。",
    "从技术架构来看，Transformer架构采用了自注意力机制作为其核心方法。",
    "实验结果表明，该方法在标准评测基准上达到了业界领先的准确率。",
    "这一突破为计算机视觉领域的发展开辟了新的方向，引发了广泛关注。",
    "与此同时，研究人员也在探索多模态学习与强化学习的结合应用。",
    "值得注意的是，大型语言模型在处理长文本推理问题时仍存在一定局限。",
    "未来的研究将聚焦于提升模型的可解释性与鲁棒性。",
    "从工程实践角度，模型的部署需要考虑计算成本、延迟和内存占用等因素。",
    "综述性文献指出，预训练与微调范式已成为自然语言处理的主流方法之一。",
    "代码实现上，主流模型通常基于PyTorch或JAX框架进行开发与训练。",
    "性能优化方面，研究者们提出了量化、剪枝和知识蒸馏等改进策略。",
    "在工业界，大语言模型已被广泛应用于客服、代码生成和文档摘要等场景。",
    "对比实验显示，引入位置编码改进后，模型在长文本任务上的表现提升了约15%。",
    "理论分析表明，注意力机制的时间复杂度为序列长度的平方。",
    "市场数据显示，全球云计算市场规模持续扩大，年增长率保持在两位数以上。",
    "投资者情绪、宏观经济数据和政策变化都可能引发市场的剧烈波动。",
    "从宏观经济角度，货币政策通过调节利率和货币供应量影响经济运行。",
    "中央银行运用公开市场操作、存款准备金率和再贴现率等工具实施货币政策。",
    "风险管理实践中，压力测试评估极端情景下金融机构的承受能力。",
    "投资组合理论表明，分散投资能够有效降低非系统性风险。",
    "资产配置需要根据投资者的风险偏好和投资期限进行动态调整。",
    "监管政策方面，国际协议为银行业监管提供了统一标准。",
    "行为金融学研究发现，投资者在决策过程中普遍存在认知偏差。",
    "金融科技的发展深刻改变了传统金融服务模式，移动支付和区块链技术重塑了金融业态。",
    "临床研究表明，规律运动在慢性病预防中显示出良好的健康收益。",
    "从病理机制来看，炎症反应在多种慢性疾病的发展中发挥核心作用。",
    "多项随机对照试验证实，早期干预可将心血管疾病风险显著降低。",
    "在药物代谢方面，肝脏是大多数药物的主要代谢器官。",
    "基因组学研究提示，遗传多态性可能影响个体对特定药物的反应。",
    "法律实践中，合同的成立需要具备要约、承诺和对价三个基本要素。",
    "司法解释指出，知识产权保护范围应结合具体行业标准进行认定。",
    "仲裁作为替代性纠纷解决机制，具有保密性强、效率较高的优势。",
    "国际私法领域，冲突规范的适用需遵循最密切联系原则。",
    "合规管理体系的建立有助于企业降低监管风险和声誉风险。",
    "量子计算利用量子叠加和纠缠原理，在特定问题上具备指数级加速潜力。",
    "新能源汽车的普及推动了锂电池技术和充电基础设施的快速发展。",
    "海洋碳汇研究表明，健康的海洋生态系统每年可吸收大量二氧化碳。",
    "数字孪生技术通过构建物理系统的虚拟映射，实现了智能化运维管理。",
    "航天工程的发展推动了材料科学、精密制造和通信技术的综合进步。",
    "城市化进程加快了基础设施建设需求，带动了建筑和材料行业的持续增长。",
    "气候变化政策推动各国加速能源结构转型，可再生能源装机容量快速增加。",
    "供应链管理的数字化转型提升了企业对市场波动的响应速度。",
    "人口老龄化趋势加大了医疗健康体系的运营压力，促使政策加快医疗改革步伐。",
    "教育技术的发展推动了个性化学习和自适应教育平台的广泛应用。",
]

_CORPUS_TEXT = " ".join(_CORPUS)

def build_haystack(target_chars):
    repeat = (target_chars // len(_CORPUS_TEXT)) + 2
    return (_CORPUS_TEXT * repeat)[:target_chars]

def insert_needle(haystack, needle, depth_percent):
    if depth_percent >= 100:
        return haystack + " " + needle
    insertion_point = int(len(haystack) * depth_percent / 100)
    boundary = haystack.rfind("。", max(0, insertion_point - 400), insertion_point + 1)
    if boundary != -1:
        boundary += 1
    else:
        boundary = insertion_point
    return haystack[:boundary] + " " + needle + " " + haystack[boundary:]

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

def query_model(model, context, question, base_url, max_tokens=256,
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

def score_response(pred, answer):
    pred_norm = re.sub(r"[\s\W]", "", pred).lower()
    ans_norm  = re.sub(r"[\s\W]", "", answer).lower()
    if ans_norm in pred_norm:
        return 10
    for token in re.findall(r"[一-鿿0-9a-z.]+", answer.lower()):
        if len(token) >= 2 and token in pred_norm:
            return 7
    return 1

def main():
    parser = argparse.ArgumentParser(description="Needle in a Haystack (Chinese, Ollama)")
    parser.add_argument("--model",          default="qwen2.5:7b")
    parser.add_argument("--base-url",       default=DEFAULT_BASE_URL)
    parser.add_argument("--context-lengths-tokens", type=int, nargs="+", default=None,
                        help="Context lengths in TOKENS. Defaults to per-model preset.")
    parser.add_argument("--depth-percents", type=float, nargs="+",
                        default=list(np.linspace(0, 100, 15).round(1)))
    parser.add_argument("--needle",         default=NEEDLE)
    parser.add_argument("--question",       default=QUESTION)
    parser.add_argument("--answer",         default=ANSWER)
    parser.add_argument("--max-tokens",     type=int, default=256)
    parser.add_argument("--timeout",        type=int, default=300)
    parser.add_argument("--no-plot",        action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[:/\\]", "_", args.model)

    ctx_tokens = args.context_lengths_tokens or MODEL_CTX_LENGTHS.get(
        args.model,
        [1000, 4000, 8000, 16000, 32000]
    )
    ctx_chars  = [int(t * CHARS_PER_TOKEN) for t in ctx_tokens]
    num_ctx    = MODEL_MAX_CTX.get(args.model, max(ctx_tokens))

    combos = [(ct, cc, dp)
              for ct, cc in zip(ctx_tokens, ctx_chars)
              for dp in args.depth_percents]

    print(f"\nModel    : {args.model}")
    print(f"Max ctx  : {num_ctx:,} tokens")
    print(f"Test ctx : {[f'{t//1000}K' for t in ctx_tokens]}")
    print(f"Depths   : {len(args.depth_percents)} levels")
    print(f"Combos   : {len(combos)}")
    print(f"Results  : {RESULTS_DIR}\n")

    results = []
    for ctx_tok, ctx_char, depth in tqdm(combos, desc=args.model, ncols=90):
        fname       = f"{safe_model}_tok{ctx_tok}_dep{depth:.1f}"
        result_file = RESULTS_DIR / f"{fname}_results.json"

        if result_file.exists():
            with open(result_file, encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        haystack = build_haystack(ctx_char - len(args.needle) - 200)
        context  = insert_needle(haystack, args.needle, depth)
        context  = context[:ctx_char]

        pred, latency = query_model(
            args.model, context, args.question,
            args.base_url, args.max_tokens, args.timeout,
            num_ctx=num_ctx,
        )
        s = score_response(pred, args.answer)

        result = {
            "model":                  args.model,
            "context_length_tokens":  ctx_tok,
            "context_length_chars":   ctx_char,
            "depth_percent":          depth,
            "version":                1,
            "needle":                 args.needle,
            "retrieval_question":     args.question,
            "model_response":         pred,
            "score":                  s,
            "test_duration_seconds":  round(latency, 2),
            "test_timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results.append(result)

    total   = len(results)
    perfect = sum(1 for r in results if r["score"] == 10)
    miss    = sum(1 for r in results if r["score"] == 1)
    print(f"\n{'='*55}")
    print(f"  Model  : {args.model}")
    print(f"  Total  : {total}")
    print(f"  Score 10: {perfect} ({perfect/total:.1%})")
    print(f"  Score  1: {miss}  ({miss/total:.1%})")
    print(f"{'='*55}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
