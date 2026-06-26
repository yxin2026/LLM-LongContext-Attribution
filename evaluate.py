#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test Evaluation Script
使用 Ollama（兼容 OpenAI API）对本地模型进行 PAC-Test 评测。
支持断点续跑：中途中断后重新运行同一命令会自动跳过已完成的样本。

用法：
    python evaluate.py --model qwen2.5:7b --subset all
    python evaluate.py --model deepseek-r1:7b --subset A --samples 100
    python evaluate.py --model qwen2.5:14b --subset A,B --base-url http://localhost:11434/v1
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from tqdm import tqdm

# ========== 配置 ==========

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY  = "ollama"
RESULTS_DIR      = Path(__file__).parent / "results"

# ========== Prompt 模板 ==========

SYSTEM_PROMPT = (
    "你是一个精确的信息提取助手。请根据给定的上下文文本，"
    "直接给出问题的答案，不要解释，不要复述问题，只输出答案本身。"
)

def build_user_prompt(context: str, question: str) -> str:
    return (
        f"请仔细阅读以下文本：\n\n{context}\n\n"
        f"问题：{question}\n\n"
        f"请直接给出答案（只输出答案，不要其他内容）："
    )

# ========== 评分函数 ==========

def strip_thinking(text: str) -> str:
    # strip reasoning-model thinking. handles <think>/<thinking> variants and
    # the malformed cases that leak otherwise:
    #   1) well-formed pairs  <think>...</think>
    #   2) orphan close </think> (open tag swallowed) -> answer is after last close
    #   3) orphan open <think>  (clipped before close) -> drop from open to end
    # answer for a reasoning model is whatever follows the final close tag.
    text = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', text,
                  flags=re.DOTALL | re.IGNORECASE)
    m = list(re.finditer(r'</think(?:ing)?>', text, flags=re.IGNORECASE))
    if m:                                   # orphan close: keep text after last one
        text = text[m[-1].end():]
    m = re.search(r'<think(?:ing)?>', text, flags=re.IGNORECASE)
    if m:                                   # orphan open (clipped): drop the rest
        text = text[:m.start()]
    return text.strip()

def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[，。！？；：、""''《》【】\s]',
                  '', text, flags=re.UNICODE)
    return text.lower()

def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

def contains_match(pred: str, gold: str) -> bool:
    return normalize(gold) in normalize(pred)

# ---------- NIAH 原版风格评分：LLM-as-judge (accuracy 1–10 rubric) ----------
# 复刻 gkamradt/LLMTest_NeedleInAHaystack 的 OpenAIEvaluator：
#   LangChain load_evaluator("labeled_score_string") + GPT-4，1–10 准确性 rubric。
# 离线适配：裁判换成固定的本地 Ollama 模型（默认 gemma4:31b，异家族中立强裁判），
# rubric / 1–10 / "Rating: [[N]]" 输出格式与原版一致；裁判全程固定以保证跨被测模型可比。

JUDGE_CRITERIA = (
    "评分标准（accuracy，仅看事实准确性，不看表述风格）：\n"
    "Score 1 : 回答与参考答案完全无关。\n"
    "Score 3 : 回答略有相关，但与参考答案不符。\n"
    "Score 5 : 回答中等相关，但包含事实错误。\n"
    "Score 7 : 回答与参考答案一致，但有少量遗漏。\n"
    "Score 10: 回答完全准确，与参考答案完全一致。"
)

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格、客观的评分员。请依据给定的【问题】【参考答案】，"
    "对【待评回答】的事实准确性打分。只看事实是否与参考答案一致，不看措辞与详略。"
)

def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return (
        f"{JUDGE_CRITERIA}\n\n"
        f"【问题】{question}\n"
        f"【参考答案】{reference}\n"
        f"【待评回答】{prediction or '(空)'}\n\n"
        f"请先用一句话给出理由，然后在最后一行严格按此格式输出分数："
        f"Rating: [[N]]（N 为 1 到 10 的整数）。"
    )

def _clamp_score(s: str) -> int:
    return max(1, min(10, int(round(float(s)))))

def parse_judge_score(text: str) -> Optional[int]:
    """从裁判输出里解析 1–10 分；解析失败返回 None。

    本地模型常不严格遵守 `[[N]]` 格式（实测会写成 `Rating: [10]`、`评分：8` 等），
    故按优先级容错匹配，最后兜底取文末 0–10 的数字。
    """
    text = strip_thinking(text)
    for pat in (r'\[\[\s*(\d+(?:\.\d+)?)\s*\]\]',          # [[N]]（原版格式）
                r'\[\s*(\d+(?:\.\d+)?)\s*\]',              # [N]
                r'(?:rating|score|评分|得分|打分)\D{0,6}(\d+(?:\.\d+)?)'):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _clamp_score(m.group(1))
    for tok in reversed(re.findall(r'\d+(?:\.\d+)?', text)):   # 兜底：文末数字
        if 0 <= float(tok) <= 10:
            return _clamp_score(tok)
    return None

def score_response(pred: str, gold: str,
                   question: Optional[str] = None,
                   judge: Optional["OllamaClient"] = None) -> Dict:
    pred_clean = strip_thinking(pred)
    out = {
        "em":       exact_match(pred_clean, gold),
        "contains": contains_match(pred_clean, gold),
        "pred":     pred_clean,
        "gold":     gold,
    }
    if judge is not None:                       # NIAH 主指标：1–10 裁判分
        s = judge.judge(question or "", gold, pred_clean)
        if s is not None:
            out["score"], out["score_src"] = s, "judge"
        else:                                   # 裁判调用/解析失败：退回 EM/Contains 三档
            out["score"] = 10 if out["em"] else (7 if out["contains"] else 1)
            out["score_src"] = "fallback"
        out["score_norm"] = round(out["score"] / 10.0, 3)
    return out

# ========== num_ctx 计算 ==========

CHARS_PER_TOKEN = 1.71      # 中文在 Qwen2.5 上约 1.71 字符/token（与 niah_*.py 一致）
NUM_CTX_OVERHEAD = 256      # prompt 模板 + 安全余量（token）

def compute_num_ctx(sample: Dict, max_tokens: int,
                    cap: Optional[int] = None) -> int:
    """按样本上下文长度自动推算 Ollama num_ctx。

    覆盖 = 上下文 token + 输出预留(max_tokens) + 模板余量，向上取整到 1024。
    v3 数据 total_length 单位为 tokens；v1/v2 为字符，按 1.71 折算。
    cap 为上限（token），None 表示不限制。
    """
    tl   = int(sample.get("total_length", 0) or 0)
    unit = sample.get("total_length_unit", "chars")
    ctx_tokens = tl if unit == "tokens" else int(round(tl / CHARS_PER_TOKEN))
    need = ctx_tokens + max_tokens + NUM_CTX_OVERHEAD
    need = ((need + 1023) // 1024) * 1024            # 向上取整到 1024
    if cap:
        need = min(need, cap)
    return max(need, 1024)


# ========== API 调用 ==========

class OllamaClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 120, max_retries: int = 3,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 num_ctx_cap: Optional[int] = None):
        self.base_url    = base_url.rstrip("/")
        self.headers     = {"Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"}
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.num_ctx_cap = num_ctx_cap
        # 评分请求走 Ollama 原生 /api/chat（OpenAI 兼容端点不支持 num_ctx）。
        # base_url 形如 http://host:port/v1 —— 去掉 /v1 得到原生根。
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.native_chat_url = root.rstrip("/") + "/api/chat"
        # 本地 Ollama 走直连：忽略环境里的 HTTP(S)_PROXY（如 Clash 127.0.0.1:7890），
        # 否则 localhost 请求会被代理截走、超时。
        self.session = requests.Session()
        self.session.trust_env = False

    def chat(self, context: str, question: str,
             num_ctx: Optional[int] = None) -> Tuple[str, float]:
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(context, question)},
            ],
            "stream":   False,
            "options":  options,
        }
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.session.post(
                    self.native_chat_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                latency = time.time() - t0
                resp.raise_for_status()
                data = resp.json()
                text = data["message"]["content"].strip()
                return text, latency
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    return "[TIMEOUT]", float(self.timeout)
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"[ERROR: {e}]", 0.0
                time.sleep(2 ** attempt)
        return "[FAILED]", 0.0

    def judge(self, question: str, reference: str,
              prediction: str) -> Optional[int]:
        """NIAH 原版风格 1–10 打分（self.model 作裁判）。解析失败返回 None。"""
        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_judge_prompt(question, reference, prediction)},
            ],
            "stream":  False,
            # think=False：思考型裁判（如 gemma4:31b）默认把推理塞进独立 thinking 字段，
            # 256 token 预算会耗在思考上、content 为空导致解析失败 → 关掉思考直接给分。
            # 对非思考模型该参数无害。
            "think":   False,
            "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 2048},
        }
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(self.native_chat_url, headers=self.headers,
                                     json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return parse_judge_score(resp.json()["message"]["content"])
            except Exception:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

    def test_connection(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/models",
                                headers=self.headers, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self.session.get(f"{self.base_url}/models",
                                headers=self.headers, timeout=5)
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []

# ========== 数据加载 ==========

def load_subset(data_dir: Path, subset: str,
                max_samples: Optional[int] = None) -> List[Dict]:
    fpath = data_dir / f"subset_{subset}.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(f"找不到数据文件: {fpath}")
    samples = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]
    return samples

# ========== 断点续跑：checkpoint 管理 ==========

def get_checkpoint_path(out_dir: Path, safe_model: str, subset: str) -> Path:
    return out_dir / f"{safe_model}_subset{subset}_checkpoint.jsonl"

def load_checkpoint(ckpt_path: Path) -> Dict[str, Dict]:
    """返回 {sample_id: record} 的已完成记录"""
    done = {}
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["sample_id"]] = rec
    return done

def append_checkpoint(ckpt_path: Path, record: Dict):
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ========== 单个子集评测 ==========

def evaluate_subset(client: OllamaClient,
                    samples: List[Dict],
                    subset: str,
                    ckpt_path: Path,
                    desc: str = "",
                    judge: Optional[OllamaClient] = None) -> Dict:
    # 加载已完成的记录
    done = load_checkpoint(ckpt_path)
    skipped = len(done)
    if skipped:
        print(f"  断点续跑：跳过已完成的 {skipped} 条")

    results = list(done.values())
    todo = [s for s in samples if s["sample_id"] not in done]

    for sample in tqdm(todo, desc=desc or f"Subset {subset}",
                       ncols=90, initial=skipped, total=len(samples)):
        context  = sample["context"]
        question = sample["question"]
        gold     = sample["answer"]

        num_ctx = compute_num_ctx(sample, client.max_tokens, client.num_ctx_cap)
        pred, latency = client.chat(context, question, num_ctx=num_ctx)
        scores = score_response(pred, gold, question=question, judge=judge)

        rec = {
            "sample_id": sample["sample_id"],
            "subset":    subset,
            "num_ctx":   num_ctx,
            "latency":   round(latency, 2),
            **scores,
        }
        for key in ["domain", "total_length", "total_length_unit", "position_ratio",
                    "dilution_type", "dilution_ratio", "noise_density",
                    "similarity_level", "distance_level",
                    "chain_type", "num_hops"]:
            if key in sample:
                rec[key] = sample[key]

        results.append(rec)
        append_checkpoint(ckpt_path, rec)

    n = len(results)
    em_total       = sum(r["em"]       for r in results)
    contains_total = sum(r["contains"] for r in results)
    avg_latency    = sum(r["latency"]  for r in results) / n if n else 0

    out = {
        "subset":        subset,
        "n":             n,
        "em":            round(em_total / n, 4) if n else 0,
        "contains":      round(contains_total / n, 4) if n else 0,
        "avg_latency_s": round(avg_latency, 2),
        "details":       results,
    }
    # NIAH 主指标（有裁判分时）：平均 1–10 分、归一化分、满分(=10)率
    scored = [r for r in results if "score" in r]
    if scored:
        m = len(scored)
        out["mean_score"] = round(sum(r["score"]      for r in scored) / m, 3)
        out["score_norm"] = round(sum(r["score_norm"] for r in scored) / m, 4)
        out["acc10"]      = round(sum(1 for r in scored if r["score"] == 10) / m, 4)
    return out

# ========== 分层分析 ==========

def primary_norm(r: Dict) -> float:
    """归一化主指标 [0,1]：有 NIAH 裁判分时用 score_norm，否则退回 EM。"""
    if "score_norm" in r:
        return r["score_norm"]
    return 1.0 if r["em"] else 0.0

def breakdown(results: List[Dict], group_keys: List[str]) -> Dict:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = tuple(str(r.get(k, "N/A")) for k in group_keys)
        groups[key].append(primary_norm(r))
    out = {}
    for k, vals in sorted(groups.items()):
        label = " | ".join(f"{gk}={v}" for gk, v in zip(group_keys, k))
        out[label] = {"n": len(vals), "val": round(sum(vals)/len(vals), 4)}
    return out

# ========== 保存最终结果 ==========

def save_results(all_results: Dict, model_name: str,
                 out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r'[:/\\]', '_', model_name)
    base = out_dir / f"{safe_model}_{ts}"

    with open(f"{base}_details.jsonl", "w", encoding="utf-8") as f:
        for res in all_results.values():
            for r in res["details"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
               for k, v in all_results.items()}
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存：\n  {base}_details.jsonl\n  {base}_summary.json")
    return base

# ========== 控制台报告 ==========

def print_report(all_results: Dict, model_name: str):
    print("\n" + "=" * 65)
    print(f"  模型：{model_name}")
    print(f"  时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    overall_norm = []
    for subset, res in all_results.items():
        n   = res["n"]
        em  = res["em"]
        cnt = res["contains"]
        lat = res["avg_latency_s"]
        print(f"\n── Subset {subset} ────────────────────────────────")
        if "mean_score" in res:                 # NIAH 主指标
            print(f"   样本数:{n}  NIAH得分:{res['mean_score']:.2f}/10"
                  f"  满分率:{res['acc10']:.1%}  (EM:{em:.1%} Contains:{cnt:.1%})"
                  f"  avg延迟:{lat}s")
        else:
            print(f"   样本数:{n}  EM:{em:.1%}  Contains:{cnt:.1%}  avg延迟:{lat}s")
        overall_norm.extend([primary_norm(r) for r in res["details"]])

        details = res["details"]
        if subset == "A":
            print("   [按位置]")
            for k, v in breakdown(details, ["position_ratio"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
            print("   [按上下文长度]")
            for k, v in breakdown(details, ["total_length"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
        elif subset == "B":
            print("   [按干扰类型]")
            for k, v in breakdown(details, ["dilution_type"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
            # v3 数据用 noise_density；v1 用 dilution_ratio
            density_key = ("noise_density"
                           if any("noise_density" in r for r in details)
                           else "dilution_ratio")
            print(f"   [按干扰密度 ({density_key})]")
            for k, v in breakdown(details, [density_key]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
        elif subset == "C":
            print("   [按实体相似度]")
            for k, v in breakdown(details, ["similarity_level"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
            print("   [按距离]")
            for k, v in breakdown(details, ["distance_level"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
        elif subset == "D":
            print("   [按推理跳数]")
            for k, v in breakdown(details, ["num_hops"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")
            print("   [按段间距]")
            for k, v in breakdown(details, ["distance_level"]).items():
                print(f"     {k}: n={v['n']}  得分={v['val']:.1%}")

    if overall_norm:
        print("\n── 总体 ────────────────────────────────────────")
        total_n = sum(res["n"] for res in all_results.values())
        overall = sum(overall_norm) / len(overall_norm)
        scored  = any("mean_score" in res for res in all_results.values())
        label   = "Overall NIAH得分(归一)" if scored else "Overall EM"
        print(f"   总样本数:{total_n}  {label}:{overall:.1%}")
    print("=" * 65)

# ========== 可视化 ==========

def plot_results(all_results: Dict, model_name: str, out_base: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        return

    subset_labels = {"A": "位置效应", "B": "干扰稀释",
                     "C": "信息覆盖", "D": "多跳衰减"}
    colors = {"A": "steelblue", "B": "coral", "C": "seagreen", "D": "mediumpurple"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PAC-Test 评测结果 — {model_name}", fontsize=14)

    for subset, ax in zip(["A", "B", "C", "D"], axes.flat):
        if subset not in all_results:
            ax.set_visible(False)
            continue
        res     = all_results[subset]
        details = res["details"]
        color   = colors[subset]

        if subset == "A":
            grp    = breakdown(details, ["position_ratio"])
            labels = [k.split("=")[1] for k in grp]
            vals   = [v["val"] for v in grp.values()]
            ax.set_xlabel("位置比例")
        elif subset == "B":
            grp    = breakdown(details, ["dilution_type"])
            labels = [k.split("=")[1][:10] for k in grp]
            vals   = [v["val"] for v in grp.values()]
            ax.set_xlabel("干扰类型")
        elif subset == "C":
            grp    = breakdown(details, ["similarity_level"])
            labels = [k.split("=")[1][:12] for k in grp]
            vals   = [v["val"] for v in grp.values()]
            ax.set_xlabel("实体相似度")
        elif subset == "D":
            grp    = breakdown(details, ["num_hops"])
            labels = [k.split("=")[1] for k in grp]
            vals   = [v["val"] for v in grp.values()]
            ax.set_xlabel("推理跳数")

        scored   = "score_norm" in res
        avg_val  = res["score_norm"] if scored else res["em"]
        y_label  = "NIAH得分 (归一 0-1)" if scored else "Exact Match"
        ax.bar(labels, vals, color=color)
        ax.set_title(f"Subset {subset}：{subset_labels.get(subset,'')}")
        ax.set_ylabel(y_label)
        ax.set_ylim(0, 1.1)
        ax.axhline(avg_val, color="red", linestyle="--", alpha=0.7,
                   label=f"平均 {avg_val:.1%}")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    fig_path = f"{out_base}_chart.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"图表已保存: {fig_path}")
    plt.close()

# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="PAC-Test 本地模型评测（支持断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",       default="qwen2.5:7b")
    parser.add_argument("--subset",      default="all",
                        help="A/B/C/D 或逗号分隔，或 all")
    parser.add_argument("--samples",     type=int, default=None,
                        help="每子集最多样本数（默认全量）")
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens",  type=int, default=1024)
    parser.add_argument("--num-ctx",     type=int, default=None,
                        help="num_ctx 上限(token)，防止超长样本撑爆显存；"
                             "默认不限，按样本 total_length+输出预留自动设")
    parser.add_argument("--timeout",     type=int, default=120)
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--judge-model", default="gemma4:31b",
                        help="NIAH 风格 1–10 评分的裁判模型（固定，保证跨模型可比）。"
                             "原版用 GPT-4；离线默认用本地 gemma4:31b——异家族中立强裁判，"
                             "使三个被测模型(qwen2.5/deepseek-r1/qwen3.5)全部异模型评判。")
    parser.add_argument("--no-judge",    action="store_true",
                        help="关闭 LLM 裁判，仅用 EM/Contains（旧行为）。")
    args = parser.parse_args()

    client = OllamaClient(
        base_url=args.base_url,
        api_key=DEFAULT_API_KEY,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_ctx_cap=args.num_ctx,
    )

    # NIAH 原版风格裁判（gkamradt OpenAIEvaluator 复刻）；固定裁判模型。
    judge = None
    if not args.no_judge:
        judge = OllamaClient(
            base_url=args.base_url, api_key=DEFAULT_API_KEY,
            model=args.judge_model, timeout=args.timeout,
            temperature=0.0, max_tokens=256,
        )

    if args.list_models:
        models = client.list_models()
        print("可用模型：" if models else "无法获取模型列表，请确认 ollama serve 已启动")
        for m in models:
            print(f"  {m}")
        return

    print(f"\n正在连接 {args.base_url} ...")
    if not client.test_connection():
        print("无法连接 Ollama，请先运行：ollama serve")
        sys.exit(1)
    print(f"连接成功，模型：{args.model}")
    if judge is not None:
        print(f"评分方式：NIAH 风格 LLM 裁判 1–10（裁判模型：{args.judge_model}）"
              f"；EM/Contains 作次要诊断")
    else:
        print("评分方式：EM/Contains（已关闭 LLM 裁判）")

    subsets = (["A", "B", "C", "D"] if args.subset.lower() == "all"
               else [s.strip().upper() for s in args.subset.split(",")])

    data_dir   = Path(__file__).parent / "data"
    safe_model = re.sub(r'[:/\\]', '_', args.model)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for subset in subsets:
        print(f"\n{'='*55}\n加载 Subset {subset}...")
        try:
            samples = load_subset(data_dir, subset, args.samples)
        except FileNotFoundError as e:
            print(f"  {e}，跳过")
            continue
        print(f"  共 {len(samples)} 条样本")

        ckpt_path = get_checkpoint_path(RESULTS_DIR, safe_model, subset)
        res = evaluate_subset(
            client, samples, subset, ckpt_path,
            desc=f"  Subset {subset} [{args.model}]",
            judge=judge,
        )
        all_results[subset] = res
        if "mean_score" in res:
            print(f"  完成 NIAH得分={res['mean_score']:.2f}/10  满分率={res['acc10']:.1%}"
                  f"  (EM={res['em']:.1%} Contains={res['contains']:.1%})"
                  f"  avg延迟={res['avg_latency_s']}s")
        else:
            print(f"  完成 EM={res['em']:.1%}  Contains={res['contains']:.1%}"
                  f"  avg延迟={res['avg_latency_s']}s")

    if not all_results:
        print("没有有效结果，退出。")
        sys.exit(1)

    print_report(all_results, args.model)
    out_base = save_results(all_results, args.model, RESULTS_DIR, ts)

    if not args.no_plot:
        plot_results(all_results, args.model, out_base)

    # 清理 checkpoint 文件
    for subset in all_results:
        ckpt = get_checkpoint_path(RESULTS_DIR, safe_model, subset)
        if ckpt.exists():
            ckpt.unlink()
            print(f"已清理 checkpoint: {ckpt.name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
