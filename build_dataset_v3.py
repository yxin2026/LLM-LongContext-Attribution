#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test Dataset Builder v3.2 (constraint-strict, diversified filler)

Generates data/subset_{A,B,C,D}.jsonl with **lengths measured in tokens**
using the real Qwen2.5 tokenizer (HuggingFace AutoTokenizer).

Must be run with a Python that has `transformers` + `tokenizers` installed.
On this machine, use the "Dl" conda env:
    D:\\Software\\Environment\\anaconda3\\envs\\Dl\\python.exe build_dataset_v3.py

Compliance with 实验设计/测试集选型.docx:
  - Subset A: 4 lengths × 5 positions(10/25/50/75/90%) × 50 facts = 1000
              + fixed 500-token prefix at start of every sample
  - Subset B: 5 noise densities(0/25/50/75/90%) × 3 types × 3 lengths × 40 = 1800
              needle at the very beginning, distractor density controls
              non-needle composition
  - Subset C: 4 sim levels × 3 distances × 50 samples = 600
              length sampled per cell from {8K,16K,32K}; sim classes
              capped to 5 pair groups each for balance
  - Subset D: 3 hops(2/3/4) × 3 distances × 4 lengths × 40 = 1440
              4-hop chains read from fact_chains.json if present;
              if KB has no 4-hop chains, those cells are skipped and
              the script reports the deficit.

Inter-hop distances for Subset D follow design (1K/4K/8K tokens).
Infeasible combinations (needles + (hops-1)*gap > length) are skipped.
"""

import json
import argparse
import random
import string
import sys
from collections import Counter
from pathlib import Path

random.seed(42)

# ==================== load real tokenizer ====================

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.stderr.write(
        "ERROR: transformers not installed in this Python.\n"
        "Use the Dl conda env, e.g.:\n"
        "  D:\\Software\\Environment\\anaconda3\\envs\\Dl\\python.exe build_dataset_v3.py\n"
    )
    sys.exit(1)

print("Loading Qwen2.5 tokenizer …", flush=True)
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(f"  vocab_size={TOKENIZER.vocab_size}", flush=True)


def n_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def trim_to_tokens(text: str, target: int) -> str:
    encoded = TOKENIZER(text, add_special_tokens=False,
                        return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) <= target:
        return text
    # Cut the original string at a tokenizer boundary. Decoding ids[:target]
    # is not safe here: normalization of whitespace can make the decoded text
    # re-tokenize to fewer than `target` tokens.
    cut_char = encoded["offset_mapping"][target - 1][1]
    candidate = text[:cut_char]
    current = n_tokens(candidate)
    while current > target and candidate:
        candidate = candidate[:-1]
        current = n_tokens(candidate)
    if current < target:
        # Chinese full-width commas are stable single tokens in Qwen2.5. This
        # fallback is normally only needed for a tokenizer boundary edge case.
        candidate += "，" * (target - current)
    final = n_tokens(candidate)
    if final != target:
        raise RuntimeError(f"unable to trim exactly: wanted {target}, got {final}")
    return candidate

# ==================== filler / distractor pools ====================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KB_DIR   = BASE_DIR / "core_knowledge"
TEMPLATE_DIR = BASE_DIR / "templates"

FILLER_SEGMENTS = {
    "computer_science": [
        "近年来，深度学习在自然语言处理领域取得了突破性进展。基于Transformer架构的预训练语言模型展现出强大的文本理解和生成能力。",
        "研究表明，注意力机制能够有效捕捉长距离依赖关系，这对于理解复杂语义至关重要。多头注意力通过并行计算不同子空间的表示，进一步增强了模型的表达能力。",
        "从工程实践角度，大模型的部署需要考虑计算效率、内存占用和推理延迟等因素。模型压缩技术如知识蒸馏和量化剪枝，能够在保持性能的同时显著降低资源消耗。",
        "在计算机视觉领域，卷积神经网络通过局部连接和权值共享，有效提取图像的层次化特征。残差连接解决了深层网络的梯度消失问题，使得网络可以扩展到上百层。",
        "联邦学习作为一种分布式机器学习范式，允许多个参与方在不共享原始数据的前提下协作训练模型。这为数据隐私保护提供了有效的技术方案。",
        "对比学习通过构造正负样本对，学习更好的表示空间。自监督预训练利用大规模无标注数据，显著降低了对标注数据的依赖。",
        "图神经网络专门处理图结构数据，通过消息传递机制聚合邻居信息。在社交网络分析、分子性质预测等任务上表现出色。",
        "多模态学习旨在融合文本、图像、音频等多种模态的信息。视觉-语言预训练模型能够理解图像内容并生成相关描述。",
        "强化学习通过与环境的交互学习最优策略。深度强化学习将神经网络与强化学习结合，在游戏AI和机器人控制中取得显著成果。",
        "元学习关注如何快速适应新任务，使模型具备举一反三的能力。少样本学习在仅有少量标注样本的情况下也能取得不错的效果。",
    ],
    "medicine": [
        "临床研究表明，免疫检查点抑制剂在非小细胞肺癌治疗中显示出良好疗效。PD-1/PD-L1通路阻断能够激活T细胞对肿瘤的免疫应答。",
        "从病理机制来看，CAR-T细胞疗法通过基因工程改造患者自身的T细胞，使其能够识别并攻击特定的肿瘤细胞。",
        "多项随机对照试验证实，mRNA疫苗技术在传染病防控中展现出快速响应的优势。脂质纳米颗粒递送系统有效保护了mRNA的稳定性。",
        "在药物代谢方面，肝脏是主要的代谢器官。细胞色素P450酶系参与了大多数药物的氧化代谢过程。",
        "基因组学研究提示，BRCA基因突变与乳腺癌和卵巢癌的发病风险显著相关。基因检测为高危人群的筛查提供了重要依据。",
        "精准医疗根据患者的基因型、表型和环境因素，制定个体化的治疗方案。药物基因组学指导下的用药能够提高疗效并减少不良反应。",
        "医学影像AI技术能够辅助放射科医生进行病灶检测和诊断。深度学习模型在肺结节筛查、眼底病变识别等任务上达到专家水平。",
        "再生医学利用干细胞技术和组织工程方法，修复或替代受损的组织和器官。3D生物打印技术为构建复杂组织提供了新的可能。",
        "流行病学调查表明，慢性非传染性疾病已成为全球主要的疾病负担。生活方式干预是预防心血管疾病和糖尿病的重要措施。",
        "抗生素耐药性是当今世界面临的重大公共卫生挑战。合理使用抗生素、开发新型抗菌药物是应对耐药性的关键策略。",
    ],
    "law": [
        "从立法目的来看，知识产权保护法旨在激励创新和促进知识传播。著作权、专利权和商标权构成了知识产权保护的核心内容。",
        "司法实践中，合同纠纷的审理需要考虑合同的订立、履行和解除等环节。诚实信用原则是合同法的基本原则之一。",
        "根据相关司法解释，侵权责任的认定通常采用过错责任原则。但在某些特殊领域，法律规定了无过错责任或过错推定责任。",
        "比较法研究表明，大陆法系和英美法系在诉讼程序、证据规则和裁判方式上存在显著差异。法律移植需要结合本土实际进行改造。",
        "从法学理论角度，自然法学派强调法律的道德基础和正义价值。实证主义法学派则主张将法律与道德分离，专注于法律规范的逻辑分析。",
        "劳动法律制度致力于保护劳动者的合法权益，同时兼顾用人单位的用工自主权。劳动合同的订立应当遵循合法、公平、平等自愿的原则。",
        "环境保护法通过确立环境保护的基本原则和制度，防治环境污染和生态破坏。污染者付费原则是环境法的重要经济手段。",
        "数据安全法和个人信息保护法的实施，为数字经济时代的数据治理提供了法律框架。企业应当建立健全数据分类分级保护制度。",
        "反垄断法旨在维护市场竞争秩序，防止和制止垄断行为。经营者集中审查是反垄断执法的重要组成部分。",
        "国际私法解决涉外民事关系的法律适用问题。冲突规范通过连结点的指引，确定应当适用的准据法。",
    ],
    "finance": [
        "从宏观经济角度，货币政策通过调节利率和货币供应量影响经济运行。中央银行运用公开市场操作、存款准备金率和再贴现率等工具实施货币政策。",
        "市场数据显示，股票市场的波动性受到多种因素影响。投资者情绪、宏观经济数据和政策变化都可能引发市场的剧烈波动。",
        "投资组合理论表明，分散投资能够有效降低非系统性风险。资产配置需要根据投资者的风险偏好和投资期限进行动态调整。",
        "风险管理实践中，价值在险模型被广泛用于度量市场风险。压力测试则评估极端情景下金融机构的承受能力。",
        "从估值模型来看，现金流折现模型适用于稳定盈利的企业估值。相对估值法通过比较同类公司的估值倍数进行定价。",
        "行为金融学研究发现，投资者在决策过程中普遍存在认知偏差。过度自信、损失厌恶和羊群效应是影响投资行为的常见心理因素。",
        "监管政策方面，巴塞尔协议为国际银行业监管提供了统一标准。资本充足率要求和流动性覆盖率是核心监管指标。",
        "金融科技的发展深刻改变了传统金融服务模式。移动支付、智能投顾和区块链技术正在重塑金融业态。",
        "国际收支平衡反映了一国对外经济交往的总体状况。经常账户和资本金融账户是国际收支的两大组成部分。",
        "汇率制度的选择需要在货币政策独立性和汇率稳定性之间进行权衡。浮动汇率制允许汇率根据市场供求自由波动。",
    ],
    "education": [
        "从学习理论角度，建构主义认为知识是学习者主动建构的，而非被动接受的。情境学习理论强调知识应当在真实情境中习得。",
        "教学实践表明，项目式学习能够有效培养学生的综合能力和创新思维。学生通过解决真实问题，将所学知识应用于实践。",
        "形成性评价在教学过程中持续收集学生的学习信息，为教学调整提供依据。总结性评价则在教学结束后对学生的学习成果进行鉴定。",
        "差异化教学要求教师根据学生的能力水平、学习风格和兴趣特点，提供适宜的学习内容和活动。",
        "教育技术为教学创新提供了工具和平台。在线学习打破了时空限制，使优质教育资源得以更广泛地传播。",
        "教师专业发展是一个持续的学习过程。反思性实践帮助教师不断改进教学策略，提升教学效果。",
        "核心素养框架从文化基础、自主发展和社会参与三个维度界定了学生应当具备的关键能力和必备品格。",
        "STEM教育强调科学、技术、工程和数学的跨学科整合。项目式学习和问题解决是STEM教学的核心方法。",
        "翻转课堂将知识传授移至课前，课堂时间用于深化理解和协作探究。这种模式提高了课堂互动的效率。",
        "教育公平是社会公平的重要基础。教育资源的均衡配置和弱势群体的补偿机制，是实现教育公平的重要保障。",
    ],
    "general": [
        "春天来了，大地回春，万物复苏。公园里的樱花竞相开放，吸引了许多市民前来观赏拍照。",
        "这家餐厅的菜品种类丰富，口味独特。招牌菜红烧肉的火候恰到好处，肥而不腻。",
        "阅读是获取知识的重要途径。每天抽出时间阅读，不仅能够增长见识，还能陶冶情操。",
        "运动是保持健康的关键。无论是跑步、游泳还是瑜伽，坚持锻炼都能增强体质。",
        "旅行能够开阔视野，体验不同的风土人情。计划一次说走就走的旅行，给生活增添色彩。",
        "音乐有着独特的治愈力量。无论是古典音乐还是流行歌曲，都能在忙碌的生活中带来片刻宁静。",
        "环保是每个人的责任。减少使用一次性塑料制品，选择绿色出行方式，从点滴做起保护地球。",
        "科技的发展改变了人们的生活方式。智能手机让沟通更加便捷，也让信息获取变得前所未有的容易。",
        "家庭教育对孩子的成长至关重要。父母的言传身教和良好的家庭氛围，是孩子健康成长的基石。",
        "时间管理是提高工作效率的关键。合理规划每天的时间，分清轻重缓急，才能在有限的时间内完成更多的事情。",
    ],
}

NOISE_WORDS   = ["xyz", "abc", "qwe", "mnp", "rst", "uvw", "ijk", "lmn", "opq", "def"]
NOISE_NUMBERS = ["123", "456", "789", "321", "654", "987", "111", "222"]
NOISE_SYMBOLS = ["@#$", "%^&", "*()", "!@#", "<>?", "{}|"]

# The original v3 builder repeated ten hand-written passages until the target
# length was reached.  At 32K/64K this produced hundreds of verbatim repeats.
# Reuse the repository's compositional templates to build a substantially
# larger deterministic pool while retaining the hand-written passages as
# high-quality seed material.
GENERAL_TEMPLATES = [
    "{season}时节，{location}的{scene}呈现出新的变化。人们在遵守公共秩序的同时，进行{activity}。",
    "社区近期围绕{topic}组织了交流活动。参与者分享实际经验，并讨论如何改进{aspect}。",
    "一份日常观察记录提到，{location}在{time}的人流较为平稳。管理人员随后检查了{facility}。",
    "为了完成既定安排，工作人员先确认{item}，再记录{result}。整个过程按照常规流程进行。",
    "在一次公开讲座中，主讲人介绍了{topic}的基本概念，并用生活案例说明{aspect}的重要性。",
    "居民可以通过{method}了解最新通知。若遇到疑问，可向服务人员核实具体安排。",
]
GENERAL_VARIABLES = {
    "season": ["春季", "夏季", "秋季", "冬季"],
    "location": ["城市公园", "社区图书馆", "公共广场", "文化中心", "街区服务站", "郊外步道"],
    "scene": ["植物景观", "公共空间", "活动区域", "步行环境", "服务设施"],
    "activity": ["散步", "阅读", "参观", "交流", "志愿服务", "体育锻炼"],
    "topic": ["时间管理", "健康生活", "公共安全", "环境保护", "阅读方法", "出行规划"],
    "aspect": ["执行步骤", "信息核验", "资源利用", "沟通方式", "日常习惯"],
    "time": ["工作日上午", "周末午后", "傍晚时段", "节假日前夕"],
    "facility": ["照明设施", "指引标识", "借阅设备", "休息区域", "消防通道"],
    "item": ["活动名单", "设备状态", "开放时间", "材料数量", "场地安排"],
    "result": ["检查结果", "反馈意见", "完成进度", "后续事项"],
    "method": ["公告栏", "官方网站", "服务热线", "现场咨询", "电子邮件"],
}


def _render_template(template: str, variables: dict, rng: random.Random) -> str:
    """Fill one template without inventing or evaluating field expressions."""
    values = {}
    for _, field, _, _ in string.Formatter().parse(template):
        if field:
            choices = variables.get(field, [field])
            values[field] = rng.choice(choices)
    return template.format(**values)


def _build_filler_pool(domain: str, pool_size: int = 1024):
    """Create a reproducible, deduplicated paragraph pool for one domain."""
    rng = random.Random(f"PAC-v3-natural-filler:{domain}")
    pool = list(FILLER_SEGMENTS.get(domain, FILLER_SEGMENTS["general"]))
    if domain == "general":
        # Remove the conspicuous sentence that dominated the old fixed prefix.
        pool = [p for p in pool if not p.startswith("春天来了，大地回春")]

    if domain == "general":
        templates = GENERAL_TEMPLATES
        variables = GENERAL_VARIABLES
    else:
        spec = json.load(open(TEMPLATE_DIR / "filler_templates.json", encoding="utf-8"))
        templates = spec["templates"][domain]
        variables = spec["variables"][domain]

    seen = set(pool)
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 20:
        paragraph = _render_template(rng.choice(templates), variables, rng)
        attempts += 1
        if paragraph not in seen:
            seen.add(paragraph)
            pool.append(paragraph)
    return pool

# Precompute token counts for each filler segment to avoid repeated tokenization
print("Precomputing segment token counts …", flush=True)
SEG_WITH_TOK = {
    dom: [(seg, n_tokens(seg)) for seg in _build_filler_pool(dom)]
    for dom in FILLER_SEGMENTS
}
print("  " + ", ".join(f"{d}={len(v)}" for d, v in SEG_WITH_TOK.items()), flush=True)


def gen_filler_tokens(domain: str, target_tokens: int, rng=None) -> str:
    """Generate diverse paragraph filler with an exact token budget."""
    if target_tokens <= 0:
        return ""
    rng = rng or random
    pool = SEG_WITH_TOK.get(domain, SEG_WITH_TOK["general"])
    out = []
    approx = 0
    previous = None
    # Generate from estimates, then measure the assembled paragraphs. Token
    # counts are not additive across paragraph boundaries.
    while approx < target_tokens:
        seg, n = rng.choice(pool)
        while seg == previous and len(pool) > 1:
            seg, n = rng.choice(pool)
        out.append(seg)
        previous = seg
        approx += n
    text = "\n\n".join(out)
    while n_tokens(text) < target_tokens:
        seg, _ = rng.choice(pool)
        while seg == previous and len(pool) > 1:
            seg, _ = rng.choice(pool)
        out.append(seg)
        previous = seg
        text = "\n\n".join(out)
    return trim_to_tokens(text, target_tokens)


def gen_distractor_tokens(noise_type: str, domain: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    if noise_type == "random_noise":
        out = []
        approx = 0
        # Use char heuristic: 1 char ≈ 0.5 tok for ASCII noise; overshoot then trim
        target_chars_approx = target_tokens * 4
        chars_so_far = 0
        while chars_so_far < target_chars_approx:
            tok = (f"{random.choice(NOISE_SYMBOLS)} {random.choice(NOISE_WORDS)} "
                   f"{random.choice(NOISE_NUMBERS)} {random.choice(NOISE_WORDS)} ")
            out.append(tok)
            chars_so_far += len(tok)
        return trim_to_tokens("\n".join(out), target_tokens)
    if noise_type == "out_domain_unrelated":
        # Use a real but different discipline. Generic G blocks use the
        # neutral `general` pool, so D and G remain experimentally distinct.
        out_domain = {
            "computer_science": "medicine",
            "medicine": "law",
            "law": "finance",
            "finance": "education",
            "education": "computer_science",
        }.get(domain, "computer_science")
        return gen_filler_tokens(out_domain, target_tokens)
    return gen_filler_tokens(domain, target_tokens)


def pad_to_token_length(text: str, target_tokens: int, domain: str = "general") -> str:
    # Re-check after concatenation because BPE token counts are not additive:
    # the boundary between `text` and `extra` can merge tokens.
    for _ in range(8):
        cur = n_tokens(text)
        if cur == target_tokens:
            return text
        if cur > target_tokens:
            return trim_to_tokens(text, target_tokens)
        extra = gen_filler_tokens(domain, target_tokens - cur)
        text = text + "\n\n" + extra

    # Stable one-token fallback for an exceptionally stubborn boundary case.
    cur = n_tokens(text)
    if cur < target_tokens:
        text += "，" * (target_tokens - cur)
    return trim_to_tokens(text, target_tokens)


def assemble_context(parts, target_tokens: int, domain: str = "general") -> str:
    """Join logical sections at paragraph boundaries and enforce exact length."""
    return pad_to_token_length("\n\n".join(p for p in parts if p),
                               target_tokens, domain)


def exact_gap_filler(left: str, right: str, domain: str,
                     target_gap: int, rng=None) -> str:
    """Return filler so end(left) -> start(right) is exactly target_gap tokens."""
    rng = rng or random
    budget = max(0, target_gap - n_tokens("\n\n\n\n"))
    for _ in range(12):
        filler = gen_filler_tokens(domain, budget, rng=rng)
        actual = n_tokens("\n\n" + filler + "\n\n")
        if actual == target_gap:
            return filler
        budget = max(0, budget + target_gap - actual)
    raise RuntimeError(f"cannot construct exact {target_gap}-token gap")


def finish_with_tail(head: str, target_tokens: int, domain: str,
                     rng=None) -> str:
    """Complete a context by changing only trailing non-critical filler."""
    rng = rng or random
    budget = max(0, target_tokens - n_tokens(head) - 2)
    for _ in range(12):
        tail = gen_filler_tokens(domain, budget, rng=rng) if budget else ""
        context = head + ("\n\n" + tail if tail else "")
        actual = n_tokens(context)
        if actual == target_tokens:
            return context
        budget = max(0, budget + target_tokens - actual)
    raise RuntimeError(f"cannot finish context at {target_tokens} tokens")


# ==================== fixed 500-token prefix for Subset A ====================

def make_fixed_prefix(target_tokens: int = 500) -> str:
    base = (
        "以下文档为长上下文理解能力评测任务提供的参考资料汇编。"
        "本资料涵盖了计算机科学、医学、法律、金融及教育等五个学科领域的代表性研究成果与基础知识。"
        "请仔细阅读以下全部内容，并根据文档信息回答末尾给出的问题。"
    )
    # The 500-token prefix is required by the experiment.  Fill it with a
    # deterministic but diverse neutral briefing instead of one repeated
    # spring/flower sentence.
    remaining = target_tokens - n_tokens(base + "\n\n")
    filler = gen_filler_tokens("general", remaining,
                               rng=random.Random("PAC-fixed-prefix-v2"))
    return pad_to_token_length(base + "\n\n" + filler, target_tokens, "general")


print("Building fixed 500-token prefix for Subset A …", flush=True)
FIXED_PREFIX_500 = make_fixed_prefix(500)
assert n_tokens(FIXED_PREFIX_500) == 500
print(f"  prefix tokens = {n_tokens(FIXED_PREFIX_500)}", flush=True)


# ==================== Subset A : 位置效应 ====================

SUBSET_A_LENGTHS_TOK = [4000, 8000, 16000, 32000]
SUBSET_A_POSITIONS   = [0.10, 0.25, 0.50, 0.75, 0.90]


def build_subset_a(facts, samples_per_config: int = 10):
    """4 length × 5 position × 5 domain × 10 facts = 1000.
    Layout per sample (tokens):
        [FIXED_PREFIX_500] [pre_filler] [needle] [post_filler]
    with pre_filler / post_filler sized so the needle sits at position p of
    the post-prefix body. Total tokens == declared length.
    """
    print("\nBuilding Subset A …", flush=True)
    dataset = []
    domains = sorted({f["domain"] for f in facts})
    sid = 0
    for domain in domains:
        domain_facts = [f for f in facts if f["domain"] == domain]
        for length in SUBSET_A_LENGTHS_TOK:
            for pos in SUBSET_A_POSITIONS:
                for i in range(samples_per_config):
                    fact = domain_facts[i % len(domain_facts)]
                    needle = fact["fact_text"]
                    needle_tok = n_tokens(needle)
                    body_tok = length - 500 - needle_tok
                    if body_tok < 100:
                        # too small to host needle after prefix, skip
                        continue
                    desired_before = 500 + int(body_tok * pos)
                    pre_budget = max(0, desired_before - 504)
                    pair_rng = random.Random(
                        f"A:{domain}:{length}:{i}:{int(pos * 100)}")
                    for _ in range(12):
                        pre = gen_filler_tokens(domain, pre_budget, rng=pair_rng)
                        before = n_tokens(FIXED_PREFIX_500 + "\n\n" + pre + "\n\n")
                        if before == desired_before:
                            break
                        pre_budget = max(0, pre_budget + desired_before - before)
                    else:
                        raise RuntimeError("cannot place subset-A needle exactly")
                    head = FIXED_PREFIX_500 + "\n\n" + pre + "\n\n" + needle
                    context = finish_with_tail(head, length, domain, rng=pair_rng)

                    dataset.append({
                        "sample_id":            f"A_{domain[:3]}_{length}_p{int(pos*100):02d}_{sid:04d}",
                        "subset":               "A",
                        "domain":               domain,
                        "total_length":         length,
                        "total_length_unit":    "tokens",
                        "position_ratio":       pos,
                        "context":              context,
                        "question":             f"{fact['entity_name']}的{fact['attribute']}是什么？",
                        "answer":               fact["value"],
                    })
                    sid += 1
        print(f"  domain {domain}: {sid} cumulative", flush=True)
    print(f"  Subset A total: {len(dataset)}", flush=True)
    return dataset


# ==================== Subset B : 干扰稀释 ====================

SUBSET_B_LENGTHS_TOK = [8000, 16000, 32000]
SUBSET_B_DENSITIES   = [0.0, 0.25, 0.5, 0.75, 0.9]
SUBSET_B_NOISE_TYPES = ["in_domain_related", "out_domain_unrelated", "random_noise"]
SUBSET_B_CHUNKS      = 8


def build_subset_b(facts, samples_per_config: int = 40):
    """3 lengths × 5 noise_density × 3 noise_types × 40 facts = 1800.
    Needle at the very start; remainder = noise_density × distractor_type
    interleaved with (1 - noise_density) × generic filler.
    """
    print("\nBuilding Subset B …", flush=True)
    dataset = []
    sid = 0
    for length in SUBSET_B_LENGTHS_TOK:
        for d in SUBSET_B_DENSITIES:
            for noise_type in SUBSET_B_NOISE_TYPES:
                for i in range(samples_per_config):
                    fact = facts[i % len(facts)]
                    needle = fact["fact_text"]
                    needle_tok = n_tokens(needle)
                    rest_tok = length - needle_tok
                    if rest_tok < 100:
                        continue
                    d_tok = int(d * rest_tok)
                    # Reserve separator overhead from generic filler; distractor
                    # tokens remain fixed so the measured density is controlled.
                    g_total = max(1, rest_tok - d_tok - 32)
                    d_budgets = ([d_tok // SUBSET_B_CHUNKS
                                  + (1 if k < d_tok % SUBSET_B_CHUNKS else 0)
                                  for k in range(SUBSET_B_CHUNKS)] if d_tok else [])
                    d_chunks = [gen_distractor_tokens(noise_type, fact["domain"], b)
                                for b in d_budgets]

                    fixed_g_budget = max(1, g_total // SUBSET_B_CHUNKS)
                    fixed_g_chunks = [
                        gen_filler_tokens(
                            "general", fixed_g_budget,
                            rng=random.Random(f"B:G:{sid}:{k}"))
                        for k in range(SUBSET_B_CHUNKS - 1)
                    ]
                    last_g_budget = max(
                        1, g_total - fixed_g_budget * (SUBSET_B_CHUNKS - 1))
                    for _ in range(20):
                        last_g = gen_filler_tokens(
                            "general", last_g_budget,
                            rng=random.Random(f"B:G:{sid}:last"))
                        g_chunks = fixed_g_chunks + [last_g]
                        chunks = []
                        for k in range(SUBSET_B_CHUNKS):
                            if d_chunks:
                                chunks.append(d_chunks[k])
                            chunks.append(g_chunks[k])
                        context = needle + "\n\n" + "\n\n".join(chunks)
                        actual = n_tokens(context)
                        if actual == length:
                            break
                        last_g_budget = max(
                            1, last_g_budget + length - actual)
                    else:
                        raise RuntimeError("cannot satisfy subset-B token budget")

                    dataset.append({
                        "sample_id":         f"B_{noise_type[:3]}_d{int(d*100):02d}_{length}_{sid:04d}",
                        "subset":            "B",
                        "dilution_type":     noise_type,
                        "noise_density":     d,
                        "total_length":      length,
                        "total_length_unit": "tokens",
                        "domain":            fact["domain"],
                        "context":           context,
                        "question":          f"{fact['entity_name']}的{fact['attribute']}是什么？",
                        "answer":            fact["value"],
                    })
                    sid += 1
        print(f"  length {length}: {sid} cumulative", flush=True)
    print(f"  Subset B total: {len(dataset)}", flush=True)
    return dataset


# ==================== Subset C : 信息覆盖 ====================

SUBSET_C_LENGTHS_TOK = [8000, 16000, 32000]
SUBSET_C_DISTANCES   = ["near", "medium", "far"]
SUBSET_C_SIM_ORDER   = ["same_name_same_domain", "same_name_diff_domain",
                        "diff_name_same_domain", "completely_different"]
SUBSET_C_GROUPS_PER_SIM = 5
SUBSET_C_SAMPLES_PER_CELL = 50          # = design "50 samples per (sim,dist)"


def _resolve_entity(group, entity, fallback_domain="general"):
    name   = entity.get("name") or group.get("name") or "未知"
    domain = entity.get("domain") or group.get("domain") or fallback_domain
    return (name, domain,
            entity.get("attribute", "属性"),
            entity.get("value", "未知"),
            entity.get("distinguishing", ""))


def build_subset_c(entity_pairs):
    """4 sim × 3 distance × 50 samples = 600. Length sampled uniformly per cell.
    Distance interpretation (per design: 'near = 500 tokens 内'):
        near   -> gap = 400 tokens
        medium -> gap = 2000 tokens
        far    -> gap fills the context
    """
    print("\nBuilding Subset C …", flush=True)
    dataset = []
    sid = 0
    rng = random.Random(202605)

    for sim_level in SUBSET_C_SIM_ORDER:
        groups = entity_pairs.get(sim_level, [])[:SUBSET_C_GROUPS_PER_SIM]
        if not groups:
            continue
        for dist_label in SUBSET_C_DISTANCES:
            for k in range(SUBSET_C_SAMPLES_PER_CELL):
                grp = groups[k % len(groups)]
                entities = grp.get("entities", [])
                if len(entities) < 2:
                    continue
                e1, e2 = entities[0], entities[1]
                n1, dom1, attr1, val1, dist1 = _resolve_entity(grp, e1)
                n2, dom2, attr2, val2, dist2 = _resolve_entity(grp, e2, fallback_domain=dom1)
                desc1 = f"{n1}（{dist1}），其{attr1}是{val1}。"
                desc2 = f"{n2}（{dist2}），其{attr2}是{val2}。"
                desc1_t = n_tokens(desc1)
                desc2_t = n_tokens(desc2)

                length = rng.choice(SUBSET_C_LENGTHS_TOK)
                if dist_label == "near":
                    gap_t = 400
                elif dist_label == "medium":
                    gap_t = 2000
                else:                                              # far
                    gap_t = max(400, length - desc1_t - desc2_t - 400)

                used_t = desc1_t + desc2_t + gap_t
                remain = length - used_t
                pre_t  = remain // 2
                post_t = remain - pre_t

                filler_rng = random.Random(
                    f"C:{sim_level}:{dist_label}:{k}:{length}")
                pre_filler = gen_filler_tokens(
                    dom1, max(0, pre_t - 6), rng=filler_rng)
                if rng.random() < 0.5:
                    first, second = desc1, desc2
                else:
                    first, second = desc2, desc1
                gap_filler = exact_gap_filler(
                    first, second, dom1, gap_t, rng=filler_rng)
                head = "\n\n".join([pre_filler, first, gap_filler, second])
                context = finish_with_tail(
                    head, length, dom1, rng=filler_rng)

                target_is_first = rng.random() < 0.5
                if target_is_first:
                    t_name, t_dist, t_attr, t_val = n1, dist1, attr1, val1
                else:
                    t_name, t_dist, t_attr, t_val = n2, dist2, attr2, val2

                dataset.append({
                    "sample_id":         f"C_{sim_level[:3]}_{dist_label}_{length}_{sid:04d}",
                    "subset":            "C",
                    "similarity_level":  sim_level,
                    "distance_level":    dist_label,
                    "total_length":      length,
                    "total_length_unit": "tokens",
                    "domain":            dom1,
                    "context":           context,
                    "question":          f"{t_dist}的{t_name}，其{t_attr}是什么？",
                    "answer":            t_val,
                })
                sid += 1
        print(f"  sim {sim_level}: {sid} cumulative", flush=True)
    print(f"  Subset C total: {len(dataset)}", flush=True)
    return dataset


# ==================== Subset D : 多跳衰减 ====================

SUBSET_D_LENGTHS_TOK = [8000, 16000, 32000, 64000]
SUBSET_D_HOPS        = [2, 3, 4]                       # 4-hop requires KB extension
SUBSET_D_GAPS_TOK    = {"near": 1000, "medium": 4000, "far": 8000}
SUBSET_D_SAMPLES_PER_CELL = 40


def build_subset_d(fact_chains):
    """3 hops × 3 distances × 4 lengths × 40 = 1440.
    Per design: inter-hop gap ∈ {1K, 4K, 8K} tokens.
    Combinations where total < num_hops*needle + (hops-1)*gap + 400 are skipped.
    4-hop is only generated if fact_chains.json contains chains with ≥4 hops.
    """
    print("\nBuilding Subset D …", flush=True)
    dataset = []
    sid = 0
    rng = random.Random(202606)

    # Flatten all chains; pre-compute needle token counts per chain
    all_chains = []
    for chain_type, chains in fact_chains.items():
        for c in chains:
            evs = [h["evidence"] for h in c["hops"]]
            ev_tok = [n_tokens(e) for e in evs]
            all_chains.append({"type": chain_type, "hops": c["hops"],
                               "ev_tok": ev_tok,
                               "question_template": c.get("question_template"),
                               "answer": c.get("answer")})

    # For 4-hop, filter chains with ≥4 hops
    deficit = {2: 0, 3: 0, 4: 0}
    for num_hops in SUBSET_D_HOPS:
        usable = [c for c in all_chains if len(c["hops"]) >= num_hops]
        if not usable:
            print(f"  [WARN] no chains with ≥{num_hops} hops in KB — skipping all {num_hops}-hop cells",
                  flush=True)
            deficit[num_hops] = 1
            continue

        for dist_label, gap_tok in SUBSET_D_GAPS_TOK.items():
            for length in SUBSET_D_LENGTHS_TOK:
                # Feasibility check using max needle-token count among usable chains
                max_ev_tot = max(sum(c["ev_tok"][:num_hops]) for c in usable)
                min_required = max_ev_tot + (num_hops - 1) * gap_tok + 400
                if min_required > length:
                    continue

                for k in range(SUBSET_D_SAMPLES_PER_CELL):
                    chain = usable[k % len(usable)]
                    hops_used = chain["hops"][:num_hops]
                    ev_used = [
                        f"{h['entity']}的{h['relation']}是{h['target']}。{h['evidence']}"
                        for h in hops_used
                    ]
                    ev_tok_used = [n_tokens(e) for e in ev_used]
                    ev_total = sum(ev_tok_used)

                    gaps_total = (num_hops - 1) * gap_tok
                    remain = length - ev_total - gaps_total
                    if remain < 200:
                        continue
                    pre_t  = remain // 2
                    post_t = remain - pre_t

                    body_parts = [ev_used[0]]
                    for idx in range(1, num_hops):
                        gap_filler = exact_gap_filler(
                            ev_used[idx - 1], ev_used[idx],
                            "computer_science", gap_tok, rng=rng)
                        body_parts.extend([gap_filler, ev_used[idx]])
                    pre = gen_filler_tokens(
                        "computer_science", max(0, pre_t - 6), rng=rng)
                    head = "\n\n".join([pre, *body_parts])
                    context = finish_with_tail(
                        head, length, "computer_science", rng=rng)

                    # Question
                    if chain["question_template"] and num_hops == len(chain["hops"]):
                        question = chain["question_template"]
                        answer   = chain["answer"] or hops_used[-1]["target"]
                    else:
                        rels = " 的 ".join(h["relation"] for h in hops_used)
                        question = f"{hops_used[0]['entity']} 的 {rels} 是什么？"
                        answer   = hops_used[-1]["target"]

                    dataset.append({
                        "sample_id":         f"D_{chain['type'][:3]}_{num_hops}h_{dist_label}_{length}_{sid:04d}",
                        "subset":            "D",
                        "chain_type":        chain["type"],
                        "num_hops":          num_hops,
                        "distance_level":    dist_label,
                        "total_length":      length,
                        "total_length_unit": "tokens",
                        "fact_chain":        hops_used,
                        "context":           context,
                        "question":          question,
                        "answer":            answer,
                    })
                    sid += 1
        print(f"  hops={num_hops}: {sid} cumulative", flush=True)
    print(f"  Subset D total: {len(dataset)}", flush=True)
    if deficit[4]:
        print("  [NOTE] KB has no 4-hop chains; extend core_knowledge/fact_chains.json "
              "to enable 4-hop cells, then re-run.", flush=True)
    return dataset


# ==================== IO + verification ====================

def save_jsonl(dataset, filepath: Path):
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for s in dataset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp.replace(filepath)


def load_jsonl(filepath: Path):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def verify(dataset):
    print("\nQuality Verification:", flush=True)
    issues_len  = 0
    issues_miss = 0
    legacy_phrase_hits = 0
    immediate_duplicate_paragraphs = 0
    worst_paragraph_share = 0.0
    for s in dataset:
        actual_tok = n_tokens(s["context"])
        declared = s["total_length"]
        if actual_tok != declared:
            issues_len += 1
        if s["answer"] not in s["context"]:
            issues_miss += 1
        legacy_phrase_hits += s["context"].count(
            "春天来了，大地回春，万物复苏。公园里的樱花竞相开放")
        paragraphs = [p.strip() for p in s["context"].split("\n\n") if p.strip()]
        immediate_duplicate_paragraphs += sum(
            a == b for a, b in zip(paragraphs, paragraphs[1:]))
        if paragraphs:
            worst_paragraph_share = max(
                worst_paragraph_share,
                max(Counter(paragraphs).values()) / len(paragraphs))
    print(f"  exact-length issues      : {issues_len}/{len(dataset)}")
    print(f"  answer-missing issues   : {issues_miss}/{len(dataset)}")
    print(f"  legacy repeated phrase  : {legacy_phrase_hits}")
    print(f"  adjacent duplicate paras: {immediate_duplicate_paragraphs}")
    print(f"  worst single-para share : {worst_paragraph_share:.2%}")
    if issues_len:
        raise RuntimeError(
            f"token-strict verification failed: {issues_len} contexts are not exact")
    if legacy_phrase_hits:
        raise RuntimeError(
            f"legacy filler verification failed: {legacy_phrase_hits} occurrences remain")

    bs = Counter(s["subset"] for s in dataset)
    print(f"\nBy subset: {dict(sorted(bs.items()))}")
    for sub in sorted(bs):
        subs = [s for s in dataset if s["subset"] == sub]
        lens = Counter(s["total_length"] for s in subs)
        actual = [n_tokens(s["context"]) for s in subs[:50]]  # sample first 50 for speed
        print(f"  [{sub}] declared {dict(sorted(lens.items()))}")
        print(f"        actual (first 50 sampled) min/median/max tok: "
              f"{min(actual)} / {sorted(actual)[len(actual)//2]} / {max(actual)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsets", default="A,B,C,D",
        help="comma-separated subsets to rebuild; unselected subsets are reused")
    args = parser.parse_args()
    selected = {x.strip().upper() for x in args.subsets.split(",") if x.strip()}
    if not selected or not selected <= set("ABCD"):
        raise SystemExit("--subsets must contain only A,B,C,D")

    print("=" * 60)
    print("PAC-Test Dataset Builder v3.2 (constraint-strict, diversified filler, Qwen2.5)")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    facts        = json.load(open(KB_DIR / "facts_library.json", encoding="utf-8"))
    entity_pairs = json.load(open(KB_DIR / "entity_pairs.json", encoding="utf-8"))
    fact_chains  = json.load(open(KB_DIR / "fact_chains.json",  encoding="utf-8"))
    print(f"\nLoaded KB: {len(facts)} facts, "
          f"{sum(len(v) for v in entity_pairs.values())} entity-pair groups, "
          f"{sum(len(v) for v in fact_chains.values())} fact chains", flush=True)

    builders = {
        "A": lambda: build_subset_a(facts),
        "B": lambda: build_subset_b(facts),
        "C": lambda: build_subset_c(entity_pairs),
        "D": lambda: build_subset_d(fact_chains),
    }
    all_data = []
    for sub in "ABCD":
        path = DATA_DIR / f"subset_{sub}.jsonl"
        if sub in selected:
            ds = builders[sub]()
            save_jsonl(ds, path)
        else:
            print(f"\nReusing existing Subset {sub} …", flush=True)
            ds = load_jsonl(path)
        all_data += ds
    save_jsonl(all_data, DATA_DIR / "PAC-Test_complete.jsonl")

    verify(all_data)

    print(f"\n{'=' * 60}")
    print("Dataset construction complete.")
    print(f"Output:        {DATA_DIR}")
    print(f"Total samples: {len(all_data)}")
    print("Tokens checked against Qwen2.5-7B-Instruct tokenizer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
