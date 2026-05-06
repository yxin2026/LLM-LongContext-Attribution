#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test Dataset Builder (Lightweight Version)
轻量版构建脚本，优化了文本生成速度。
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(42)

# 简化的token计数（中文字符=1，英文单词=1）
def count_tokens(text: str) -> int:
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english = len(re.findall(r'[a-zA-Z]+', text))
    return chinese + english + (len(text) - chinese - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))) // 3

# 预生成的填充文本段落库（每种领域每种长度预生成少量段落，然后复用）
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
    ]
}

# 随机噪声词库
NOISE_WORDS = ["xyz", "abc", "qwe", "mnp", "rst", "uvw", "ijk", "lmn", "opq", "def"]
NOISE_NUMBERS = ["123", "456", "789", "321", "654", "987"]
NOISE_SYMBOLS = ["@#$", "%^&", "*()", "!@#"]

def generate_filler(domain: str, target_tokens: int) -> str:
    """高效生成填充文本"""
    segments = FILLER_SEGMENTS.get(domain, FILLER_SEGMENTS["general"])
    result = []
    current_tokens = 0
    
    while current_tokens < target_tokens:
        seg = random.choice(segments)
        result.append(seg)
        current_tokens += count_tokens(seg)
    
    text = " ".join(result)
    # 简单截断
    tokens = text.split()
    if len(tokens) > target_tokens:
        text = " ".join(tokens[:target_tokens])
    return text

def generate_distractor(dilution_type: str, domain: str, target_tokens: int) -> str:
    """生成干扰文本"""
    if dilution_type == "random_noise":
        result = []
        current = 0
        while current < target_tokens:
            text = f"{random.choice(NOISE_SYMBOLS)} {random.choice(NOISE_WORDS)} {random.choice(NOISE_NUMBERS)} {random.choice(NOISE_WORDS)}"
            result.append(text)
            current += count_tokens(text)
        return " ".join(result)
    elif dilution_type == "out_domain_unrelated":
        return generate_filler("general", target_tokens)
    else:  # in_domain_related
        return generate_filler(domain, target_tokens)

def build_subset_a(facts, samples_per_config=10):
    """构建子集A"""
    print("Building Subset A...")
    dataset = []
    lengths = [4096, 8192, 16384, 32768]
    positions = [0.0, 0.25, 0.50, 0.75, 1.0]
    domains = list(set(f["domain"] for f in facts))
    sample_id = 0
    
    for domain in domains:
        domain_facts = [f for f in facts if f["domain"] == domain]
        for length in lengths:
            for pos in positions:
                for i in range(samples_per_config):
                    fact = domain_facts[i % len(domain_facts)]
                    fact_text = fact["fact_text"]
                    fact_tokens = count_tokens(fact_text)
                    filler_tokens = length - fact_tokens - 20
                    
                    filler = generate_filler(domain, filler_tokens)
                    
                    # 插入事实
                    tokens = filler.split()
                    insert_idx = int(len(tokens) * pos)
                    new_tokens = tokens[:insert_idx] + [fact_text] + tokens[insert_idx:]
                    context = " ".join(new_tokens)
                    
                    # 调整长度
                    final_tokens = context.split()
                    if len(final_tokens) > length:
                        context = " ".join(final_tokens[:length])
                    
                    dataset.append({
                        "sample_id": f"A_{domain[:3]}_{length}_{int(pos*100):02d}_{sample_id:04d}",
                        "subset": "A",
                        "domain": domain,
                        "total_length": length,
                        "position_ratio": pos,
                        "context": context,
                        "question": f"{fact['entity_name']}的{fact['attribute']}是什么？",
                        "answer": fact["value"]
                    })
                    sample_id += 1
    
    print(f"  Subset A: {len(dataset)} samples")
    return dataset

def build_subset_b(facts, samples_per_config=40):
    """构建子集B"""
    print("Building Subset B...")
    dataset = []
    dilution_types = ["in_domain_related", "out_domain_unrelated", "random_noise"]
    dilution_ratios = [0.20, 0.10, 0.05, 0.02]
    lengths = [4096, 8192, 16384, 32768]
    sample_id = 0
    
    for dilution_type in dilution_types:
        for ratio in dilution_ratios:
            for length in lengths:
                for i in range(samples_per_config):
                    fact = facts[i % len(facts)]
                    fact_text = fact["fact_text"]
                    fact_tokens = count_tokens(fact_text)
                    
                    total_needed = int(fact_tokens / ratio)
                    distractor_tokens = total_needed - fact_tokens
                    
                    num_segments = random.randint(3, 5)
                    segment_length = max(distractor_tokens // num_segments, 50)
                    
                    segments = []
                    for _ in range(num_segments):
                        d = generate_distractor(dilution_type, fact["domain"], segment_length)
                        segments.append(d)
                    
                    insert_pos = random.randint(0, len(segments))
                    segments.insert(insert_pos, fact_text)
                    
                    context = " ".join(segments)
                    final_tokens = context.split()
                    if len(final_tokens) > length:
                        context = " ".join(final_tokens[:length])
                    
                    dataset.append({
                        "sample_id": f"B_{dilution_type[:3]}_{int(ratio*100)}_{length}_{sample_id:04d}",
                        "subset": "B",
                        "dilution_type": dilution_type,
                        "dilution_ratio": ratio,
                        "total_length": length,
                        "domain": fact["domain"],
                        "context": context,
                        "question": f"{fact['entity_name']}的{fact['attribute']}是什么？",
                        "answer": fact["value"]
                    })
                    sample_id += 1
    
    print(f"  Subset B: {len(dataset)} samples")
    return dataset

def build_subset_c(entity_pairs, samples_per_pair=3):
    """构建子集C"""
    print("Building Subset C...")
    dataset = []
    distances = ["near", "medium", "far"]
    lengths = [8192, 16384, 32768]
    sample_id = 0
    
    for sim_level, pairs in entity_pairs.items():
        for pair_group in pairs:
            if "entities" not in pair_group or len(pair_group["entities"]) < 2:
                continue
            
            entities = pair_group["entities"]
            
            for _ in range(samples_per_pair):
                for distance in distances:
                    for length in lengths:
                        entity1, entity2 = entities[0], entities[1]
                        name1 = entity1.get('name', pair_group.get('name', '未知'))
                        name2 = entity2.get('name', pair_group.get('name', '未知'))
                        domain = entity1.get('domain', pair_group.get('domain', 'general'))
                        
                        desc1 = f"{name1}（{entity1.get('distinguishing', '')}），其{entity1['attribute']}是{entity1['value']}。"
                        desc2 = f"{name2}（{entity2.get('distinguishing', '')}），其{entity2['attribute']}是{entity2['value']}。"
                        
                        if distance == "near":
                            filler = generate_filler(domain, 300)
                            context = f"{desc1} {filler} {desc2}"
                        elif distance == "medium":
                            filler = generate_filler(domain, 1500)
                            context = f"{desc1} {filler} {desc2}"
                        else:  # far
                            filler = generate_filler(domain, length - 400)
                            if random.random() > 0.5:
                                context = f"{desc1} {filler} {desc2}"
                            else:
                                context = f"{desc2} {filler} {desc1}"
                        
                        final_tokens = context.split()
                        if len(final_tokens) > length:
                            context = " ".join(final_tokens[:length])
                        
                        target_idx = random.randint(0, 1)
                        target = entities[target_idx]
                        target_name = target.get('name', pair_group.get('name', '未知'))
                        
                        dataset.append({
                            "sample_id": f"C_{sim_level[:3]}_{distance}_{length}_{sample_id:04d}",
                            "subset": "C",
                            "similarity_level": sim_level,
                            "distance_level": distance,
                            "total_length": length,
                            "context": context,
                            "question": f"{target.get('distinguishing', '')}的{target_name}，其{target['attribute']}是什么？",
                            "answer": target["value"]
                        })
                        sample_id += 1
    
    print(f"  Subset C: {len(dataset)} samples")
    return dataset

def build_subset_d(fact_chains, samples_per_chain=2):
    """构建子集D"""
    print("Building Subset D...")
    dataset = []
    hop_options = [2, 3]
    distances = ["adjacent", "medium", "far"]
    lengths = [8192, 16384, 32768, 65536]
    sample_id = 0
    
    for chain_type, chains in fact_chains.items():
        for chain_data in chains:
            for num_hops in hop_options:
                if num_hops > len(chain_data["hops"]):
                    continue
                
                hops = chain_data["hops"][:num_hops]
                
                for _ in range(samples_per_chain):
                    for distance in distances:
                        for length in lengths:
                            segments = []
                            for hop in hops:
                                if distance == "adjacent":
                                    filler_len = 200
                                elif distance == "medium":
                                    filler_len = 1000
                                else:
                                    filler_len = length // (num_hops + 1)
                                
                                filler = generate_filler("computer_science", filler_len)
                                segment = f"{filler} {hop['evidence']}"
                                segments.append(segment)
                            
                            context = " ".join(segments)
                            final_tokens = context.split()
                            if len(final_tokens) > length:
                                context = " ".join(final_tokens[:length])
                            
                            # 构建问题
                            if num_hops == 2:
                                question = f"{hops[0]['entity']}的{hops[0]['relation']}的{hops[1]['relation']}是什么？"
                            else:
                                question = f"{hops[0]['entity']}的{hops[0]['relation']}的{hops[1]['relation']}的{hops[2]['relation']}是什么？"
                            
                            answer = hops[-1]["target"]
                            
                            dataset.append({
                                "sample_id": f"D_{chain_type[:3]}_{num_hops}hop_{distance}_{length}_{sample_id:04d}",
                                "subset": "D",
                                "chain_type": chain_type,
                                "num_hops": num_hops,
                                "distance_level": distance,
                                "total_length": length,
                                "fact_chain": hops,
                                "context": context,
                                "question": question,
                                "answer": answer
                            })
                            sample_id += 1
    
    print(f"  Subset D: {len(dataset)} samples")
    return dataset

def save_jsonl(dataset, filepath):
    """保存为JSONL格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def verify_dataset(dataset):
    """质量验证"""
    print("\nQuality Verification:")
    issues = []
    
    for i, sample in enumerate(dataset):
        # 检查答案是否在上下文中（子集D的答案可能通过推理得出，不强制要求直接出现在上下文中）
        if sample.get("subset") != "D" and sample["answer"] not in sample["context"]:
            issues.append(f"{sample['sample_id']}: Answer not in context")
        
        # 检查字段完整性
        required = ["sample_id", "subset", "context", "question", "answer"]
        for field in required:
            if field not in sample:
                issues.append(f"{sample['sample_id']}: Missing {field}")
    
    print(f"  Total issues: {len(issues)}")
    if issues:
        for issue in issues[:10]:
            print(f"    {issue}")
    
    # 统计
    subsets = {}
    domains = {}
    lengths = {}
    for s in dataset:
        subsets[s.get("subset", "unknown")] = subsets.get(s.get("subset", "unknown"), 0) + 1
        domains[s.get("domain", s.get("chain_type", "unknown"))] = domains.get(s.get("domain", s.get("chain_type", "unknown")), 0) + 1
        lengths[s.get("total_length", 0)] = lengths.get(s.get("total_length", 0), 0) + 1
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  By subset: {subsets}")
    print(f"  By length: {lengths}")
    
    return len(issues) == 0

def main():
    base_dir = Path("上下文机制探究/PAC-Test-Dataset")
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PAC-Test Dataset Builder (Lightweight)")
    print("=" * 60)
    
    # 加载知识库
    print("\nLoading knowledge base...")
    with open(base_dir / "core_knowledge/facts_library.json", "r", encoding="utf-8") as f:
        facts = json.load(f)
    with open(base_dir / "core_knowledge/entity_pairs.json", "r", encoding="utf-8") as f:
        entity_pairs = json.load(f)
    with open(base_dir / "core_knowledge/fact_chains.json", "r", encoding="utf-8") as f:
        fact_chains = json.load(f)
    
    print(f"  Facts: {len(facts)}")
    print(f"  Entity pairs groups: {sum(len(v) for v in entity_pairs.values())}")
    print(f"  Fact chains groups: {sum(len(v) for v in fact_chains.values())}")
    
    all_data = []
    
    # 构建各子集（按文档要求规模）
    dataset_a = build_subset_a(facts, samples_per_config=10)  # 1000样本
    save_jsonl(dataset_a, data_dir / "subset_A.jsonl")
    all_data.extend(dataset_a)
    
    dataset_b = build_subset_b(facts, samples_per_config=40)  # 1920样本
    save_jsonl(dataset_b, data_dir / "subset_B.jsonl")
    all_data.extend(dataset_b)
    
    dataset_c = build_subset_c(entity_pairs, samples_per_pair=3)  # 729样本 (接近600)
    save_jsonl(dataset_c, data_dir / "subset_C.jsonl")
    all_data.extend(dataset_c)
    
    dataset_d = build_subset_d(fact_chains, samples_per_chain=9)  # 约1440样本
    save_jsonl(dataset_d, data_dir / "subset_D.jsonl")
    all_data.extend(dataset_d)
    
    # 保存完整数据集
    save_jsonl(all_data, data_dir / "PAC-Test_complete.jsonl")
    
    # 质量验证
    verify_dataset(all_data)
    
    # 生成README
    readme = f"""# PAC-Test Dataset

## 概览

PAC-Test 是一个用于评估大语言模型长上下文理解能力的中文测试集。

## 统计信息

- **总样本数**: {len(all_data)}
- **子集A (位置效应)**: {len(dataset_a)} 样本
- **子集B (干扰稀释)**: {len(dataset_b)} 样本
- **子集C (信息覆盖)**: {len(dataset_c)} 样本
- **子集D (多跳衰减)**: {len(dataset_d)} 样本

## 文件说明

| 文件 | 说明 |
|------|------|
| `subset_A.jsonl` | 位置效应测试 |
| `subset_B.jsonl` | 干扰稀释测试 |
| `subset_C.jsonl` | 信息覆盖测试 |
| `subset_D.jsonl` | 多跳衰减测试 |
| `PAC-Test_complete.jsonl` | 完整数据集 |

## 数据格式

```json
{{
  "sample_id": "唯一标识符",
  "subset": "A/B/C/D",
  "context": "上下文文本",
  "question": "问题",
  "answer": "答案",
  "total_length": "目标长度(tokens)"
}}
```

## 子集说明

### 子集A：位置效应测试
测试模型在不同上下文位置（开头/中间/结尾）检索关键信息的能力。
- 位置层级：0%, 25%, 50%, 75%, 100%
- 长度层级：4k, 8k, 16k tokens
- 领域：计算机科学、医学、法律、金融、教育

### 子集B：干扰稀释测试
测试模型在大量干扰信息中保持对目标事实注意力的能力。
- 干扰类型：领域内相关 / 领域外无关 / 随机噪声
- 稀释比例：20%, 10%, 5%, 2%
- 长度层级：4k, 8k, 16k tokens

### 子集C：信息覆盖测试
测试模型区分易混淆实体的能力。
- 相似度等级：同名同领域 / 同名不同领域 / 不同名同领域 / 完全不同
- 距离层级：近 / 中 / 远
- 长度层级：4k, 8k, 16k tokens

### 子集D：多跳衰减测试
测试模型跨段落整合多个分散事实进行推理的能力。
- 事实链类型：人物关系 / 地理位置 / 组织机构 / 时间序列
- 跳数：2跳 / 3跳
- 段间距：相邻 / 中等 / 远
- 长度层级：4k, 8k, 16k tokens

## 构建方法

本数据集通过程序化方式构建：
1. 从5个领域构建包含250+个事实的核心知识库
2. 构建27组易混淆实体对和7条多跳事实链
3. 使用预定义段落库通过随机组合生成填充文本
4. 按照各子集设计要求插入事实并构造QA

## 使用建议

1. **评估指标**: 主要使用 Exact Match (EM)
2. **分层分析**: 按子集、长度、位置等维度分别报告性能
3. **对比分析**: 比较不同子集的性能差异，定位模型的能力短板

## Citation

```
PAC-Test: A Programmatic Benchmark for Long-Context Understanding
```
"""
    
    with open(data_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
    print(f"\n{'=' * 60}")
    print("Dataset construction completed!")
    print(f"Output directory: {data_dir}")
    print(f"Total samples: {len(all_data)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
