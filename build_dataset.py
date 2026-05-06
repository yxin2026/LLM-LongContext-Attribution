#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test Dataset Builder
构建完整的PAC-Test数据集，包括4个子集。
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys

# 尝试导入tiktoken，如果没有则使用简单的近似方法
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(ENCODER.encode(text))
except ImportError:
    print("Warning: tiktoken not available, using approximate token count")
    def count_tokens(text: str) -> int:
        # 简单的近似：中文字符算1个token，英文单词算1个token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words + other // 3

random.seed(42)

class TextGenerator:
    """基于模板生成填充文本"""
    
    def __init__(self, templates_path: str, variables_path: str = None):
        with open(templates_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'templates' in data:
                self.templates = data['templates']
                self.variables = data.get('variables', {})
            else:
                self.templates = data
                self.variables = {}
        
        self.distractor_templates = {}
        distractor_path = Path(templates_path).parent / 'distractor_templates.json'
        if distractor_path.exists():
            with open(distractor_path, 'r', encoding='utf-8') as f:
                self.distractor_templates = json.load(f)
    
    def generate_filler_text(self, domain: str, target_tokens: int, topic: str = None) -> str:
        """生成指定长度的填充文本"""
        if domain not in self.templates:
            domain = random.choice(list(self.templates.keys()))
        
        templates_list = self.templates[domain]
        vars_pool = self.variables.get(domain, {})
        
        generated_text = ""
        generated_tokens = 0
        
        while generated_tokens < target_tokens:
            # 随机选择一个模板
            template = random.choice(templates_list)
            
            # 填充变量
            filled = template
            for var_name, var_values in vars_pool.items():
                if f"{{{var_name}}}" in filled:
                    filled = filled.replace(f"{{{var_name}}}", random.choice(var_values))
            
            # 如果没有被替换的变量，使用topic
            if topic and "{topic}" in filled:
                filled = filled.replace("{topic}", topic)
            
            # 如果没有topic，随机选择一个
            if "{topic}" in filled and "topic" in vars_pool:
                filled = filled.replace("{topic}", random.choice(vars_pool["topic"]))
            
            # 移除未被替换的占位符
            filled = re.sub(r'\{\w+\}', '相关技术', filled)
            
            generated_text += filled + " "
            generated_tokens = count_tokens(generated_text)
            
            # 防止无限循环
            if len(generated_text) > target_tokens * 10:
                break
        
        # 截断到目标长度
        tokens = generated_text.split()
        approx_text = " ".join(tokens[:target_tokens])
        return approx_text[:int(target_tokens * 3)]  # 简单截断
    
    def generate_distractor(self, dilution_type: str, domain: str, topic: str = None, target_tokens: int = 500) -> str:
        """生成干扰文本"""
        if dilution_type == "random_noise":
            return self._generate_random_noise(target_tokens)
        elif dilution_type == "out_domain_unrelated":
            return self._generate_out_domain_text(target_tokens)
        else:  # in_domain_related
            return self._generate_in_domain_distractor(domain, topic, target_tokens)
    
    def _generate_random_noise(self, target_tokens: int) -> str:
        """生成随机噪声文本"""
        noise_data = self.distractor_templates.get("random_noise", {})
        patterns = noise_data.get("patterns", ["{word1} {number} {word2}"])
        words = noise_data.get("words", ["xyz", "abc"])
        numbers = noise_data.get("numbers", ["123"])
        symbols = noise_data.get("symbols", ["@#$"])
        
        result = []
        current_tokens = 0
        
        while current_tokens < target_tokens:
            pattern = random.choice(patterns)
            text = pattern
            text = text.replace("{word1}", random.choice(words))
            text = text.replace("{word2}", random.choice(words))
            text = text.replace("{word3}", random.choice(words))
            text = text.replace("{word4}", random.choice(words))
            text = text.replace("{number}", random.choice(numbers))
            text = text.replace("{symbol}", random.choice(symbols))
            
            result.append(text)
            current_tokens += count_tokens(text)
        
        return " ".join(result)
    
    def _generate_out_domain_text(self, target_tokens: int) -> str:
        """生成领域外无关文本"""
        templates = self.distractor_templates.get("out_domain_unrelated", {}).get("general", [])
        if not templates:
            return self._generate_random_noise(target_tokens)
        
        result = []
        current_tokens = 0
        
        while current_tokens < target_tokens:
            text = random.choice(templates)
            result.append(text)
            current_tokens += count_tokens(text)
        
        return " ".join(result)
    
    def _generate_in_domain_distractor(self, domain: str, topic: str, target_tokens: int) -> str:
        """生成领域内相关干扰文本"""
        templates_dict = self.distractor_templates.get("in_domain_related", {})
        
        if domain not in templates_dict:
            domain = random.choice(list(templates_dict.keys()))
        
        templates = templates_dict[domain]
        vars_pool = self.variables.get(domain, {})
        
        result = []
        current_tokens = 0
        
        while current_tokens < target_tokens:
            template = random.choice(templates)
            filled = template
            
            for var_name, var_values in vars_pool.items():
                if f"{{{var_name}}}" in filled:
                    filled = filled.replace(f"{{{var_name}}}", random.choice(var_values))
            
            if topic and "{topic}" in filled:
                filled = filled.replace("{topic}", topic)
            
            filled = re.sub(r'\{\w+\}', '相关内容', filled)
            
            result.append(filled)
            current_tokens += count_tokens(filled)
        
        return " ".join(result)


class SubsetABuilder:
    """子集A：位置效应测试"""
    
    def __init__(self, facts_library: List[Dict], text_generator: TextGenerator):
        self.facts = facts_library
        self.text_gen = text_generator
        self.domains = list(set(f["domain"] for f in facts_library))
        self.lengths = [4096, 8192, 16384]  # 减少最大长度以加速构建
        self.positions = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    def build(self, samples_per_config: int = 50) -> List[Dict]:
        """构建子集A数据集"""
        dataset = []
        sample_id = 0
        
        print(f"Building Subset A: {len(self.domains)} domains × {len(self.lengths)} lengths × {len(self.positions)} positions × {samples_per_config} samples")
        
        for domain in self.domains:
            domain_facts = [f for f in self.facts if f["domain"] == domain]
            
            for length in self.lengths:
                for position in self.positions:
                    for i in range(samples_per_config):
                        fact = domain_facts[i % len(domain_facts)]
                        sample = self._build_sample(fact, domain, length, position, sample_id)
                        dataset.append(sample)
                        sample_id += 1
                        
                        if sample_id % 100 == 0:
                            print(f"  Generated {sample_id} samples...")
        
        print(f"Subset A completed: {len(dataset)} samples")
        return dataset
    
    def _build_sample(self, fact: Dict, domain: str, length: int, position: float, sample_id: int) -> Dict:
        """构建单个样本"""
        # 事实文本
        fact_text = fact["fact_text"]
        fact_tokens = count_tokens(fact_text)
        
        # 需要的填充文本长度
        filler_target = length - fact_tokens - 50  # 50 tokens buffer
        
        # 生成填充文本
        filler = self.text_gen.generate_filler_text(domain, filler_target, fact["entity_name"])
        
        # 在指定位置插入事实
        context = self._insert_at_position(filler, fact_text, position)
        
        # 调整长度
        context = self._adjust_length(context, length)
        
        # 构建QA
        question = f"{fact['entity_name']}的{fact['attribute']}是什么？"
        answer = fact["value"]
        
        return {
            "sample_id": f"A_{domain[:3]}_{length}_{int(position*100):02d}_{sample_id:04d}",
            "subset": "A",
            "domain": domain,
            "total_length": length,
            "position_level": int(position * 4),  # 0,1,2,3,4
            "position_ratio": position,
            "entity": fact["entity_name"],
            "attribute": fact["attribute"],
            "context": context,
            "question": question,
            "answer": answer
        }
    
    def _insert_at_position(self, text: str, insertion: str, ratio: float) -> str:
        """在指定比例位置插入文本"""
        tokens = text.split()
        insert_idx = int(len(tokens) * ratio)
        
        # 确保在句子边界
        if insert_idx < len(tokens):
            # 简单处理：在中间插入
            new_tokens = tokens[:insert_idx] + [insertion] + tokens[insert_idx:]
            return " ".join(new_tokens)
        else:
            return text + " " + insertion
    
    def _adjust_length(self, text: str, target_length: int) -> str:
        """调整文本长度"""
        tokens = text.split()
        if len(tokens) > target_length:
            return " ".join(tokens[:target_length])
        return text


class SubsetBBuilder:
    """子集B：干扰稀释测试"""
    
    def __init__(self, facts_library: List[Dict], text_generator: TextGenerator):
        self.facts = facts_library
        self.text_gen = text_generator
        self.dilution_types = ["in_domain_related", "out_domain_unrelated", "random_noise"]
        self.dilution_ratios = [0.20, 0.10, 0.05, 0.02]
        self.lengths = [4096, 8192, 16384]  # 减少最大长度以加速构建
    
    def build(self, samples_per_config: int = 40) -> List[Dict]:
        """构建子集B数据集"""
        dataset = []
        sample_id = 0
        
        print(f"Building Subset B: {len(self.dilution_types)} types × {len(self.dilution_ratios)} ratios × {len(self.lengths)} lengths × {samples_per_config} samples")
        
        for dilution_type in self.dilution_types:
            for ratio in self.dilution_ratios:
                for length in self.lengths:
                    for i in range(samples_per_config):
                        fact = self.facts[i % len(self.facts)]
                        sample = self._build_sample(fact, dilution_type, ratio, length, sample_id)
                        dataset.append(sample)
                        sample_id += 1
                        
                        if sample_id % 100 == 0:
                            print(f"  Generated {sample_id} samples...")
        
        print(f"Subset B completed: {len(dataset)} samples")
        return dataset
    
    def _build_sample(self, fact: Dict, dilution_type: str, ratio: float, length: int, sample_id: int) -> Dict:
        """构建单个样本"""
        fact_text = fact["fact_text"]
        fact_tokens = count_tokens(fact_text)
        
        # 计算干扰文本需要的token数
        # 总长度 = 目标事实 / 密度
        # 干扰 = 总长度 - 目标事实
        total_needed = int(fact_tokens / ratio)
        distractor_tokens = total_needed - fact_tokens
        
        # 将干扰分成多段
        num_segments = random.randint(3, 5)
        segment_length = distractor_tokens // num_segments
        
        segments = []
        for _ in range(num_segments):
            distractor = self.text_gen.generate_distractor(
                dilution_type, fact["domain"], fact["entity_name"], segment_length
            )
            segments.append(distractor)
        
        # 随机位置插入目标事实
        insert_pos = random.randint(0, len(segments))
        segments.insert(insert_pos, fact_text)
        
        context = " ".join(segments)
        context = self._adjust_length(context, length)
        
        return {
            "sample_id": f"B_{dilution_type[:3]}_{int(ratio*100)}_{length}_{sample_id:04d}",
            "subset": "B",
            "dilution_type": dilution_type,
            "dilution_ratio": ratio,
            "total_length": length,
            "domain": fact["domain"],
            "context": context,
            "question": f"{fact['entity_name']}的{fact['attribute']}是什么？",
            "answer": fact["value"]
        }
    
    def _adjust_length(self, text: str, target_length: int) -> str:
        tokens = text.split()
        if len(tokens) > target_length:
            return " ".join(tokens[:target_length])
        return text


class SubsetCBuilder:
    """子集C：信息覆盖测试"""
    
    def __init__(self, entity_pairs: Dict, text_generator: TextGenerator):
        self.entity_pairs = entity_pairs
        self.text_gen = text_generator
        self.similarity_levels = list(entity_pairs.keys())
        self.distances = ["near", "medium", "far"]
        self.lengths = [4096, 8192, 16384]  # 减少最大长度以加速构建
    
    def build(self, samples_per_pair: int = 5) -> List[Dict]:
        """构建子集C数据集"""
        dataset = []
        sample_id = 0
        
        print(f"Building Subset C: {len(self.similarity_levels)} similarity × {len(self.distances)} distances × samples...")
        
        for sim_level in self.similarity_levels:
            pairs = self.entity_pairs[sim_level]
            
            for pair_group in pairs:
                for _ in range(samples_per_pair):
                    for distance in self.distances:
                        for length in self.lengths:
                            sample = self._build_sample(pair_group, sim_level, distance, length, sample_id)
                            if sample:
                                dataset.append(sample)
                                sample_id += 1
                                
                                if sample_id % 100 == 0:
                                    print(f"  Generated {sample_id} samples...")
        
        print(f"Subset C completed: {len(dataset)} samples")
        return dataset
    
    def _build_sample(self, pair_group: Dict, sim_level: str, distance: str, length: int, sample_id: int) -> Dict:
        """构建单个样本"""
        # 解析实体对
        if "entities" not in pair_group:
            return None
        
        entities = pair_group["entities"]
        
        if len(entities) < 2:
            return None
        
        entity1, entity2 = entities[0], entities[1]
        
        # 处理不同结构：有些 name 在顶层，有些在 entity 内部
        name1 = entity1.get('name', pair_group.get('name', '未知'))
        name2 = entity2.get('name', pair_group.get('name', '未知'))
        
        # 构建实体描述
        desc1 = f"{name1}（{entity1.get('distinguishing', '')}），其{entity1['attribute']}是{entity1['value']}。"
        desc2 = f"{name2}（{entity2.get('distinguishing', '')}），其{entity2['attribute']}是{entity2['value']}。"
        
        # 根据距离构建上下文
        if distance == "near":
            filler = self.text_gen.generate_filler_text(entity1.get('domain', 'general'), 500)
            context = f"{desc1} {filler} {desc2}"
        elif distance == "medium":
            filler = self.text_gen.generate_filler_text(entity1.get('domain', 'general'), 2000)
            context = f"{desc1} {filler} {desc2}"
        else:  # far
            filler1 = self.text_gen.generate_filler_text(entity1.get('domain', 'general'), length // 2 - 500)
            filler2 = self.text_gen.generate_filler_text(entity1.get('domain', 'general'), length // 2 - 500)
            # 随机顺序
            if random.random() > 0.5:
                context = f"{desc1} {filler1} {filler2} {desc2}"
            else:
                context = f"{desc2} {filler1} {filler2} {desc1}"
        
        context = self._adjust_length(context, length)
        
        # 随机选择目标实体
        target_idx = random.randint(0, 1)
        target = entities[target_idx]
        target_name = target.get('name', pair_group.get('name', '未知'))
        
        # 构建带区分特征的问题
        question = f"{target.get('distinguishing', '')}的{target_name}，其{target['attribute']}是什么？"
        answer = target["value"]
        
        return {
            "sample_id": f"C_{sim_level[:3]}_{distance}_{length}_{sample_id:04d}",
            "subset": "C",
            "similarity_level": sim_level,
            "distance_level": distance,
            "total_length": length,
            "entity1": entity1,
            "entity2": entity2,
            "target_index": target_idx,
            "context": context,
            "question": question,
            "answer": answer
        }
    
    def _adjust_length(self, text: str, target_length: int) -> str:
        tokens = text.split()
        if len(tokens) > target_length:
            return " ".join(tokens[:target_length])
        return text


class SubsetDBuilder:
    """子集D：多跳衰减测试"""
    
    def __init__(self, fact_chains: Dict, text_generator: TextGenerator):
        self.fact_chains = fact_chains
        self.text_gen = text_generator
        self.chain_types = list(fact_chains.keys())
        self.hop_options = [2, 3]
        self.distances = ["adjacent", "medium", "far"]
        self.lengths = [4096, 8192, 16384]  # 减少最大长度和跳数以加速构建
    
    def build(self, samples_per_chain: int = 20) -> List[Dict]:
        """构建子集D数据集"""
        dataset = []
        sample_id = 0
        
        print(f"Building Subset D: {len(self.chain_types)} types × {len(self.hop_options)} hops × samples...")
        
        for chain_type in self.chain_types:
            chains = self.fact_chains[chain_type]
            
            for chain_data in chains:
                for num_hops in self.hop_options:
                    if num_hops > len(chain_data["hops"]):
                        continue
                    
                    for _ in range(samples_per_chain):
                        for distance in self.distances:
                            for length in self.lengths[:3]:  # 限制长度避免太大
                                sample = self._build_sample(
                                    chain_data, chain_type, num_hops, distance, length, sample_id
                                )
                                dataset.append(sample)
                                sample_id += 1
                                
                                if sample_id % 100 == 0:
                                    print(f"  Generated {sample_id} samples...")
        
        print(f"Subset D completed: {len(dataset)} samples")
        return dataset
    
    def _build_sample(self, chain_data: Dict, chain_type: str, num_hops: int, 
                      distance: str, length: int, sample_id: int) -> Dict:
        """构建单个样本"""
        hops = chain_data["hops"][:num_hops]
        
        # 生成每跳的段落
        segments = []
        for hop in hops:
            evidence = hop["evidence"]
            
            if distance == "adjacent":
                filler_len = 200
            elif distance == "medium":
                filler_len = 1500
            else:  # far
                filler_len = length // (num_hops + 1)
            
            filler = self.text_gen.generate_filler_text("computer_science", filler_len)
            segment = f"{filler} {evidence}"
            segments.append(segment)
        
        context = " ".join(segments)
        context = self._adjust_length(context, length)
        
        # 构建链式问题
        question = self._build_chain_question(hops)
        answer = hops[-1]["target"]
        
        return {
            "sample_id": f"D_{chain_type[:3]}_{num_hops}hop_{distance}_{length}_{sample_id:04d}",
            "subset": "D",
            "chain_type": chain_type,
            "num_hops": num_hops,
            "distance_level": distance,
            "total_length": length,
            "fact_chain": hops,
            "reasoning_path": [f"{h['entity']}->{h['target']}" for h in hops],
            "context": context,
            "question": question,
            "answer": answer
        }
    
    def _build_chain_question(self, hops: List[Dict]) -> str:
        """构建链式问题"""
        if len(hops) == 2:
            return f"{hops[0]['entity']}的{hops[0]['relation']}的{hops[1]['relation']}是什么？"
        elif len(hops) == 3:
            return f"{hops[0]['entity']}的{hops[0]['relation']}的{hops[1]['relation']}的{hops[2]['relation']}是什么？"
        else:
            return f"根据文中信息，{hops[0]['entity']}相关的最终结论是什么？"
    
    def _adjust_length(self, text: str, target_length: int) -> str:
        tokens = text.split()
        if len(tokens) > target_length:
            return " ".join(tokens[:target_length])
        return text


class QualityController:
    """质量控制"""
    
    @staticmethod
    def verify_sample(sample: Dict) -> Tuple[bool, List[str]]:
        """验证单个样本"""
        issues = []
        
        # 检查基本字段
        required_fields = ["sample_id", "subset", "context", "question", "answer"]
        for field in required_fields:
            if field not in sample:
                issues.append(f"Missing field: {field}")
        
        # 检查答案是否在上下文中
        if "context" in sample and "answer" in sample:
            if sample["answer"] not in sample["context"]:
                issues.append("Answer not found in context")
        
        # 检查答案唯一性（简单检查：答案出现次数）
        if "context" in sample and "answer" in sample:
            count = sample["context"].count(sample["answer"])
            if count == 0:
                issues.append("Answer not in context")
            elif count > 5:  # 出现太多次可能有问题
                issues.append(f"Answer appears {count} times")
        
        # 检查长度
        if "total_length" in sample and "context" in sample:
            actual_length = count_tokens(sample["context"])
            expected = sample["total_length"]
            ratio = abs(actual_length - expected) / expected
            if ratio > 0.2:  # 20% 容差
                issues.append(f"Length mismatch: {actual_length} vs {expected}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def analyze_dataset(dataset: List[Dict]) -> Dict:
        """分析数据集统计信息"""
        stats = {
            "total_samples": len(dataset),
            "by_subset": {},
            "by_domain": {},
            "avg_context_length": 0,
            "avg_answer_length": 0
        }
        
        for sample in dataset:
            subset = sample.get("subset", "unknown")
            stats["by_subset"][subset] = stats["by_subset"].get(subset, 0) + 1
            
            domain = sample.get("domain", "unknown")
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            
            if "context" in sample:
                stats["avg_context_length"] += count_tokens(sample["context"])
            if "answer" in sample:
                stats["avg_answer_length"] += len(sample["answer"])
        
        if dataset:
            stats["avg_context_length"] /= len(dataset)
            stats["avg_answer_length"] /= len(dataset)
        
        return stats


def main():
    base_dir = Path("上下文机制探究/PAC-Test-Dataset")
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PAC-Test Dataset Builder")
    print("=" * 60)
    
    # 加载知识库
    print("\nLoading knowledge base...")
    with open(base_dir / "core_knowledge/facts_library.json", "r", encoding="utf-8") as f:
        facts_library = json.load(f)
    
    with open(base_dir / "core_knowledge/entity_pairs.json", "r", encoding="utf-8") as f:
        entity_pairs = json.load(f)
    
    with open(base_dir / "core_knowledge/fact_chains.json", "r", encoding="utf-8") as f:
        fact_chains = json.load(f)
    
    # 初始化文本生成器
    text_gen = TextGenerator(str(base_dir / "templates/filler_templates.json"))
    
    all_datasets = []
    
    # 构建子集A
    print("\n" + "=" * 60)
    builder_a = SubsetABuilder(facts_library, text_gen)
    dataset_a = builder_a.build(samples_per_config=5)  # 减少样本数以控制大小
    all_datasets.extend(dataset_a)
    
    # 保存子集A
    with open(data_dir / "subset_A.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset_a:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved subset A to {data_dir / 'subset_A.jsonl'}")
    
    # 构建子集B
    print("\n" + "=" * 60)
    builder_b = SubsetBBuilder(facts_library, text_gen)
    dataset_b = builder_b.build(samples_per_config=5)  # 减少样本数
    all_datasets.extend(dataset_b)
    
    with open(data_dir / "subset_B.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset_b:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved subset B to {data_dir / 'subset_B.jsonl'}")
    
    # 构建子集C
    print("\n" + "=" * 60)
    builder_c = SubsetCBuilder(entity_pairs, text_gen)
    dataset_c = builder_c.build(samples_per_pair=5)  # 减少样本数
    all_datasets.extend(dataset_c)
    
    with open(data_dir / "subset_C.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset_c:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved subset C to {data_dir / 'subset_C.jsonl'}")
    
    # 构建子集D
    print("\n" + "=" * 60)
    builder_d = SubsetDBuilder(fact_chains, text_gen)
    dataset_d = builder_d.build(samples_per_chain=10)  # 减少样本数
    all_datasets.extend(dataset_d)
    
    with open(data_dir / "subset_D.jsonl", "w", encoding="utf-8") as f:
        for sample in dataset_d:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved subset D to {data_dir / 'subset_D.jsonl'}")
    
    # 质量控制
    print("\n" + "=" * 60)
    print("Quality Control")
    print("=" * 60)
    
    qc = QualityController()
    stats = qc.analyze_dataset(all_datasets)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  By subset: {stats['by_subset']}")
    print(f"  By domain: {stats['by_domain']}")
    print(f"  Avg context length (tokens): {stats['avg_context_length']:.1f}")
    print(f"  Avg answer length (chars): {stats['avg_answer_length']:.1f}")
    
    # 验证样本
    print(f"\nValidating {min(100, len(all_datasets))} random samples...")
    random_samples = random.sample(all_datasets, min(100, len(all_datasets)))
    passed = 0
    for sample in random_samples:
        ok, issues = qc.verify_sample(sample)
        if ok:
            passed += 1
        elif issues:
            print(f"  Sample {sample.get('sample_id', 'unknown')}: {issues}")
    
    print(f"  Validation passed: {passed}/{len(random_samples)} ({passed/len(random_samples)*100:.1f}%)")
    
    # 保存完整数据集
    print("\n" + "=" * 60)
    print("Saving complete dataset...")
    with open(data_dir / "PAC-Test_complete.jsonl", "w", encoding="utf-8") as f:
        for sample in all_datasets:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # 生成README
    readme_content = f"""# PAC-Test Dataset

## 数据集概览

PAC-Test (Position, Attention, Coverage) 是一个用于评估大语言模型长上下文能力的中文测试集。

### 数据集统计

- **总样本数**: {stats['total_samples']}
- **平均上下文长度**: {stats['avg_context_length']:.0f} tokens
- **平均答案长度**: {stats['avg_answer_length']:.1f} 字符

### 子集分布

"""
    for subset, count in stats['by_subset'].items():
        readme_content += f"- **子集{subset}**: {count} 样本\n"
    
    readme_content += """
### 文件说明

- `subset_A.jsonl`: 位置效应测试 (Position Effect)
- `subset_B.jsonl`: 干扰稀释测试 (Dilution Effect)
- `subset_C.jsonl`: 信息覆盖测试 (Overwriting Effect)
- `subset_D.jsonl`: 多跳衰减测试 (Multi-hop Decay)
- `PAC-Test_complete.jsonl`: 完整数据集

### 数据格式

每个样本包含以下字段：

```json
{
  "sample_id": "唯一标识符",
  "subset": "所属子集 (A/B/C/D)",
  "context": "上下文文本",
  "question": "问题",
  "answer": "答案",
  "total_length": "目标长度 (tokens)",
  // 子集特有字段...
}
```

### 构建方法

本数据集通过程序化方式构建：
1. 从5个领域（计算机科学、医学、法律、金融、教育）构建事实库
2. 使用模板和随机组合生成填充文本
3. 按照各子集的设计要求插入事实并构造QA

### 使用建议

1. **评估指标**: 使用 Exact Match (EM) 作为主要评估指标
2. **长度分组**: 可按 `total_length` 字段进行分层评估
3. **子集分析**: 建议分别报告4个子集的性能，以分析模型的不同能力维度

## Citation

If you use this dataset, please cite:

```
PAC-Test: A Comprehensive Benchmark for Long-Context Understanding
```
"""
    
    with open(data_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"Saved README to {data_dir / 'README.md'}")
    print("\n" + "=" * 60)
    print("Dataset construction completed!")
    print(f"Output directory: {data_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
