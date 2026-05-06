#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test Dataset Knowledge Base Generator
生成核心知识库文件，供后续数据集构建使用。
"""

import json
import random
import os

random.seed(42)

# ==================== 1. 事实库 (子集A/B) ====================

def generate_facts_library():
    """生成5个领域的事实库，每个领域50个事实"""
    
    domains = {
        "computer_science": {
            "templates": [
                ("{entity}的{attr}是{val}", [
                    ("RoPE", "提出者", "Su et al."),
                    ("LoRA", "提出年份", "2021"),
                    ("FlashAttention", "核心优化", "IO感知注意力"),
                    ("Mamba", "基础架构", "SSM"),
                    ("Mixtral", "核心机制", "MoE"),
                    ("QLoRA", "量化精度", "4-bit"),
                    ("RAG", "核心组件", "检索器"),
                    ("BERT", "预训练任务", "MLM"),
                    ("GPT-3", "参数量", "175B"),
                    ("T5", "统一框架", "Text-to-Text"),
                    ("ViT", "核心思想", "Patch嵌入"),
                    ("ResNet", "关键创新", "残差连接"),
                    ("BatchNorm", "提出年份", "2015"),
                    ("Dropout", "核心思想", "随机失活"),
                    ("Adam", "优化特性", "自适应学习率"),
                    ("LSTM", "核心机制", "门控循环"),
                    ("GRU", "门控数量", "2个"),
                    ("Transformer", "核心组件", "自注意力"),
                    ("BERT-large", "层数", "24层"),
                    ("GPT-2", "参数量", "1.5B"),
                    ("Word2Vec", "核心方法", "Skip-gram"),
                    ("GloVe", "训练目标", "全局统计"),
                    ("FastText", "核心扩展", "子词信息"),
                    ("ELMo", "核心思想", "双向语言模型"),
                    ("ULMFiT", "迁移方法", "逐层解冻"),
                    ("Seq2Seq", "编码器类型", "RNN"),
                    ("Attention", "起源论文", "Bahdanau"),
                    ("LayerNorm", "归一化维度", "特征维度"),
                    ("GeLU", "激活函数类型", "非线性"),
                    ("SwiGLU", "门控机制", "Swish+GLU"),
                    ("RMSNorm", "归一化方式", "均方根"),
                    ("ALiBi", "位置编码方式", "线性偏置"),
                    ("xFormers", "核心优化", "内存高效"),
                    ("DeepSpeed", "核心技术", "ZeRO"),
                    ("Megatron-LM", "并行策略", "张量并行"),
                    ("Colossal-AI", "核心特性", "统一并行"),
                    ("vLLM", "核心优化", "PagedAttention"),
                    ("TensorRT", "优化目标", "推理加速"),
                    ("ONNX", "标准格式", "开放神经网络"),
                    ("HuggingFace", "核心平台", "Transformers"),
                    ("WandB", "核心功能", "实验追踪"),
                    ("MLflow", "管理平台", "机器学习生命周期"),
                    ("DVC", "版本控制对象", "数据与模型"),
                    ("Optuna", "优化算法", "贝叶斯优化"),
                    ("Ray", "核心功能", "分布式计算"),
                    ("Horovod", "通信后端", "Ring-Allreduce"),
                    ("Fairseq", "开发团队", "Meta AI"),
                    ("OpenNMT", "核心框架", "神经机器翻译"),
                    ("spaCy", "核心功能", "工业级NLP"),
                    ("StanfordNLP", "基础模型", "CoreNLP"),
                ])
            ]
        },
        "medicine": {
            "templates": [
                ("{entity}的{attr}是{val}", [
                    ("CRISPR-Cas9", "发现者", "Doudna & Charpentier"),
                    ("mRNA疫苗", "核心技术", "脂质纳米颗粒"),
                    ("PD-1抑制剂", "代表药物", "Pembrolizumab"),
                    ("CAR-T疗法", "靶向对象", "CD19"),
                    ("AlphaFold", "预测目标", "蛋白质结构"),
                    ("青霉素", "发现者", "Fleming"),
                    ("胰岛素", "发现者", "Banting & Best"),
                    ("阿司匹林", "原始来源", "柳树皮"),
                    ("他汀类药物", "作用靶点", "HMG-CoA还原酶"),
                    ("二甲双胍", "原始来源", "山羊豆"),
                    ("吗啡", "提取来源", "罂粟"),
                    ("奎宁", "提取来源", "金鸡纳树皮"),
                    ("紫杉醇", "提取来源", "太平洋紫杉"),
                    ("青蒿素", "发现者", "屠呦呦"),
                    ("利妥昔单抗", "靶点", "CD20"),
                    ("曲妥珠单抗", "靶点", "HER2"),
                    ("贝伐珠单抗", "靶点", "VEGF"),
                    ("西妥昔单抗", "靶点", "EGFR"),
                    ("阿达木单抗", "靶点", "TNF-α"),
                    ("英夫利昔单抗", "靶点", "TNF-α"),
                    ("奥司他韦", "作用靶点", "神经氨酸酶"),
                    ("瑞德西韦", "作用机制", "RNA聚合酶抑制"),
                    (" Paxlovid", "作用靶点", "3CL蛋白酶"),
                    ("HPV疫苗", "预防疾病", "宫颈癌"),
                    ("乙肝疫苗", "抗原类型", "表面抗原"),
                    ("百白破疫苗", "预防疾病数", "3种"),
                    ("MMR疫苗", "预防疾病数", "3种"),
                    ("卡介苗", "预防疾病", "结核病"),
                    ("脊髓灰质炎疫苗", "发明者", "Sabin"),
                    ("天花疫苗", "发明者", "Jenner"),
                    ("PCR技术", "发明者", "Mullis"),
                    ("DNA测序", "第一代方法", "Sanger法"),
                    ("基因芯片", "核心技术", "杂交测序"),
                    ("流式细胞术", "检测对象", "细胞表面标记"),
                    ("免疫组化", "检测对象", "抗原定位"),
                    ("ELISA", "检测原理", "抗原抗体结合"),
                    ("Western Blot", "检测对象", "蛋白质"),
                    ("Northern Blot", "检测对象", "RNA"),
                    ("Southern Blot", "检测对象", "DNA"),
                    ("CT扫描", "成像原理", "X射线断层"),
                    ("MRI", "成像原理", "核磁共振"),
                    ("PET", "成像原理", "正电子发射"),
                    ("超声成像", "成像原理", "声波反射"),
                    ("X射线", "发现者", "Röntgen"),
                    ("心电图", "发明者", "Einthoven"),
                    ("脑电图", "检测对象", "脑电活动"),
                    ("内窥镜", "检查部位", "空腔脏器"),
                    ("腹腔镜", "手术特点", "微创手术"),
                    ("达芬奇手术系统", "技术类型", "机器人辅助"),
                    ("3D打印器官", "应用方向", "组织工程"),
                ])
            ]
        },
        "law": {
            "templates": [
                ("{entity}的{attr}是{val}", [
                    ("《民法典》", "施行时间", "2021年"),
                    ("《刑法》", "最新修正案", "十二"),
                    ("《宪法》", "现行版本年份", "1982年"),
                    ("《劳动合同法》", "施行时间", "2008年"),
                    ("《公司法》", "最新修订年份", "2023年"),
                    ("《著作权法》", "保护期限", "作者终生加50年"),
                    ("《专利法》", "发明专利期限", "20年"),
                    ("《商标法》", "注册商标有效期", "10年"),
                    ("《反垄断法》", "执法机构", "市场监管总局"),
                    ("《数据安全法》", "施行时间", "2021年"),
                    ("《个人信息保护法》", "施行时间", "2021年"),
                    ("《网络安全法》", "施行时间", "2017年"),
                    ("《电子商务法》", "施行时间", "2019年"),
                    ("《环境保护法》", "修订年份", "2014年"),
                    ("《食品安全法》", "最新修订", "2021年"),
                    ("《药品管理法》", "最新修订", "2019年"),
                    ("《建筑法》", "调整对象", "建筑活动"),
                    ("《道路交通安全法》", "最新修订", "2021年"),
                    ("《税收征管法》", "调整对象", "税收征收"),
                    ("《仲裁法》", "基本原则", "一裁终局"),
                    ("《民事诉讼法》", "基本制度", "两审终审"),
                    ("《刑事诉讼法》", "基本原则", "无罪推定"),
                    ("《行政诉讼法》", "审查对象", "行政行为"),
                    ("《国家赔偿法》", "赔偿类型", "行政与刑事"),
                    ("《物权法》", "现行状态", "纳入民法典"),
                    ("《合同法》", "现行状态", "纳入民法典"),
                    ("《侵权责任法》", "现行状态", "纳入民法典"),
                    ("《婚姻法》", "现行状态", "纳入民法典"),
                    ("《继承法》", "现行状态", "纳入民法典"),
                    ("《收养法》", "现行状态", "纳入民法典"),
                    ("《担保法》", "现行状态", "纳入民法典"),
                    ("《海商法》", "调整对象", "海上运输"),
                    ("《票据法》", "调整对象", "票据行为"),
                    ("《保险法》", "基本原则", "最大诚信"),
                    ("《证券法》", "最新修订", "2019年"),
                    ("《信托法》", "核心制度", "信托财产独立"),
                    ("《破产法》", "适用范围", "企业法人"),
                    ("《劳动法》", "核心原则", "保护劳动者"),
                    ("《工会法》", "组织性质", "职工自愿结合"),
                    ("《公务员法》", "管理原则", "分类管理"),
                    ("《法官法》", "任职条件", "法律职业资格"),
                    ("《检察官法》", "任职条件", "法律职业资格"),
                    ("《律师法》", "执业许可", "律师执业证书"),
                    ("《公证法》", "证明效力", "法定证据效力"),
                    ("《监狱法》", "执行目的", "惩罚与改造"),
                    ("《戒严法》", "适用条件", "严重骚乱"),
                    ("《国防法》", "国防原则", "全民自卫"),
                    ("《兵役法》", "兵役制度", "义务兵与志愿兵"),
                    ("《现役军官法》", "军官来源", "院校培训"),
                    ("《预备役军官法》", "预备役分类", "军官与士兵"),
                    ("《香港基本法》", "法律地位", "宪制性法律"),
                ])
            ]
        },
        "finance": {
            "templates": [
                ("{entity}的{attr}是{val}", [
                    ("市盈率", "计算方式", "股价除以每股收益"),
                    ("市净率", "计算方式", "股价除以每股净资产"),
                    ("夏普比率", "衡量指标", "风险调整后收益"),
                    ("贝塔系数", "衡量指标", "系统性风险"),
                    ("VaR", "风险度量", "最大潜在损失"),
                    ("Black-Scholes", "应用领域", "期权定价"),
                    ("CAPM", "核心概念", "资本资产定价"),
                    ("APT", "理论基础", "无套利均衡"),
                    ("有效市场假说", "提出者", "Fama"),
                    ("行为金融学", "理论基础", "心理偏差"),
                    ("量化宽松", "实施主体", "中央银行"),
                    ("逆回购", "交易方向", "央行投放流动性"),
                    ("MLF", "操作期限", "1年期"),
                    ("LPR", "报价机制", "报价行加点"),
                    ("Shibor", "利率类型", "银行间拆借"),
                    ("LIBOR", "替代基准", "SOFR"),
                    ("Federal Funds Rate", "决策机构", "FOMC"),
                    ("存款准备金", "调整机构", "中央银行"),
                    ("M2", "货币层次", "广义货币"),
                    ("M1", "货币层次", "狭义货币"),
                    ("CPI", "衡量对象", "消费品价格"),
                    ("PPI", "衡量对象", "生产者价格"),
                    ("GDP", "核算方法", "支出法"),
                    ("GNP", "计算范围", "国民概念"),
                    ("PMI", "荣枯线", "50%"),
                    ("失业率", "统计标准", "调查失业率"),
                    ("基尼系数", "衡量指标", "收入分配"),
                    ("恩格尔系数", "衡量指标", "食品支出占比"),
                    ("菲利普斯曲线", "关系描述", "通胀与失业"),
                    ("IS-LM模型", "分析对象", "产品货币市场"),
                    ("AD-AS模型", "分析对象", "总需求总供给"),
                    ("蒙代尔-弗莱明", "分析框架", "开放经济"),
                    ("购买力平价", "理论基础", "一价定律"),
                    ("利率平价", "核心关系", "利差与汇率"),
                    ("J曲线效应", "描述现象", "贸易收支调整"),
                    ("特里芬难题", "内在矛盾", "储备货币"),
                    ("巴塞尔协议III", "核心指标", "资本充足率"),
                    ("CAMELS评级", "评估维度", "6项"),
                    ("IPO", "审核制度", "注册制"),
                    ("借壳上市", "实质行为", "资产重组"),
                    ("绿鞋机制", "超额配售", "15%"),
                    ("对赌协议", "法律性质", "估值调整"),
                    ("VIE架构", "协议控制", "可变利益实体"),
                    ("SPAC", "上市方式", "特殊目的收购"),
                    ("REITs", "基础资产", "不动产"),
                    ("ABS", "底层资产", "证券化资产"),
                    ("CDO", "产品结构", "债务担保证券"),
                    ("CDS", "本质功能", "信用保险"),
                    ("股指期货", "标的指数", "沪深300"),
                    ("期权希腊值", "风险指标", "Delta/Gamma"),
                ])
            ]
        },
        "education": {
            "templates": [
                ("{entity}的{attr}是{val}", [
                    ("建构主义", "代表人物", "Piaget"),
                    ("行为主义", "代表人物", "Skinner"),
                    ("认知主义", "代表人物", "Ausubel"),
                    ("人本主义", "代表人物", "Rogers"),
                    ("多元智能理论", "提出者", "Gardner"),
                    ("成长型思维", "提出者", "Dweck"),
                    ("最近发展区", "提出者", "Vygotsky"),
                    ("脚手架理论", "理论基础", "最近发展区"),
                    ("布鲁姆分类法", "认知层次", "6个层次"),
                    ("ADDIE模型", "阶段数", "5个阶段"),
                    ("SAM模型", "迭代特点", "敏捷开发"),
                    ("翻转课堂", "核心特征", "课前自学"),
                    ("项目式学习", "核心方法", "真实问题驱动"),
                    ("探究式学习", "理论基础", "建构主义"),
                    ("合作学习", "核心要素", "积极互赖"),
                    ("差异化教学", "实施依据", "学习者特征"),
                    ("形成性评价", "评价时机", "教学过程中"),
                    ("总结性评价", "评价时机", "教学结束后"),
                    ("诊断性评价", "评价目的", "了解起点能力"),
                    (" rubric", "评价工具", "评分准则"),
                    ("自我效能感", "提出者", "Bandura"),
                    ("目标设定理论", "提出者", "Locke"),
                    ("期望价值理论", "核心变量", "期望与价值"),
                    ("成就目标理论", "分类维度", "掌握与成绩"),
                    ("认知负荷理论", "提出者", "Sweller"),
                    ("工作记忆", "容量限制", "7±2个组块"),
                    ("长时记忆", "存储特点", "容量无限"),
                    ("精细加工策略", "本质操作", "建立联系"),
                    ("组织策略", "本质操作", "归类整合"),
                    ("复述策略", "本质操作", "重复记忆"),
                    ("元认知策略", "监控对象", "认知过程"),
                    ("迁移理论", "经典理论", "共同要素说"),
                    ("情境学习", "理论基础", "合法的边缘参与"),
                    ("隐性课程", "存在形式", "非正式学习"),
                    ("螺旋式课程", "提出者", "Bruner"),
                    ("核心素养", "维度数", "3个维度"),
                    ("STEAM教育", "学科融合", "5个学科"),
                    ("计算思维", "核心要素", "4个维度"),
                    ("信息素养", "核心能力", "检索与评估"),
                    ("媒介素养", "核心能力", "批判性理解"),
                    ("数字素养", "欧盟框架", "DigComp"),
                    ("微格教学", "训练单元", "小规模教学"),
                    ("行动研究", "研究主体", "实践者"),
                    ("课例研究", "起源国家", "日本"),
                    ("学习分析", "数据来源", "学习管理系统"),
                    ("教育数据挖掘", "核心技术", "模式发现"),
                    ("自适应学习", "技术基础", "知识追踪"),
                    ("智能导学系统", "核心功能", "个性化反馈"),
                    ("MOOC", "平台特征", "大规模开放"),
                    ("SPOC", "平台特征", "小规模私有"),
                    ("混合式学习", " blend比例", "线上与线下结合"),
                ])
            ]
        }
    }
    
    facts = []
    fact_id = 0
    for domain, data in domains.items():
        for template, entries in data["templates"]:
            for entity, attr, val in entries:
                fact_text = template.format(entity=entity, attr=attr, val=val)
                facts.append({
                    "id": fact_id,
                    "domain": domain,
                    "entity_name": entity,
                    "attribute": attr,
                    "value": val,
                    "fact_text": fact_text
                })
                fact_id += 1
    
    return facts

# ==================== 2. 易混淆实体对 (子集C) ====================

def generate_entity_pairs():
    """生成4种相似度等级的易混淆实体对"""
    
    pairs = {
        "same_name_same_domain": [
            {
                "name": "李明",
                "domain": "医学",
                "entities": [
                    {"attribute": "所在医院", "value": "协和医院", "distinguishing": "心内科主任"},
                    {"attribute": "所在医院", "value": "华西医院", "distinguishing": "神经外科副主任"}
                ]
            },
            {
                "name": "王磊",
                "domain": "计算机科学",
                "entities": [
                    {"attribute": "工作单位", "value": "清华大学", "distinguishing": " NLP实验室负责人"},
                    {"attribute": "工作单位", "value": "北京大学", "distinguishing": "计算机视觉实验室教授"}
                ]
            },
            {
                "name": "张伟",
                "domain": "法律",
                "entities": [
                    {"attribute": "执业领域", "value": "知识产权", "distinguishing": "专利诉讼律师"},
                    {"attribute": "执业领域", "value": "刑事辩护", "distinguishing": "资深刑辩律师"}
                ]
            },
            {
                "name": "刘洋",
                "domain": "金融",
                "entities": [
                    {"attribute": "任职机构", "value": "高盛", "distinguishing": "量化投资总监"},
                    {"attribute": "任职机构", "value": "摩根士丹利", "distinguishing": "并购重组部主管"}
                ]
            },
            {
                "name": "陈静",
                "domain": "教育",
                "entities": [
                    {"attribute": "任职学校", "value": "北京师范大学", "distinguishing": "教育技术学教授"},
                    {"attribute": "任职学校", "value": "华东师范大学", "distinguishing": "课程与教学论专家"}
                ]
            },
            {
                "name": "赵强",
                "domain": "医学",
                "entities": [
                    {"attribute": "所在科室", "value": "骨科", "distinguishing": "关节置换专家"},
                    {"attribute": "所在科室", "value": "骨科", "distinguishing": "脊柱外科主任"}
                ]
            },
            {
                "name": "孙丽",
                "domain": "计算机科学",
                "entities": [
                    {"attribute": "研究方向", "value": "自然语言处理", "distinguishing": "大语言模型优化"},
                    {"attribute": "研究方向", "value": "自然语言处理", "distinguishing": "机器翻译系统"}
                ]
            },
            {
                "name": "周涛",
                "domain": "法律",
                "entities": [
                    {"attribute": "执业地区", "value": "上海", "distinguishing": "国际贸易仲裁专家"},
                    {"attribute": "执业地区", "value": "上海", "distinguishing": "海事海商律师"}
                ]
            },
            {
                "name": "吴敏",
                "domain": "金融",
                "entities": [
                    {"attribute": "专业资格", "value": "CFA", "distinguishing": "固定收益分析师"},
                    {"attribute": "专业资格", "value": "CFA", "distinguishing": "权益投资策略师"}
                ]
            },
            {
                "name": "郑华",
                "domain": "教育",
                "entities": [
                    {"attribute": "教学学段", "value": "高中", "distinguishing": "数学特级教师"},
                    {"attribute": "教学学段", "value": "高中", "distinguishing": "物理竞赛教练"}
                ]
            },
            {
                "name": "黄军",
                "domain": "医学",
                "entities": [
                    {"attribute": "所在医院", "value": "301医院", "distinguishing": "肝胆外科主任"},
                    {"attribute": "所在医院", "value": "301医院", "distinguishing": "泌尿外科教授"}
                ]
            },
            {
                "name": "林峰",
                "domain": "计算机科学",
                "entities": [
                    {"attribute": "毕业院校", "value": "MIT", "distinguishing": "分布式系统博士"},
                    {"attribute": "毕业院校", "value": "MIT", "distinguishing": "数据库系统博士"}
                ]
            },
        ],
        "same_name_diff_domain": [
            {
                "name": "王芳",
                "entities": [
                    {"domain": "医学", "attribute": "专业方向", "value": "肿瘤免疫治疗", "distinguishing": "肿瘤内科主治医师"},
                    {"domain": "教育", "attribute": "专业方向", "value": "教育心理学", "distinguishing": "高校心理咨询中心主任"}
                ]
            },
            {
                "name": "李强",
                "entities": [
                    {"domain": "法律", "attribute": "执业领域", "value": "公司法", "distinguishing": "红圈所合伙人"},
                    {"domain": "金融", "attribute": "从业方向", "value": "风险管理", "distinguishing": "外资银行风控总监"}
                ]
            },
            {
                "name": "张敏",
                "entities": [
                    {"domain": "计算机科学", "attribute": "技术专长", "value": "强化学习", "distinguishing": "OpenAI研究员"},
                    {"domain": "医学", "attribute": "技术专长", "value": "医学影像AI", "distinguishing": "放射科副主任医师"}
                ]
            },
            {
                "name": "刘洋",
                "entities": [
                    {"domain": "教育", "attribute": "研究领域", "value": "STEM教育", "distinguishing": "教科院研究员"},
                    {"domain": "计算机科学", "attribute": "研究领域", "value": "推荐系统", "distinguishing": "字节跳动算法专家"}
                ]
            },
            {
                "name": "陈杰",
                "entities": [
                    {"domain": "金融", "attribute": "专业领域", "value": "行为金融学", "distinguishing": "商学院教授"},
                    {"domain": "法律", "attribute": "专业领域", "value": "金融监管法", "distinguishing": "央行法律顾问"}
                ]
            },
        ],
        "diff_name_same_domain": [
            {
                "domain": "医学",
                "entities": [
                    {"name": "张伟", "attribute": "所在医院", "value": "协和医院", "distinguishing": "心外科主任"},
                    {"name": "张伟强", "attribute": "所在医院", "value": "天坛医院", "distinguishing": "神经外科主任"}
                ]
            },
            {
                "domain": "计算机科学",
                "entities": [
                    {"name": "李明", "attribute": "研究方向", "value": "计算机视觉", "distinguishing": "目标检测专家"},
                    {"name": "李明远", "attribute": "研究方向", "value": "自然语言处理", "distinguishing": "预训练模型研究者"}
                ]
            },
            {
                "domain": "法律",
                "entities": [
                    {"name": "王刚", "attribute": "执业领域", "value": "知识产权", "distinguishing": "商标诉讼律师"},
                    {"name": "王刚强", "attribute": "执业领域", "value": "专利诉讼", "distinguishing": "专利无效宣告专家"}
                ]
            },
            {
                "domain": "金融",
                "entities": [
                    {"name": "赵敏", "attribute": "任职机构", "value": "中金公司", "distinguishing": "投行部董事总经理"},
                    {"name": "赵敏华", "attribute": "任职机构", "value": "中信证券", "distinguishing": "研究部首席分析师"}
                ]
            },
            {
                "domain": "教育",
                "entities": [
                    {"name": "孙丽", "attribute": "任职学校", "value": "人大附中", "distinguishing": "高中数学教研组长"},
                    {"name": "孙丽华", "attribute": "任职学校", "value": "北京四中", "distinguishing": "初中数学骨干教师"}
                ]
            },
        ],
        "completely_different": [
            {
                "entities": [
                    {"name": "爱因斯坦", "domain": "物理学", "attribute": "主要贡献", "value": "相对论", "distinguishing": "诺贝尔物理学奖得主"},
                    {"name": "莎士比亚", "domain": "文学", "attribute": "代表作品", "value": "哈姆雷特", "distinguishing": "英国文艺复兴剧作家"}
                ]
            },
            {
                "entities": [
                    {"name": "乔布斯", "domain": "科技", "attribute": "创立公司", "value": "Apple", "distinguishing": "苹果公司联合创始人"},
                    {"name": "贝多芬", "domain": "音乐", "attribute": "代表作品", "value": "第九交响曲", "distinguishing": "古典主义作曲家"}
                ]
            },
            {
                "entities": [
                    {"name": "牛顿", "domain": "物理学", "attribute": "主要发现", "value": "万有引力", "distinguishing": "经典力学奠基人"},
                    {"name": "达芬奇", "domain": "艺术", "attribute": "代表作品", "value": "蒙娜丽莎", "distinguishing": "文艺复兴全才"}
                ]
            },
            {
                "entities": [
                    {"name": "图灵", "domain": "计算机科学", "attribute": "主要贡献", "value": "图灵机", "distinguishing": "计算机科学之父"},
                    {"name": "弗洛伊德", "domain": "心理学", "attribute": "主要理论", "value": "精神分析", "distinguishing": "精神分析学派创始人"}
                ]
            },
            {
                "entities": [
                    {"name": "马斯克", "domain": "科技", "attribute": "领导公司", "value": "Tesla", "distinguishing": "SpaceX创始人"},
                    {"name": "莫扎特", "domain": "音乐", "attribute": "代表作品", "value": "魔笛", "distinguishing": "古典主义音乐家"}
                ]
            },
        ]
    }
    
    return pairs

# ==================== 3. 事实链 (子集D) ====================

def generate_fact_chains():
    """生成多跳事实链"""
    
    chains = {
        "person_relationship": [
            {
                "hops": [
                    {"entity": "张三", "relation": "导师", "target": "李四", "evidence": "张三的博士生导师是李四教授"},
                    {"entity": "李四", "relation": "合作者", "target": "王五", "evidence": "李四教授与王五博士是长期科研合作者"},
                    {"entity": "王五", "relation": "就职于", "target": "Google DeepMind", "evidence": "王五博士目前就职于Google DeepMind"}
                ],
                "question_template": "张三的导师的主要合作者目前就职于哪家公司？",
                "answer": "Google DeepMind"
            },
            {
                "hops": [
                    {"entity": "小明", "relation": "导师", "target": "小红", "evidence": "小明的硕士导师是小红副教授"},
                    {"entity": "小红", "relation": "导师", "target": "老王", "evidence": "小红副教授的导师是老王院士"},
                    {"entity": "老王", "relation": "荣誉", "target": "国家最高科技奖", "evidence": "老王院士于2020年获得国家最高科技奖"}
                ],
                "question_template": "小明的导师的导师获得了什么荣誉？",
                "answer": "国家最高科技奖"
            },
            {
                "hops": [
                    {"entity": "Alice", "relation": "同事", "target": "Bob", "evidence": "Alice与Bob是MIT CSAIL的同事"},
                    {"entity": "Bob", "relation": "学生", "target": "Charlie", "evidence": "Bob指导的博士生Charlie于今年毕业"},
                    {"entity": "Charlie", "relation": "创立", "target": "OpenAI", "evidence": "Charlie博士毕业后参与创立了OpenAI"}
                ],
                "question_template": "Alice的同事的学生参与创立了哪家公司？",
                "answer": "OpenAI"
            },
        ],
        "geography": [
            {
                "hops": [
                    {"entity": "深圳市", "relation": "属于", "target": "广东省", "evidence": "深圳市是广东省下辖的副省级城市"},
                    {"entity": "广东省", "relation": "接壤", "target": "湖南省", "evidence": "广东省北部与湖南省接壤"},
                    {"entity": "湖南省", "relation": "省会", "target": "长沙市", "evidence": "湖南省的省会是长沙市"}
                ],
                "question_template": "深圳市所在省份的邻省的省会是什么？",
                "answer": "长沙市"
            },
            {
                "hops": [
                    {"entity": "柏林", "relation": "属于", "target": "德国", "evidence": "柏林是德国的首都和最大城市"},
                    {"entity": "德国", "relation": "邻国", "target": "法国", "evidence": "德国西部与法国接壤"},
                    {"entity": "法国", "relation": "首都", "target": "巴黎", "evidence": "法国的首都是巴黎"}
                ],
                "question_template": "柏林所在国家的邻国的首都是什么？",
                "answer": "巴黎"
            },
        ],
        "organization": [
            {
                "hops": [
                    {"entity": "抖音", "relation": "隶属于", "target": "字节跳动", "evidence": "抖音是字节跳动旗下的短视频平台"},
                    {"entity": "字节跳动", "relation": "投资", "target": "MiniMax", "evidence": "字节跳动领投了MiniMax的最新一轮融资"},
                    {"entity": "MiniMax", "relation": "创始人", "target": "闫俊杰", "evidence": "MiniMax的创始人兼CEO是闫俊杰"}
                ],
                "question_template": "抖音母公司投资的AI公司的创始人是谁？",
                "answer": "闫俊杰"
            },
        ],
        "timeline": [
            {
                "hops": [
                    {"entity": "第一次工业革命", "relation": "早于", "target": "第二次工业革命", "evidence": "第一次工业革命始于18世纪60年代"},
                    {"entity": "第二次工业革命", "relation": "早于", "target": "第三次工业革命", "evidence": "第二次工业革命始于19世纪70年代"},
                    {"entity": "第三次工业革命", "relation": "核心特征", "target": "信息技术", "evidence": "第三次工业革命以信息技术为核心"}
                ],
                "question_template": "在第二次工业革命之后的工业革命的核心特征是什么？",
                "answer": "信息技术"
            },
        ]
    }
    
    return chains

# ==================== 4. 填充文本模板 ====================

def generate_filler_templates():
    """生成用于填充的文本片段模板"""
    
    templates = {
        "computer_science": [
            "近年来，{topic}领域的研究取得了显著进展。",
            "多项研究表明，{topic}在{application}场景中表现出色。",
            "从技术架构来看，{topic}采用了{method}作为其核心方法。",
            "实验结果表明，该方法在{dataset}数据集上达到了{metric}的准确率。",
            "这一突破为{field}的发展开辟了新的方向。",
            "与此同时，研究人员也在探索{topic}与{related}的结合。",
            "值得注意的是，{topic}在处理{challenge}问题时仍存在局限。",
            "未来的研究将聚焦于提升{topic}的{property}。",
            "从工程实践角度，{topic}的部署需要考虑{factor}等因素。",
            "综述性文献指出，{topic}已成为{field}的主流方法之一。",
            "代码实现上，{topic}通常基于{framework}框架进行开发。",
            "性能优化方面，研究者们提出了{method}等改进策略。",
            "在工业界，{topic}已被广泛应用于{application}等场景。",
            "对比实验显示，{topic}相比{baseline}在{metric}上提升了{percent}。",
            "理论分析表明，{topic}的时间复杂度为{complexity}。",
        ],
        "medicine": [
            "临床研究表明，{topic}在{condition}治疗中显示出良好疗效。",
            "从病理机制来看，{topic}主要通过{mechanism}发挥作用。",
            "多项随机对照试验证实，{topic}可将{outcome}风险降低{percent}。",
            "在药物代谢方面，{topic}主要通过{organ}进行代谢。",
            "常见的不良反应包括{symptom}等，但通常较为轻微。",
            "从分子机制角度，{topic}能够靶向{target}发挥作用。",
            "药代动力学研究显示，{topic}的半衰期约为{duration}。",
            "联合用药方案中，{topic}常与{drug}搭配使用。",
            "基因组学研究提示，{gene}基因多态性可能影响{topic}的疗效。",
            "在真实世界研究中，{topic}的临床治愈率达到了{percent}。",
            "免疫学机制方面，{topic}能够调节{cell}的功能。",
            "从公共卫生角度，{topic}的普及可显著降低{disease}的发病率。",
        ],
        "law": [
            "从立法目的来看，{topic}旨在保护{interest}。",
            "司法实践中，{topic}的适用需要考虑{factor}等因素。",
            "根据相关司法解释，{topic}的认定标准包括{standard}。",
            "比较法研究表明，{country}在{topic}方面采取了{approach}。",
            "从法学理论角度，{topic}涉及{theory}等核心理论。",
            "典型案例分析显示，{topic}的裁判尺度存在{issue}。",
            "立法沿革上，{topic}经历了从{old}到{new}的转变。",
            "行政执法中，{topic}的处罚标准通常为{penalty}。",
            "从权利义务配置来看，{topic}平衡了{party1}与{party2}的利益。",
            "法律经济学分析表明，{topic}的制度设计具有{feature}。",
        ],
        "finance": [
            "从宏观经济角度，{topic}受到{factor}等因素的影响。",
            "市场数据显示，{topic}在过去{period}内波动率为{percent}。",
            "投资组合理论表明，{topic}与{asset}的相关性为{correlation}。",
            "风险管理实践中，{topic}的VaR值通常设定为{value}。",
            "从估值模型来看，{topic}适用于{model}进行分析。",
            "行业研究发现，{topic}的盈利能力受{cycle}周期影响显著。",
            "监管政策方面，{topic}需要遵守{regulation}等规定。",
            "从市场结构来看，{topic}属于{structure}类型。",
            "行为金融学解释，投资者对{topic}存在{bias}等认知偏差。",
            "国际比较显示，{topic}在{country}市场的发展更为成熟。",
        ],
        "education": [
            "从学习理论角度，{topic}基于{theory}的理论框架。",
            "教学实践表明，{topic}能有效提升{skill}。",
            "评估研究数据显示，采用{topic}的学生{outcome}提高了{percent}。",
            "从课程设计来看，{topic}需要结合{element}等要素。",
            "教师专业发展方面，{topic}对教师的{ability}提出了新要求。",
            "技术整合视角下，{topic}可以借助{technology}实现优化。",
            "跨文化研究表明，{topic}在不同{context}中的效果存在差异。",
            "从认知负荷理论分析，{topic}的设计应避免{problem}。",
            "教育政策层面，{topic}已被纳入{level}课程标准。",
            " longitudinal研究显示，{topic}的效应在{period}后仍持续存在。",
        ]
    }
    
    # 每个领域的填充变量库
    variables = {
        "computer_science": {
            "topic": ["深度学习", "Transformer", "图神经网络", "联邦学习", "对比学习", "自监督学习", "多模态学习"],
            "application": ["图像分类", "机器翻译", "推荐系统", "语音识别", "自动驾驶", "医疗诊断"],
            "method": ["注意力机制", "卷积运算", "递归结构", "图卷积", "对比损失", "掩码预测"],
            "dataset": ["ImageNet", "COCO", "WMT", "GLUE", "LAION-5B"],
            "field": ["计算机视觉", "自然语言处理", "语音处理", "多模态学习"],
            "related": ["强化学习", "迁移学习", "元学习", "知识蒸馏"],
            "challenge": ["长尾分布", "数据稀缺", "域迁移", "实时推理"],
            "property": ["泛化能力", "鲁棒性", "可解释性", "计算效率"],
            "factor": ["延迟", "吞吐量", "内存占用", "能耗"],
            "framework": ["PyTorch", "TensorFlow", "JAX", "MindSpore"],
            "baseline": ["ResNet", "LSTM", "BERT-base", "Random Forest"],
            "metric": ["准确率", "F1分数", "BLEU", "mAP"],
            "percent": ["5%", "10%", "15%", "20%"],
            "complexity": ["O(n)", "O(n log n)", "O(n²)"],
        },
        "medicine": {
            "topic": ["免疫检查点抑制剂", "CAR-T细胞疗法", "基因编辑技术", "mRNA疫苗", "精准医疗"],
            "condition": ["非小细胞肺癌", "淋巴瘤", "遗传性疾病", "传染病", "罕见病"],
            "mechanism": ["PD-1/PD-L1通路阻断", "T细胞活化", "DNA双链断裂修复"],
            "outcome": ["无进展生存", "总生存", "复发", "转移"],
            "percent": ["30%", "40%", "50%", "60%"],
            "organ": ["肝脏", "肾脏", "肠道"],
            "target": ["特定抗原", "酪氨酸激酶", "血管生成因子"],
            "duration": ["8小时", "12小时", "24小时"],
            "drug": ["化疗药物", "靶向药物", "激素类药物"],
            "gene": ["BRCA1", "EGFR", "ALK"],
            "cell": ["T细胞", "B细胞", "巨噬细胞"],
            "disease": ["心血管疾病", "糖尿病", "癌症"],
            "symptom": ["疲劳", "恶心", "皮疹"],
        },
        "law": {
            "topic": ["知识产权", "消费者权益", "数据隐私", "劳动合同", "环境保护"],
            "interest": ["创新激励", "弱势群体", "个人信息安全", "劳动者权益", "生态环境"],
            "factor": ["主观恶意", "损害后果", "因果关系"],
            "standard": ["合理性标准", "必要性标准", "比例原则"],
            "country": ["德国", "美国", "日本", "欧盟"],
            "approach": ["严格责任", "过错责任", "无过错责任"],
            "theory": ["法经济学", "自然法学", "实证主义"],
            "issue": ["地区差异", "标准不统一", "适用困难"],
            "old": ["原则性规定", "单一标准", "行政处罚"],
            "new": ["精细化规则", "多元标准", "综合责任"],
            "penalty": ["罚款", "停业整顿", "吊销执照"],
            "party1": ["经营者", "用人单位", "侵权人"],
            "party2": ["消费者", "劳动者", "受害人"],
            "feature": ["效率性", "公平性", "激励相容"],
        },
        "finance": {
            "topic": ["股票估值", "债券收益率", "汇率波动", "大宗商品", "加密货币"],
            "factor": ["利率政策", "通胀预期", "地缘政治"],
            "period": ["一年", "三年", "五年"],
            "percent": ["15%", "20%", "25%"],
            "asset": ["国债", "黄金", "房地产"],
            "correlation": ["0.3", "0.5", "0.7"],
            "value": ["95%置信水平", "99%置信水平"],
            "model": ["DCF模型", "相对估值法", "期权定价模型"],
            "cycle": ["经济", "行业", "库存"],
            "regulation": ["巴塞尔协议", "信息披露", "反洗钱"],
            "structure": ["完全竞争", "寡头垄断", "垄断竞争"],
            "bias": ["过度自信", "锚定效应", "损失厌恶"],
            "country": ["美国", "英国", "新加坡"],
        },
        "education": {
            "topic": ["项目式学习", "翻转课堂", "游戏化学习", "自适应学习", "协作学习"],
            "theory": ["建构主义", "社会文化理论", "认知负荷理论"],
            "skill": ["批判性思维", "问题解决能力", "创造力", "合作能力"],
            "percent": ["20%", "30%", "40%"],
            "outcome": ["学业成绩", "学习动机", "知识保持率"],
            "element": ["真实情境", "协作工具", "形成性评价"],
            "ability": ["技术整合能力", "数据分析能力", "差异化教学能力"],
            "technology": ["学习分析", "虚拟现实", "人工智能"],
            "context": ["文化", "学段", "学科"],
            "problem": ["认知超载", "冗余信息", "分裂注意力"],
            "level": ["义务教育", "高中", "高等教育"],
            "period": ["六个月", "一年", "两年"],
        }
    }
    
    return templates, variables

# ==================== 5. 干扰文本模板 (子集B) ====================

def generate_distractor_templates():
    """生成三类干扰文本模板"""
    
    templates = {
        "in_domain_related": {
            "computer_science": [
                "在{topic}的研究中，{method}方法被广泛采用。",
                "{framework}框架为{topic}提供了高效的实现方案。",
                "最新的{topic}模型在{dataset}上刷新了记录。",
                "{topic}的核心挑战在于解决{challenge}问题。",
                "研究者们提出了基于{method}的{topic}改进方案。",
            ],
            "medicine": [
                "{topic}在{condition}的临床试验中显示出{percent}的缓解率。",
                "研究表明{topic}通过{mechanism}发挥治疗作用。",
                "{topic}与{drug}的联合应用成为新的研究热点。",
                "在{organ}代谢方面，{topic}表现出良好的药代动力学特征。",
                "{topic}治疗{condition}的{outcome}改善显著。",
            ],
            "law": [
                "{topic}案件在司法实践中通常适用{standard}进行判断。",
                "从{theory}角度分析，{topic}涉及多方利益平衡。",
                "{country}在{topic}领域采取了{approach}的立法模式。",
                "{topic}的{feature}特征使其在{interest}保护中发挥重要作用。",
                "近年来{topic}相关的{issue}问题引起了学界关注。",
            ],
            "finance": [
                "{topic}在{period}内的{metric}表现优于市场基准。",
                "{factor}的变化对{topic}的价格产生显著影响。",
                "从{model}角度评估，{topic}的当前估值处于{level}。",
                "{topic}与{asset}的相关性系数约为{correlation}。",
                "{regulation}政策对{topic}市场产生了深远影响。",
            ],
            "education": [
                "{topic}在提升学生{skill}方面表现出显著效果。",
                "基于{theory}的{topic}设计原则已被广泛接受。",
                "{technology}技术为{topic}的实施提供了新的可能。",
                "研究表明{topic}对不同{context}的学习者效果存在差异。",
                "{topic}在{level}教育阶段的应用取得了积极成效。",
            ]
        },
        "out_domain_unrelated": {
            "general": [
                "今天天气晴朗，适合外出散步。公园里的花开得很漂亮，吸引了不少游客。",
                "这道菜的做法很简单，只需要准备食材然后按照步骤烹饪即可。",
                "最近看了一部电影，剧情非常精彩，演员的表演也很到位。",
                "旅游是一种很好的放松方式，可以让人开阔眼界，体验不同的文化。",
                "运动对健康非常重要，每天坚持锻炼可以增强体质，提高免疫力。",
                "读书是一种很好的习惯，可以让人获取知识，提升思维能力。",
                "音乐能够陶冶情操，不同类型的音乐可以带来不同的情感体验。",
                "环保是每个人的责任，从小事做起，共同保护我们的地球家园。",
                "科技改变了我们的生活方式，智能手机让沟通变得更加便捷。",
                "家庭教育对孩子的成长至关重要，父母的言传身教影响深远。",
            ]
        },
        "random_noise": {
            "patterns": [
                "{word1} {word2} {number} {symbol} {word3} {word4}。",
                "{symbol} {word1} {number} {word2} {symbol} {word3}。",
                "{number} {word1} {word2} {symbol} {word3} {number}。",
            ],
            "words": ["xyz", "abc", "qwe", "mnp", "rst", "uvw", "ijk", "lmn", "opq", "def"],
            "numbers": ["123", "456", "789", "321", "654", "987", "147", "258", "369"],
            "symbols": ["@#$", "%^&", "*()", "!@#", "&*(",")(*", "!@$", "%#@"]
        }
    }
    
    return templates

# ==================== 主函数 ====================

def main():
    base_dir = "上下文机制探究/PAC-Test-Dataset"
    
    # 生成事实库
    facts = generate_facts_library()
    with open(f"{base_dir}/core_knowledge/facts_library.json", "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    print(f"Generated facts library: {len(facts)} facts")
    
    # 生成实体对
    entity_pairs = generate_entity_pairs()
    with open(f"{base_dir}/core_knowledge/entity_pairs.json", "w", encoding="utf-8") as f:
        json.dump(entity_pairs, f, ensure_ascii=False, indent=2)
    print(f"Generated entity pairs: {sum(len(v) for v in entity_pairs.values())} pairs")
    
    # 生成事实链
    fact_chains = generate_fact_chains()
    with open(f"{base_dir}/core_knowledge/fact_chains.json", "w", encoding="utf-8") as f:
        json.dump(fact_chains, f, ensure_ascii=False, indent=2)
    print(f"Generated fact chains: {sum(len(v) for v in fact_chains.values())} chains")
    
    # 生成填充模板
    filler_templates, variables = generate_filler_templates()
    with open(f"{base_dir}/templates/filler_templates.json", "w", encoding="utf-8") as f:
        json.dump({"templates": filler_templates, "variables": variables}, f, ensure_ascii=False, indent=2)
    print("Generated filler templates")
    
    # 生成干扰模板
    distractor_templates = generate_distractor_templates()
    with open(f"{base_dir}/templates/distractor_templates.json", "w", encoding="utf-8") as f:
        json.dump(distractor_templates, f, ensure_ascii=False, indent=2)
    print("Generated distractor templates")
    
    print("\nAll knowledge base files generated successfully!")

if __name__ == "__main__":
    main()
