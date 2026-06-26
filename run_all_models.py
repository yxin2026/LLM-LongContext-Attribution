#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依次对多个模型运行完整 PAC-Test 评测，实时打印进度。
每个模型完成后立即输出结果；支持断点续跑。
"""

import subprocess
import sys
import os
from pathlib import Path

SCRIPT = Path(__file__).parent / "evaluate.py"

MODELS = [
    "qwen2.5:7b",
    "deepseek-r1:7b",
    "qwen3.5:9b",
]

def run_model(model: str):
    print(f"\n{'#'*60}")
    print(f"  开始评测：{model}")
    print(f"{'#'*60}\n")
    cmd = [
        sys.executable, str(SCRIPT),
        "--model", model,
        "--subset", "all",
    ]
    # 实时输出，不捕获
    result = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        print(f"[WARN] {model} 评测异常退出 (code={result.returncode})，继续下一个模型")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for model in MODELS:
        run_model(model)
    print("\n所有模型评测完成！运行 python compare_models.py 查看对比结果。")
