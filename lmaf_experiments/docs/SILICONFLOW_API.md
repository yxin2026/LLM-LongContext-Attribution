# SiliconFlow API 使用手册

本项目现在支持两种运行方式：

- 本地 vLLM：`--provider local`
- SiliconFlow 硅基流动 API：`--provider siliconflow`

SiliconFlow 的 Chat Completions 接口是 OpenAI-compatible 风格，默认 base URL 使用：

```text
https://api.siliconflow.cn/v1
```

项目会调用 `/chat/completions`，并把 `Authorization: Bearer <API_KEY>` 交给 OpenAI Python SDK 处理。

## 1. 准备 API Key

在 PowerShell 里设置：

```powershell
$env:SILICONFLOW_API_KEY="sk-你的硅基流动APIKey"
```

也可以复制 `.env.example` 为 `.env` 自己管理，但当前脚本不会自动读取 `.env`；最稳的是直接 export 环境变量，或者每次命令加 `--api-key`。

## 2. 先跑健康检查

```powershell
python scripts/siliconflow_healthcheck.py --model Qwen/Qwen3.5-9B
```

如果你想用项目别名，也可以：

```powershell
python scripts/siliconflow_healthcheck.py --model qwen35_9b
```

当前内置别名：

| 项目别名 | SiliconFlow model |
|---|---|
| `qwen35_9b` | `Qwen/Qwen3.5-9B` |
| `qwen35_27b` | `Qwen/Qwen3.5-27B` |
| `qwen35_35b_a3b` | `Qwen/Qwen3.5-35B-A3B` |
| `qwen35_122b_a10b` | `Qwen/Qwen3.5-122B-A10B` |
| `qwen3_8b_baseline` | `Qwen/Qwen3-8B` |

如果平台模型广场里的模型名不在上表，直接把完整模型名传给 `--model`。

## 3. 生成 NIAH smoke 数据

```powershell
python scripts/run_niah.py `
  --generate-only `
  --lengths 4096 `
  --positions 50 `
  --samples-per-cell 2 `
  --output data/generated/smoke_niah
```

## 4. 用 SiliconFlow 跑 NIAH smoke

```powershell
python scripts/run_niah.py `
  --provider siliconflow `
  --model Qwen/Qwen3.5-9B `
  --input data/generated/smoke_niah `
  --output results/raw/smoke/siliconflow_qwen35_9b.jsonl `
  --resume
```

也可以用别名：

```powershell
python scripts/run_niah.py `
  --provider siliconflow `
  --model qwen35_9b `
  --input data/generated/smoke_niah `
  --output results/raw/smoke/siliconflow_qwen35_9b.jsonl `
  --resume
```

结果行里会同时记录：

- `model`：你命令里传入的名字
- `provider`：`siliconflow`
- `api_model`：实际发给 SiliconFlow 的模型名

## 5. 跑 PAC-Test A

先生成数据：

```powershell
python scripts/run_pac.py `
  --generate-only `
  --subset A_position `
  --length 16384 `
  --positions 10,25,50,75,90 `
  --samples-per-cell 50 `
  --output data/generated/pac/A_position
```

再调用 SiliconFlow：

```powershell
python scripts/run_pac.py `
  --provider siliconflow `
  --model Qwen/Qwen3.5-9B `
  --subset A_position `
  --input data/generated/pac/A_position `
  --output results/raw/pac/A_position/siliconflow_qwen35_9b.jsonl `
  --resume
```

## 5.5 一键批量跑 NIAH 模型池

如果你要按 `Framework_V2.0.docx` 的模型池自动跑 NIAH，用：

```powershell
python scripts/run_niah_batch.py `
  --suite smoke `
  --profile minimal `
  --dry-run
```

确认命令没问题后，先跑最小 smoke：

```powershell
python scripts/run_niah_batch.py `
  --suite smoke `
  --profile minimal
```

完整说明见：

```text
docs/NIAH_BATCH.md
```

## 6. LongBench 和 RULER 同样使用 provider 参数

LongBench：

```powershell
python scripts/run_longbench.py `
  --provider siliconflow `
  --model Qwen/Qwen3.5-9B `
  --tasks narrativeqa,qasper,hotpotqa,gov_report `
  --input data/processed/longbench `
  --output results/raw/longbench/siliconflow_qwen35_9b.jsonl `
  --resume
```

RULER fallback：

```powershell
python scripts/run_ruler.py `
  --provider siliconflow `
  --model Qwen/Qwen3.5-9B `
  --input data/generated/ruler `
  --output results/raw/ruler/siliconflow_qwen35_9b.jsonl `
  --resume
```

## 7. enable_thinking 开关

为保证评分稳定，本项目默认对 SiliconFlow 请求加：

```json
{"enable_thinking": false}
```

如果你确实要打开模型思考，可以加：

```powershell
--enable-thinking --thinking-budget 4096
```

注意：打开 thinking 可能增加延迟和 token 成本，也可能让模型更容易输出解释性文本。长上下文评测建议先保持关闭。

## 8. 汇总和画图

```powershell
python scripts/aggregate_results.py `
  --input results/raw/niah `
  --experiment niah `
  --output results/aggregate/niah_results.csv

python scripts/plot_results.py `
  --input results/aggregate/niah_results.csv `
  --plot niah_position_curve `
  --output results/figures/niah_position_curve.png
```

## 9. 常见问题

`SILICONFLOW_API_KEY is required`

说明没有设置环境变量，也没有传 `--api-key`。

`model not found`

去 SiliconFlow 模型广场确认模型名，然后把完整模型名传给 `--model`。

请求超时

长上下文可能很慢。可以提高：

```powershell
--timeout 1800
```

输出很多解释

保持默认 `enable_thinking=false`，并继续使用固定推理参数 `temperature=0.0`、`top_p=1.0`、`max_tokens=512`。
