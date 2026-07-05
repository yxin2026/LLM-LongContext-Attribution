# NIAH 自动批量测试脚本

脚本：

```text
scripts/run_niah_batch.py
```

它用于按 `Framework_V2.0.docx` 的 NIAH 设计自动完成两件事：

1. 生成 NIAH 数据。
2. 按模型列表逐个调用 API/本地服务测试，并保存 JSONL 结果。

## 内置 NIAH 套件

`--suite smoke`

- Single-NIAH
- 4K
- 50% 位置
- 默认 2 条样本
- 用来确认 API、模型名、评分链路是否正常。

`--suite fast16k`

- Single-NIAH
- 16K
- 10% / 50% / 90%
- 默认 50 条/位置
- 对应 Framework V2.0 的快速 Lost in the Middle 筛查。

`--suite framework_v2`

严格按表格生成三类：

- Single-NIAH：4K / 16K / 32K / 64K，位置 10% / 50% / 90%
- Multi-NIAH：16K / 32K，均匀分布 / 聚集分布
- Sequential-NIAH：16K / 32K，链式分布

## 内置模型组

`--profile minimal`

- `qwen35_9b`
- `qwen3_8b`
- `deepseek_r1_distill_qwen_14b`

`--profile single_card`

包含 Framework V2.0 单卡优先顺序里的主线模型：

- `qwen35_9b`
- `qwen3_8b`
- `deepseek_r1_distill_qwen_14b`
- `qwen35_27b`
- `gemma4_31b`
- `qwen35_35b_a3b`
- `gemma4_26b_a4b`
- `qwen35_122b_a10b`

`--profile all_framework`

包含 Framework V2.0 表格里所有已写入脚本的模型，包括 Hunyuan、Seed-OSS、Qwen3-14B thinking/no-thinking 对照。

## 推荐执行顺序

先设置 SiliconFlow API key：

```powershell
$env:SILICONFLOW_API_KEY="sk-你的key"
```

先看脚本会执行什么，不真正调用 API：

```powershell
python scripts/run_niah_batch.py `
  --suite smoke `
  --profile minimal `
  --dry-run
```

再跑最小 smoke：

```powershell
python scripts/run_niah_batch.py `
  --suite smoke `
  --profile minimal
```

如果 smoke 正常，再跑 16K 主筛查：

```powershell
python scripts/run_niah_batch.py `
  --suite fast16k `
  --profile minimal
```

最后再考虑全量 Framework V2.0：

```powershell
python scripts/run_niah_batch.py `
  --suite framework_v2 `
  --profile single_card
```

## 指定模型

不用内置 profile，也可以直接指定：

```powershell
python scripts/run_niah_batch.py `
  --suite fast16k `
  --models qwen35_9b,qwen3_8b
```

也可以传完整 SiliconFlow 模型名：

```powershell
python scripts/run_niah_batch.py `
  --suite smoke `
  --models Qwen/Qwen3.5-9B
```

## 输出位置

数据默认写到：

```text
data/generated/niah_batch/{suite}/
```

结果默认写到：

```text
results/raw/niah_batch/{suite}/{run_id}/{model_alias}.jsonl
```

每个结果目录还会写：

```text
niah_batch_metadata.json
```

## 安全长度处理

脚本内置每个模型的 `max_model_len`。如果样本长度超过模型安全长度，`run_niah.py` 会写入：

```text
error=skipped_by_model_length
metric=skipped_by_model_length
```

不会真的调用 API。

## 注意

`framework_v2` 全量样本会比较多，尤其多个模型一起跑会消耗明显 API 额度。建议先跑：

```text
smoke -> fast16k -> framework_v2
```

