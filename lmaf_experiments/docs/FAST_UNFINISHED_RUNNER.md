# 高速补跑未完成实验

`scripts/run_unfinished_fast.py` 用于补跑已有实验中尚未成功的样本。它会读取当前 raw 结果文件，只对缺失或 API 报错的 `sample_id` 再调用 SiliconFlow；只要某个样本历史上已经成功过，就不会重复调用。

## 先做这两件事

1. 停掉旧的顺序脚本，例如 `run_niah_batch.py`、`run_longbench_ruler_batch.py`、`run_pac_batch.py`。不要让两个脚本同时写同一批结果文件。
2. 设置 SiliconFlow key。

CMD:

```bat
set "SILICONFLOW_API_KEY=你的key"
```

PowerShell:

```powershell
$env:SILICONFLOW_API_KEY="你的key"
```

多个 key 可以用逗号分隔，脚本会随机分摊请求：

```bat
set "SILICONFLOW_API_KEYS=key1,key2,key3"
```

## 推荐命令

先预览还剩多少，不会请求 API：

```bat
python scripts\run_unfinished_fast.py --dry-run
```

正式高速补跑全部实验：

```bat
python scripts\run_unfinished_fast.py --max-workers 8
```

如果 API 额度和限速比较宽，可以提高并发：

```bat
python scripts\run_unfinished_fast.py --max-workers 12
```

如果开始大量报 rate limit，就降到 4 或 6：

```bat
python scripts\run_unfinished_fast.py --max-workers 6
```

## 分阶段跑

先补公开基准：

```bat
python scripts\run_unfinished_fast.py --experiments niah,longbench,ruler --max-workers 8
```

再补 PAC：

```bat
python scripts\run_unfinished_fast.py --experiments pac --pac-subsets A,B,C,D --max-workers 8
```

只跑某几个模型：

```bat
python scripts\run_unfinished_fast.py --models qwen35_9b,qwen3_8b --max-workers 8
```

## 结果位置

脚本直接追加到现有 raw 结果目录：

- NIAH: `results/raw/niah_batch/framework_v2_without_fast16k/framework_v2_extra`
- LongBench: `results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/longbench`
- RULER: `results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/ruler`
- PAC: `results/raw/pac_batch/pac_main`

补跑完成后，再运行已有整理脚本生成表格和图片。

