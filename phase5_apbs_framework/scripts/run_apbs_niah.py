from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apbs_rope_patch import APBSConfig, apply_rope_patch


def normalize(text: str) -> str:
    return text.strip().split()[0].strip(".,;:'\"`") if text.strip() else ""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id:
                ids.add(sample_id)
    return ids


def input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        return embeddings.weight.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_quantization_config(args):
    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Quantized loading requires BitsAndBytesConfig. Install bitsandbytes and a compatible transformers build."
        ) from exc

    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype in {"auto", "bf16"} else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["baseline", "ntk", "apbs"], required=True)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--target-length", type=int, default=16384)
    parser.add_argument("--train-context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-memory", default=None, help='Optional max memory, e.g. "0:18GiB,cpu:48GiB".')
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Use either --load-in-4bit or --load-in-8bit, not both.")

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": dtype_map[args.dtype],
        "device_map": args.device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            key, value = item.split(":", 1)
            max_memory[int(key) if key.isdigit() else key] = value
        model_kwargs["max_memory"] = max_memory
    if args.offload_folder:
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = args.offload_folder

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()
    model_input_device = input_device(model)

    cfg = APBSConfig(
        method=args.method,
        gamma=args.gamma,
        target_length=args.target_length,
        train_context_length=args.train_context_length,
        base=float(getattr(model.config, "rope_theta", 10000.0)),
    )
    patched = apply_rope_patch(model, cfg)
    print(f"method={args.method} gamma={args.gamma} patched_rotary_modules={patched}")

    rows = load_jsonl(Path(args.dataset))
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = existing_sample_ids(output) if args.append else set()
    if done_ids:
        before = len(rows)
        rows = [row for row in rows if row["sample_id"] not in done_ids]
        print(f"append=true skipped_existing={before - len(rows)} remaining={len(rows)}")

    mode = "a" if args.append else "w"
    model_key = args.model_key or Path(args.model).name.replace("/", "_").replace("\\", "_")
    with output.open(mode, encoding="utf-8") as f:
        for row in tqdm(rows, desc=args.method):
            inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=False)
            inputs = {k: v.to(model_input_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_norm = normalize(prediction)
            answer_norm = normalize(row["answer"])
            out = {
                **{k: row[k] for k in ["sample_id", "length", "position", "answer"]},
                "model_key": model_key,
                "model_path": args.model,
                "method": args.method,
                "gamma": args.gamma if args.method == "apbs" else None,
                "prediction": prediction,
                "correct": int(answer_norm.lower() in prediction.lower() or pred_norm.lower() == answer_norm.lower()),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
