from __future__ import annotations

import argparse
import importlib.util
import sys


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Optional model name/path to test tokenizer loading.")
    args = parser.parse_args()

    print(f"python={sys.version.split()[0]}")
    required = ["torch", "transformers", "accelerate", "pandas", "numpy", "matplotlib", "tqdm"]
    missing = [name for name in required if not has_module(name)]
    if missing:
        print("missing=" + ",".join(missing))
        print("Install with: pip install -r requirements.txt")
        raise SystemExit(1)

    import torch
    import transformers

    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gb = props.total_memory / (1024**3)
            print(f"cuda:{idx}={props.name}, memory={gb:.1f}GB")

    if args.model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"tokenizer_loaded={args.model}, vocab_size={len(tok)}")

    print("environment_ok=true")


if __name__ == "__main__":
    main()

