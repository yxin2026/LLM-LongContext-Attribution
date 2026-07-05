from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.inference.client import create_inference_client


def main() -> None:
    args = parse_args()
    client = create_inference_client(
        provider="siliconflow",
        model_name=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )
    result = client.generate(
        prompt=args.prompt,
        request_id="siliconflow_healthcheck",
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    print(f"provider=siliconflow")
    print(f"endpoint={client.endpoint}")
    print(f"api_model={client.served_model_name}")
    print(f"error={result.error}")
    print(f"latency_sec={result.latency_sec:.3f}")
    print(f"prompt_tokens={result.prompt_tokens}")
    print(f"completion_tokens={result.completion_tokens}")
    print("response:")
    print(result.response_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SiliconFlow chat-completions access.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default="Return exactly: OK")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

