from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError("Please install ModelScope first: pip install -U modelscope") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.model, local_dir=str(local_dir))
    print(f"downloaded_to={path}")
    print("Use this local path in RUN_DAY1.ps1 with -Model.")


if __name__ == "__main__":
    main()

