from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(ROOT / "PAC" / "run_pac_v21_full_no_hunyuan_queue.py"),
        "--api-key",
        args.api_key,
        "--slots-per-key",
        "1",
        "--global-request-delay-sec",
        str(args.global_request_delay_sec),
        "--per-key-delay-sec",
        "0",
        "--timeout",
        str(args.timeout),
        "--retry",
        "1",
        "--queue-max-attempts",
        str(args.queue_max_attempts),
        "--connection-max-attempts",
        str(args.connection_max_attempts),
        "--connection-cooldown-sec",
        str(args.connection_cooldown_sec),
        "--connection-burst-threshold",
        str(args.connection_burst_threshold),
        "--connection-burst-window-sec",
        str(args.connection_burst_window_sec),
        "--global-connection-cooldown-sec",
        str(args.global_connection_cooldown_sec),
        "--transient-cooldown-sec",
        str(args.transient_cooldown_sec),
        "--rate-limit-cooldown-sec",
        str(args.rate_limit_cooldown_sec),
        "--model-rate-limit-cooldown-sec",
        str(args.model_rate_limit_cooldown_sec),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.subsets:
        command.extend(["--subsets", args.subsets])
    if args.models:
        command.extend(["--models", args.models])
    if args.stop_after is not None:
        command.extend(["--stop-after", str(args.stop_after)])
    if args.dry_run:
        command.append("--dry-run")
    if args.summarize_only:
        command.append("--summarize-only")

    print("+ " + " ".join(quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PAC v2.1 top-up with exactly one SiliconFlow API key and one active request."
    )
    parser.add_argument("--api-key", required=True, help="The single SiliconFlow API key to use.")
    parser.add_argument("--run-id", default="pac_v21_full_queue")
    parser.add_argument("--subsets", default="A,B,C,D21")
    parser.add_argument("--models", default="all_no_hunyuan")
    parser.add_argument("--global-request-delay-sec", type=float, default=45.0)
    parser.add_argument("--timeout", type=float, default=720.0)
    parser.add_argument("--queue-max-attempts", type=int, default=4)
    parser.add_argument("--connection-max-attempts", type=int, default=1)
    parser.add_argument("--connection-cooldown-sec", type=float, default=300.0)
    parser.add_argument("--connection-burst-threshold", type=int, default=2)
    parser.add_argument("--connection-burst-window-sec", type=float, default=120.0)
    parser.add_argument("--global-connection-cooldown-sec", type=float, default=300.0)
    parser.add_argument("--transient-cooldown-sec", type=float, default=60.0)
    parser.add_argument("--rate-limit-cooldown-sec", type=float, default=180.0)
    parser.add_argument("--model-rate-limit-cooldown-sec", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    main()
