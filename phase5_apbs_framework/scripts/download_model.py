from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


class Redirect308Handler(HTTPRedirectHandler):
    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, code, msg, headers)


OPENER = build_opener(Redirect308Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--endpoint", default=None, help="Optional HF endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--token", default=None)
    parser.add_argument("--direct", action="store_true", help="Bypass snapshot_download and stream files directly.")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["*.gguf", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite"],
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = args.include or [
        "*.json",
        "*.jinja",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.safetensors",
        "*.py",
    ]

    if args.direct:
        direct_download(args.model, local_dir, args.endpoint or "https://huggingface.co", args.token, allow_patterns, args.exclude)
        print(f"downloaded_to={local_dir}")
        print("Use this local path in RUN_DAY1.ps1 with -Model.")
        return

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=args.token,
        allow_patterns=allow_patterns,
        ignore_patterns=args.exclude,
    )
    print(f"downloaded_to={path}")
    print("Use this local path in RUN_DAY1.ps1 with -Model.")


def auth_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "phase5-apbs-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def should_keep(path: str, include: list[str], exclude: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in include) and not any(
        fnmatch.fnmatch(path, pattern) for pattern in exclude
    )


def read_json(url: str, token: str | None) -> dict:
    request = Request(url, headers=auth_headers(token))
    with OPENER.open(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_download(url: str, dest: Path, token: str | None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    resume_at = tmp.stat().st_size if tmp.exists() else 0
    headers = auth_headers(token)
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    request = Request(url, headers=headers)
    try:
        response = OPENER.open(request, timeout=120)
    except HTTPError as exc:
        if exc.code == 416 and tmp.exists():
            tmp.replace(dest)
            return
        raise

    mode = "ab" if resume_at and response.status == 206 else "wb"
    if resume_at and response.status != 206:
        resume_at = 0
    total = response.headers.get("Content-Length")
    total_bytes = int(total) + resume_at if total and total.isdigit() else None
    downloaded = resume_at
    print(f"Downloading {dest.name} from {downloaded}/{total_bytes or '?'} bytes")
    with response, tmp.open(mode + "") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                print(f"\r  {downloaded / (1024**3):.2f}GB / {total_bytes / (1024**3):.2f}GB ({pct:.1f}%)", end="")
            else:
                print(f"\r  {downloaded / (1024**3):.2f}GB", end="")
    print()
    tmp.replace(dest)


def direct_download(
    model: str,
    local_dir: Path,
    endpoint: str,
    token: str | None,
    include: list[str],
    exclude: list[str],
) -> None:
    endpoint = endpoint.rstrip("/")
    api_model = quote(model, safe="/")
    info = read_json(f"{endpoint}/api/models/{api_model}", token)
    siblings = info.get("siblings", [])
    files = sorted(
        row["rfilename"] for row in siblings if row.get("rfilename") and should_keep(row["rfilename"], include, exclude)
    )
    if not files:
        raise RuntimeError("No files matched the include/exclude patterns.")

    print(f"Matched {len(files)} files")
    for filename in files:
        dest = local_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Exists: {filename}")
            continue
        quoted = quote(filename, safe="/")
        url = f"{endpoint}/{api_model}/resolve/main/{quoted}"
        try:
            stream_download(url, dest, token)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Failed downloading {filename} from {url}: {exc}") from exc


if __name__ == "__main__":
    main()
