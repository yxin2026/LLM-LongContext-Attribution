from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenCounter:
    """Tokenizer-backed token counter with a deterministic fallback.

    Production runs should pass the model tokenizer path or name. The fallback
    is intentionally simple so generators and tests remain usable before model
    weights are available.
    """

    tokenizer_name: str | None = None
    model_path: str | None = None

    def __post_init__(self) -> None:
        self.kind = "heuristic"
        self.tokenizer: Any | None = None

        name = self.model_path or self.tokenizer_name
        if name:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                self.kind = "transformers"
                return
            except Exception:
                self.tokenizer = None

        try:
            import tiktoken

            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.kind = "tiktoken"
        except Exception:
            self.tokenizer = None

    def encode(self, text: str) -> list[Any]:
        if self.kind == "transformers":
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        if self.kind == "tiktoken":
            return list(self.tokenizer.encode(text))
        return _heuristic_tokens(text)

    def decode(self, tokens: list[Any]) -> str:
        if self.kind == "transformers":
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        if self.kind == "tiktoken":
            return self.tokenizer.decode(tokens)
        return " ".join(str(t) for t in tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def trim_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])


def _heuristic_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def make_filler(
    target_tokens: int,
    seed: int = 42,
    forbidden: tuple[str, ...] = (),
    counter: TokenCounter | None = None,
) -> str:
    """Build neutral filler near, but not above, a target token count."""

    if target_tokens <= 0:
        return ""
    counter = counter or TokenCounter()
    words = [
        "archive",
        "lantern",
        "harbor",
        "meadow",
        "copper",
        "signal",
        "museum",
        "orbit",
        "notebook",
        "valley",
        "winter",
        "thread",
        "garden",
        "station",
        "silver",
        "canvas",
        "quiet",
        "bridge",
        "forest",
        "engine",
        "paper",
        "memory",
        "river",
        "window",
    ]
    forbidden_norm = {item.lower() for item in forbidden if item}
    words = [w for w in words if w.lower() not in forbidden_norm]

    def text_for(n_words: int) -> str:
        return " ".join(words[(seed + i) % len(words)] for i in range(n_words))

    lo = 0
    hi = max(1, target_tokens * 2)
    while counter.count(text_for(hi)) < target_tokens and hi < max(100, target_tokens * 20):
        hi *= 2

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text_for(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text_for(lo)

