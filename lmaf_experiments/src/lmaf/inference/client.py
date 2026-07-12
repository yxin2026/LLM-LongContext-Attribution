from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1"
LOCAL_ENDPOINT = "http://localhost:8000/v1"

SILICONFLOW_MODEL_ALIASES = {
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_27b": "Qwen/Qwen3.5-27B",
    "qwen35_35b_a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen35_122b_a10b": "Qwen/Qwen3.5-122B-A10B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_baseline": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_14b_no_thinking": "Qwen/Qwen3-14B",
    "qwen3_14b_thinking": "Qwen/Qwen3-14B",
    "hunyuan_a13b": "tencent/Hunyuan-A13B-Instruct",
    "seed_oss_36b": "ByteDance-Seed/Seed-OSS-36B-Instruct",
}


@dataclass
class InferenceResult:
    response_text: str
    latency_sec: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


def resolve_provider_model(provider: str, model_name: str) -> str:
    if provider == "siliconflow":
        return SILICONFLOW_MODEL_ALIASES.get(model_name, model_name)
    return model_name


def create_inference_client(
    provider: str,
    model_name: str,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float = 600,
    retry: int = 3,
    backoff: tuple[float, ...] = (2, 4, 8),
    system_prompt: str = "You are a precise evaluator. Answer with only the final answer.",
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> "OpenAICompatibleClient":
    provider = provider.lower().strip()
    resolved_model = resolve_provider_model(provider, model_name)
    extra_body: dict[str, Any] = {}

    if provider == "siliconflow":
        endpoint = endpoint or os.getenv("SILICONFLOW_BASE_URL") or SILICONFLOW_ENDPOINT
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required when --provider siliconflow is used.")
        extra_body["enable_thinking"] = bool(enable_thinking)
        if thinking_budget is not None:
            extra_body["thinking_budget"] = thinking_budget
    elif provider == "local":
        endpoint = endpoint or os.getenv("LOCAL_OPENAI_BASE_URL") or LOCAL_ENDPOINT
        api_key = api_key or os.getenv("LOCAL_OPENAI_API_KEY") or "local-token"
    elif provider == "custom":
        if not endpoint:
            raise ValueError("--endpoint is required when --provider custom is used.")
        api_key = api_key or os.getenv("OPENAI_API_KEY") or "local-token"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return OpenAICompatibleClient(
        endpoint=endpoint,
        api_key=api_key,
        served_model_name=resolved_model,
        timeout=timeout,
        retry=retry,
        backoff=backoff,
        system_prompt=system_prompt,
        provider=provider,
        extra_body=extra_body,
    )


class OpenAICompatibleClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        served_model_name: str,
        timeout: float = 600,
        retry: int = 3,
        backoff: tuple[float, ...] = (2, 4, 8),
        system_prompt: str = "You are a precise evaluator. Answer with only the final answer.",
        provider: str = "custom",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.served_model_name = served_model_name
        self.timeout = timeout
        self.retry = retry
        self.backoff = backoff
        self.system_prompt = system_prompt
        self.provider = provider
        self.extra_body = extra_body or {}

        from openai import OpenAI

        self._client = OpenAI(base_url=self.endpoint, api_key=self.api_key, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> InferenceResult:
        last_error: str | None = None
        for attempt in range(self.retry):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.served_model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=self.extra_body or None,
                    extra_headers={"X-Request-ID": request_id},
                )
                latency = time.perf_counter() - started
                choice = response.choices[0]
                usage: Any = getattr(response, "usage", None)
                return InferenceResult(
                    response_text=choice.message.content or "",
                    latency_sec=latency,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    error=None,
                )
            except Exception as exc:
                last_error = repr(exc)
                if attempt < self.retry - 1:
                    time.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
        return InferenceResult(
            response_text="",
            latency_sec=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            error=last_error or "unknown_error",
        )
