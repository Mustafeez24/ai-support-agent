"""NVIDIA LLM provider via the OpenAI-compatible `integrate.api.nvidia.com`
endpoint.

No custom HTTP client: NVIDIA's endpoint speaks the OpenAI Chat Completions
API, so we use the official `openai` SDK pointed at NVIDIA's `base_url`.
"""
from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import LLMConfig
from src.llm.base import LLMMessage, LLMNotConfiguredError, LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)


class NvidiaLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: OpenAI | None = None

    def is_configured(self) -> bool:
        return self.config.is_configured()

    def _get_client(self) -> OpenAI:
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "NVIDIA_API_KEY is not set (or is still the placeholder "
                "'YOUR_NVIDIA_API_KEY_HERE'). Copy .env.example to .env and "
                "set a real key -- see README 'NVIDIA API setup'. Callers "
                "should treat this as a signal to escalate to a human "
                "rather than crash."
            )
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()  # raises LLMNotConfiguredError before any retry loop
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resolved_model = model or self.config.model
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        retryer = Retrying(
            stop=stop_after_attempt(max(1, self.config.max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        try:
            for attempt in retryer:
                with attempt:
                    response = client.chat.completions.create(
                        model=resolved_model,
                        messages=payload,
                        temperature=resolved_temperature,
                        max_tokens=resolved_max_tokens,
                    )
        except Exception as exc:
            logger.error("NVIDIA LLM call failed after %d attempt(s): %s", self.config.max_retries, exc)
            raise LLMProviderError(f"NVIDIA API call failed after retries: {exc}") from exc

        choice = response.choices[0]
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return LLMResponse(content=choice.message.content or "", model=response.model, raw=raw)
