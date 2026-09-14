"""Provider-agnostic LLM interface.

Everything else in this codebase (agent, judge) depends only on this
module, never on `openai` or any NVIDIA-specific detail directly, so the
provider can be swapped (a different endpoint, a different vendor) by
implementing this interface once.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    raw: dict | None = None


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted without valid credentials.

    Callers (the agent's reply-generation step in particular) should catch
    this and fail gracefully -- e.g. escalate to a human with a clear
    reason -- rather than crash.
    """


class LLMProviderError(RuntimeError):
    """Raised when the provider's API call fails after retries."""


class LLMProvider(ABC):
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether real credentials are present (not just a placeholder)."""

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Raises LLMNotConfiguredError if credentials are missing, or
        LLMProviderError if the call fails after retries."""
