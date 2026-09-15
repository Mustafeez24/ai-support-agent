"""Tests for the NVIDIA LLM provider.

No network calls to the real NVIDIA API -- the `openai.OpenAI` client is
monkeypatched with a fake that mimics its `chat.completions.create`
interface. Per the assignment's own instruction, tests must not require
the real NVIDIA API unless explicitly marked as an integration test; none
here are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.config import LLMConfig
from src.llm import nvidia as nvidia_mod
from src.llm.base import LLMMessage, LLMNotConfiguredError, LLMProviderError


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list
    model: str = "fake-model"

    def model_dump(self):
        return {"model": self.model, "choices": "fake"}


class _FakeCompletions:
    def __init__(self, fail_times: int = 0, response_text: str = "Sure, here's help."):
        self.fail_times = fail_times
        self.calls = 0
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient API failure")
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=self.response_text))])


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, fail_times: int = 0, **kwargs):
        self.chat = _FakeChat(_FakeCompletions(fail_times=fail_times))


def _configured_llm_config(**overrides) -> LLMConfig:
    defaults = dict(
        api_key="real-key-123",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        timeout_seconds=5.0,
        max_retries=3,
        temperature=0.2,
        max_tokens=200,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def test_is_configured_false_for_placeholder_key():
    config = _configured_llm_config(api_key="YOUR_NVIDIA_API_KEY_HERE")
    provider = nvidia_mod.NvidiaLLMProvider(config)
    assert provider.is_configured() is False


def test_is_configured_false_for_missing_key():
    config = _configured_llm_config(api_key=None)
    provider = nvidia_mod.NvidiaLLMProvider(config)
    assert provider.is_configured() is False


def test_generate_raises_not_configured_without_calling_network(monkeypatch):
    config = _configured_llm_config(api_key="")
    provider = nvidia_mod.NvidiaLLMProvider(config)

    def _boom(*args, **kwargs):
        raise AssertionError("OpenAI() should never be constructed when not configured")

    monkeypatch.setattr(nvidia_mod, "OpenAI", _boom)
    with pytest.raises(LLMNotConfiguredError):
        provider.generate([LLMMessage(role="user", content="hi")])


def test_generate_succeeds_on_first_try(monkeypatch):
    config = _configured_llm_config()
    monkeypatch.setattr(nvidia_mod, "OpenAI", lambda **kw: _FakeOpenAIClient(fail_times=0))
    provider = nvidia_mod.NvidiaLLMProvider(config)

    response = provider.generate([LLMMessage(role="user", content="hi")])
    assert response.content == "Sure, here's help."
    assert response.model == "fake-model"


def test_generate_retries_then_succeeds(monkeypatch):
    config = _configured_llm_config(max_retries=3)
    fake_client = _FakeOpenAIClient(fail_times=2)
    monkeypatch.setattr(nvidia_mod, "OpenAI", lambda **kw: fake_client)
    provider = nvidia_mod.NvidiaLLMProvider(config)

    response = provider.generate([LLMMessage(role="user", content="hi")])
    assert response.content == "Sure, here's help."
    assert fake_client.chat.completions.calls == 3


def test_generate_raises_provider_error_after_exhausting_retries(monkeypatch):
    config = _configured_llm_config(max_retries=2)
    fake_client = _FakeOpenAIClient(fail_times=10)
    monkeypatch.setattr(nvidia_mod, "OpenAI", lambda **kw: fake_client)
    provider = nvidia_mod.NvidiaLLMProvider(config)

    with pytest.raises(LLMProviderError):
        provider.generate([LLMMessage(role="user", content="hi")])
    assert fake_client.chat.completions.calls == 2


def test_generate_passes_resolved_defaults_to_client(monkeypatch):
    config = _configured_llm_config(model="default-model", temperature=0.3, max_tokens=123)
    captured = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="ok"))])

    class _CapturingClient:
        def __init__(self, **kw):
            self.chat = _FakeChat(_CapturingCompletions())

    monkeypatch.setattr(nvidia_mod, "OpenAI", lambda **kw: _CapturingClient())
    provider = nvidia_mod.NvidiaLLMProvider(config)
    provider.generate([LLMMessage(role="user", content="hi")])

    assert captured["model"] == "default-model"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 123

    provider.generate([LLMMessage(role="user", content="hi")], model="override-model", temperature=0.9)
    assert captured["model"] == "override-model"
    assert captured["temperature"] == 0.9
