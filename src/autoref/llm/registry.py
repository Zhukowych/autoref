"""provider name -> LLMClient class. Add new backends here."""
from __future__ import annotations

from .base import LLMClient
from .anthropic_client import AnthropicClient
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient

PROVIDERS: dict[str, type[LLMClient]] = {
    "openai_compat": OpenAICompatClient,
    "openai": OpenAICompatClient,
    "anthropic": AnthropicClient,
    "ollama": OllamaClient,
}

_cache: dict[tuple, LLMClient] = {}


def make_llm(cfg) -> LLMClient:
    key = (cfg.provider, cfg.model, cfg.base_url)
    if key not in _cache:
        if cfg.provider not in PROVIDERS:
            raise ValueError(f"unknown LLM provider {cfg.provider!r}; available: {', '.join(PROVIDERS)}")
        _cache[key] = PROVIDERS[cfg.provider](cfg)
    return _cache[key]


def llm_for(topic_cfg, task: str) -> LLMClient:
    return make_llm(topic_cfg.llm.for_task(task))
