"""OpenAI-compatible chat completions: OpenAI, Ollama (/v1), LM Studio, vLLM, OpenRouter, llama.cpp."""
from __future__ import annotations

import os

from .base import LLMClient, LLMError


class OpenAICompatClient(LLMClient):
    provider = "openai_compat"

    def __init__(self, cfg):
        super().__init__(cfg)
        from openai import OpenAI
        self._client = OpenAI(
            base_url=cfg.base_url or None,
            api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            timeout=cfg.timeout,
            max_retries=3,
        )
        self._json_supported = True

    def _chat(self, system, user, *, json_schema=None):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if json_schema is not None and self._json_supported:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # some servers reject response_format -> retry without it once
            if "response_format" in kwargs and "response_format" in str(e).lower():
                self._json_supported = False
                kwargs.pop("response_format")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise LLMError(f"{self.model_id}: {e}") from e
        content = resp.choices[0].message.content
        if not content:
            raise LLMError(f"{self.model_id}: empty response")
        return content
