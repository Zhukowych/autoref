"""Native Anthropic Messages API client. JSON is forced via a single required tool."""
from __future__ import annotations

import json
import os

from .base import LLMClient, LLMError


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, cfg):
        super().__init__(cfg)
        import anthropic
        key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("anthropic provider needs api_key or ANTHROPIC_API_KEY")
        kw = {"api_key": key, "timeout": cfg.timeout, "max_retries": 3}
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        self._client = anthropic.Anthropic(**kw)

    def _chat(self, system, user, *, json_schema=None):
        kwargs = dict(model=self.model, max_tokens=self.cfg.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if json_schema is not None:
            kwargs["tools"] = [{"name": "answer", "description": "Return the structured answer.",
                                "input_schema": json_schema}]
            kwargs["tool_choice"] = {"type": "tool", "name": "answer"}
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            raise LLMError(f"{self.model_id}: {e}") from e
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        if not text:
            raise LLMError(f"{self.model_id}: empty response")
        return text
