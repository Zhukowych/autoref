"""Native Ollama /api/chat client (supports `format` = JSON schema for structured output)."""
from __future__ import annotations

import httpx

from .base import LLMClient, LLMError


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, cfg):
        super().__init__(cfg)
        base = (cfg.base_url or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._url = f"{base}/api/chat"
        self._http = httpx.Client(timeout=cfg.timeout)

    def _chat(self, system, user, *, json_schema=None):
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
        }
        if json_schema is not None:
            body["format"] = json_schema
        try:
            r = self._http.post(self._url, json=body)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"{self.model_id}: {e}") from e
        content = (r.json().get("message") or {}).get("content")
        if not content:
            raise LLMError(f"{self.model_id}: empty response")
        return content
