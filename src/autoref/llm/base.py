"""LLMClient interface. Implementations only provide `_chat`; JSON handling is shared."""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (handles ```json fences, <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in reply: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise LLMError(f"unterminated JSON in reply: {text[:200]!r}")


class LLMClient(ABC):
    """Uniform interface over local/remote LLMs.

    complete(system, user)                -> str
    complete_json(system, user, Schema)   -> Schema instance (validated, retried on bad output)
    """

    provider: str = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.model

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def _chat(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """Return raw text. If json_schema is given, the backend should try to force JSON output."""

    def complete(self, system: str, user: str) -> str:
        return self._chat(system, user)

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        js = schema.model_json_schema()
        sys_full = (f"{system}\n\nRespond ONLY with a single JSON object matching this JSON schema "
                    f"(no prose, no markdown):\n{json.dumps(js)}")
        prompt = user
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            raw = self._chat(sys_full, prompt, json_schema=js if self.cfg.json_mode else None)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, LLMError, ValueError) as e:
                last_err = e
                log.debug("invalid JSON from %s (attempt %d): %s", self.model_id, attempt + 1, e)
                prompt = (f"{user}\n\nYour previous reply was invalid:\n{raw[:2000]}\n\nError: {e}\n"
                          "Return a corrected JSON object only.")
        raise LLMError(f"{self.model_id} failed to return valid JSON: {last_err}")


class FailureGuard:
    """Abort a batch step after `limit` consecutive LLM failures (backend down / misconfigured)."""

    def __init__(self, step: str, limit: int = 3):
        self.step, self.limit, self.n = step, limit, 0

    def ok(self):
        self.n = 0

    def failed(self, err: Exception):
        self.n += 1
        if self.n >= self.limit:
            raise LLMError(f"{self.step}: {self.n} consecutive LLM failures, aborting "
                           f"(last: {err}). Check `autoref llm test`.") from err
