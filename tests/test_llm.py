import pytest
from pydantic import BaseModel

from autoref.config import LLMConfig
from autoref.llm import LLMError, extract_json
from autoref.llm.base import LLMClient


class Out(BaseModel):
    x: int


class Flaky(LLMClient):
    provider = "flaky"

    def __init__(self, cfg, replies):
        super().__init__(cfg)
        self.replies = list(replies)
        self.prompts = []

    def _chat(self, system, user, *, json_schema=None):
        self.prompts.append(user)
        return self.replies.pop(0)


def test_extract_json_variants():
    assert extract_json('<think>{"no": 1}</think> ok {"x": 1}') == '{"x": 1}'
    assert extract_json('```json\n{"x": 2}\n```') == '{"x": 2}'
    assert extract_json('pre {"a": "}{", "b": {"c": 1}} post') == '{"a": "}{", "b": {"c": 1}}'


def test_retry_on_invalid_json():
    c = Flaky(LLMConfig(model="m", max_retries=1), ['{"x": "nope"}', '{"x": 5}'])
    assert c.complete_json("s", "u", Out).x == 5
    assert "previous reply was invalid" in c.prompts[1]


def test_gives_up():
    c = Flaky(LLMConfig(model="m", max_retries=1), ["garbage", "garbage"])
    with pytest.raises(LLMError):
        c.complete_json("s", "u", Out)
