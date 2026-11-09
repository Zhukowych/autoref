import json
from pathlib import Path

import pytest
import yaml

from autoref.config import load_config
from autoref.llm.base import LLMClient
from autoref.llm.registry import PROVIDERS, _cache
from autoref.pipeline.context import Ctx

TEMPLATE = Path(__file__).parents[1] / "src/autoref/templates/mde.yaml"


class FakeLLM(LLMClient):
    """Deterministic backend: answers based on which schema is requested."""
    provider = "fake"
    calls: list = []

    def _chat(self, system, user, *, json_schema=None):
        FakeLLM.calls.append((system, user))
        if json_schema is None:
            return "pong"
        props = json_schema.get("properties", {})
        if "categories" in props:
            return json.dumps({"categories": [{"id": "mde_self_supervised", "confidence": 0.9},
                                              {"id": "nonexistent", "confidence": 0.9}][:1],
                               "rationale": "photometric loss"})
        if "relation_detail" in props:
            return 'Sure! ```json\n{"label": "uses_component", "relation_detail": "Uses the ResNet encoder of CITED",' \
                   ' "difference_summary": "adds UAV data", "evidence": ["we use [3] encoder"], "confidence": 0.8}\n```'
        if "relevance" in props:
            return '{"relevance": 0.9, "reason": "on topic"}'
        if "answer" in props:
            return '{"answer": "pong", "number": 42}'
        return "{}"


@pytest.fixture(autouse=True)
def fake_provider():
    PROVIDERS["fake"] = FakeLLM
    _cache.clear()
    FakeLLM.calls = []
    yield
    PROVIDERS.pop("fake", None)
    _cache.clear()


@pytest.fixture
def cfg_path(tmp_path):
    raw = yaml.safe_load(TEMPLATE.read_text())
    raw["db"] = str(tmp_path / "t.sqlite")
    raw["vault"] = str(tmp_path / "vault")
    raw["llm"] = {"default": {"provider": "fake", "model": "m"}}
    raw["sources"] = {"openalex": {"max_rate": 1000}}
    p = tmp_path / "configs" / "t.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.fixture
def ctx(cfg_path):
    from rich.console import Console
    c = Ctx(load_config(cfg_path), console=Console(quiet=True))
    yield c
    c.close()
