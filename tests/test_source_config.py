"""`sources:` mapping with per-source settings and max_rate."""
import logging

import pytest
import yaml
from pydantic import ValidationError

from autoref.config import TopicConfig, load_config
from autoref.sources import SourceSet, make_source
from autoref.sources.arxiv import ArxivSource
from autoref.sources.openalex import OpenAlexSource
from autoref.sources.semantic_scholar import SemanticScholarSource


@pytest.fixture
def raw(cfg_path):
    return yaml.safe_load(cfg_path.read_text())


def cfg_with(raw, sources, **extra):
    return TopicConfig.model_validate({**raw, "sources": sources, **extra})


def test_mapping_order_aliases_and_empty_bodies(raw):
    c = cfg_with(raw, {"arxiv": None, "s2": {"api_key": "k", "max_rate": 0.5}, "oa": {}})
    assert list(c.sources) == ["arxiv", "semantic_scholar", "openalex"]  # order = priority
    assert c.sources["semantic_scholar"].api_key == "k" and c.sources["semantic_scholar"].max_rate == 0.5
    assert c.enabled_sources == ["arxiv", "semantic_scholar", "openalex"]


def test_disabled_source_is_skipped(raw):
    c = cfg_with(raw, {"openalex": {}, "arxiv": {"enabled": False}})
    assert c.enabled_sources == ["openalex"]
    assert [s.name for s in SourceSet(c.sources)] == ["openalex"]


def test_validation_errors(raw):
    with pytest.raises(ValidationError, match="unknown source 'scopus'"):
        cfg_with(raw, {"scopus": {}})
    with pytest.raises(ValidationError):
        cfg_with(raw, {"arxiv": {"max_rat": 1}})            # typo -> rejected, not ignored
    with pytest.raises(ValidationError):
        cfg_with(raw, {"arxiv": {"max_rate": 0}})           # must be > 0
    with pytest.raises(ValidationError, match="listed twice"):
        cfg_with(raw, {"s2": {}, "semantic_scholar": {}})


def test_empty_env_expansion_means_no_key(raw):
    c = cfg_with(raw, {"semantic_scholar": {"api_key": "", "mailto": ""}})
    assert c.sources["semantic_scholar"].api_key is None


def test_legacy_list_and_source_settings_still_load(raw):
    c = cfg_with(raw, ["openalex", "s2"])
    assert list(c.sources) == ["openalex", "semantic_scholar"]
    c = cfg_with(raw, ["semantic_scholar", "openalex", "arxiv"],
                 source_settings={"semantic_scholar_api_key": "k", "openalex_mailto": "a@b.c", "request_delay": 2})
    assert c.sources["semantic_scholar"].api_key == "k" and c.sources["openalex"].mailto == "a@b.c"
    assert all(sc.max_rate == 0.5 for sc in c.sources.values())


def test_max_rate_sets_spacing(raw):
    c = cfg_with(raw, {"openalex": {"max_rate": 4}, "arxiv": {"max_rate": 0.25}})
    oa, ax = (make_source(n, c.sources[n]) for n in ("openalex", "arxiv"))
    assert oa.min_delay == 0.25 and oa._delay == 0.25
    assert ax.min_delay == 4.0


def test_defaults_without_settings():
    assert OpenAlexSource().min_delay == 0.15
    assert ArxivSource().min_delay == 3.1
    assert SemanticScholarSource().min_delay == 1.1


def test_s2_key_and_rate(raw):
    c = cfg_with(raw, {"semantic_scholar": {"api_key": "k"}})
    s = make_source("semantic_scholar", c.sources["semantic_scholar"])
    assert s._headers == {"x-api-key": "k"} and s.min_delay == s._delay == 1.0
    c = cfg_with(raw, {"semantic_scholar": {"api_key": "k", "max_rate": 0.5}})
    s = make_source("semantic_scholar", c.sources["semantic_scholar"])
    assert s.min_delay == s._delay == 2.0                  # explicit max_rate wins over the key default


def test_rate_above_documented_limit_warns(raw, caplog):
    c = cfg_with(raw, {"arxiv": {"max_rate": 1}, "semantic_scholar": {"max_rate": 1.0}})
    with caplog.at_level(logging.WARNING):
        make_source("semantic_scholar", c.sources["semantic_scholar"])
        assert "above the documented limit" not in caplog.text
        make_source("arxiv", c.sources["arxiv"])
    assert "arxiv: max_rate 1.00 req/s is above the documented limit" in caplog.text


def test_arxiv_categories(raw):
    c = cfg_with(raw, {"arxiv": {"categories": ["cs.RO"]}})
    assert make_source("arxiv", c.sources["arxiv"]).categories == ["cs.RO"]
    assert ArxivSource().categories == ArxivSource.default_categories


def test_shipped_configs_and_templates_load():
    from pathlib import Path
    root = Path(__file__).parents[1]
    for f in [*root.glob("configs/*.yaml"), root / "src/autoref/templates/mde.yaml"]:
        assert load_config(f).enabled_sources, f
