"""Topic configuration: one YAML file per research topic."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value):
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


class Category(BaseModel):
    id: str
    label: str
    description: str
    subcategories: list["Category"] = Field(default_factory=list)


class RelationLabel(BaseModel):
    relation_label: str
    label: str
    description: str


class LLMConfig(BaseModel):
    provider: str = "openai_compat"  # openai_compat | anthropic | ollama
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    json_mode: bool = True  # use provider-native JSON mode when available
    max_retries: int = 2
    # openai_compat only: extra request fields the server understands, e.g. for vLLM + Qwen3
    # {chat_template_kwargs: {enable_thinking: false}}
    extra_body: dict | None = None


class LLMSection(BaseModel):
    default: LLMConfig
    categorize: LLMConfig | None = None
    relations: LLMConfig | None = None
    fill: LLMConfig | None = None
    relevance: LLMConfig | None = None

    def for_task(self, task: str) -> LLMConfig:
        return getattr(self, task, None) or self.default


class FilterConfig(BaseModel):
    mode: Literal["none", "keywords", "llm"] = "keywords"
    keywords: list[str] = Field(default_factory=list)
    min_relevance: float = 0.5
    keywords_needed: int = 2  # keyword mode: relevance = min(1, matched_keywords / keywords_needed)
    # description of the topic, used by the LLM relevance filter
    topic_description: str = ""


class StopConfig(BaseModel):
    """Stop rule for `limit_per_query: all` (explore results until they stop being relevant).
    Relevance of a result = it passes the `filter.keywords` test."""
    min_yield: float = Field(0.2, ge=0, le=1)  # a page with a smaller share of relevant results is "low yield"
    patience: int = Field(2, ge=1)             # stop after this many low-yield pages in a row
    page_size: int = Field(50, ge=1)           # results per request while exploring (smaller = finer stop)
    max_results: int = Field(2000, ge=1)       # hard cap per query and source


class SearchConfig(BaseModel):
    queries: list[str]
    year_from: int | None = None
    year_to: int | None = None
    # A number caps the results per query and source. `all` (or null) pages through the results until
    # they stop being relevant (see `stop`) instead of using a fixed count.
    limit_per_query: int | None = 100
    # Overall cap on distinct papers gathered by one `search`/`update` run, across all queries and
    # sources (each query/source pair gets a fair share of what is left). null = no cap.
    total: int | None = Field(None, ge=1)
    stop: StopConfig = Field(default_factory=StopConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)

    @field_validator("limit_per_query", mode="before")
    @classmethod
    def _all_means_unlimited(cls, v):
        return None if isinstance(v, str) and v.strip().lower() in ("all", "unlimited", "none", "") else v


class SnowballConfig(BaseModel):
    depth: int = 1
    direction: Literal["references", "citations", "both"] = "references"
    max_per_paper: int = 50


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


class CategorizeConfig(BaseModel):
    mode: Literal["single", "multi"] = "multi"
    min_confidence: float = 0.3
    # Optional explicit scheme name. If omitted, it is a hash of the tree + mode, so ANY edit of the
    # tree marks all papers as stale. Set e.g. `scheme: v2` to control re-categorization yourself.
    scheme: str | None = None
    categories: list[Category]

    @property
    def scheme_id(self) -> str:
        return self.scheme or _digest({"mode": self.mode, "cats": [c.model_dump() for c in self.categories]})

    @model_validator(mode="after")
    def _unique_ids(self):
        seen: set[str] = set()
        for path, cat in iter_categories(self.categories):
            if cat.id in seen:
                raise ValueError(f"duplicate category id: {cat.id!r}")
            if "/" in cat.id:
                raise ValueError(f"category id must not contain '/': {cat.id!r}")
            seen.add(cat.id)
        return self

    def path_of(self, cat_id: str) -> str | None:
        for path, cat in iter_categories(self.categories):
            if cat.id == cat_id:
                return path
        return None

    def get(self, cat_id: str) -> Category | None:
        for _, cat in iter_categories(self.categories):
            if cat.id == cat_id:
                return cat
        return None


class RelationsConfig(BaseModel):
    labels: list[RelationLabel]
    min_confidence: float = 0.0
    scheme: str | None = None  # like categorize.scheme; default = hash of the label set

    @property
    def scheme_id(self) -> str:
        return self.scheme or _digest([l.model_dump() for l in self.labels])

    @field_validator("labels")
    @classmethod
    def _unique(cls, v):
        keys = [l.relation_label for l in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate relation_label")
        return v

    def get(self, key: str) -> RelationLabel | None:
        return next((l for l in self.labels if l.relation_label == key), None)


class FillMissingConfig(BaseModel):
    enabled: bool = True
    allow_llm: bool = True
    allow_pdf: bool = True
    fields: list[str] = Field(default_factory=lambda: ["abstract", "year", "venue", "authors"])


class UpdateConfig(BaseModel):
    overlap_days: int = 3
    track_citations_of_core: bool = True
    core_min_relevance: float = 0.8
    default_since_days: int = 30  # used when no previous successful run exists


class ExportConfig(BaseModel):
    inbox_days: int = 14  # papers added within N days get tag status/new and appear in Inbox.md
    canvas_categories: list[str] = Field(default_factory=list)  # category ids exported as .canvas


class SourceConfig(BaseModel):
    """Settings of one paper source (key under `sources:` is the source name)."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    api_key: str | None = None      # Semantic Scholar
    mailto: str | None = None       # OpenAlex "polite pool" e-mail
    max_rate: float | None = Field(default=None, gt=0)  # max requests per second; None = safe default
    categories: list[str] | None = None  # arXiv: restrict search to these categories

    @field_validator("api_key", "mailto", mode="before")
    @classmethod
    def _empty_is_none(cls, v):  # `${S2_API_KEY:-}` expands to "" when the variable is unset
        return v or None


DEFAULT_SOURCES = ("semantic_scholar", "openalex", "arxiv")


def normalize_sources(value) -> dict:
    """Accept `sources:` as mapping (preferred) or legacy list of names; canonicalize aliases (s2, oa)
    and validate names. Order is preserved: it is the source priority."""
    from .sources.registry import ALIASES, SOURCES
    if value is None:
        value = list(DEFAULT_SOURCES)
    if isinstance(value, (list, tuple)):
        value = {name: None for name in value}
    if not isinstance(value, dict):
        raise ValueError("`sources` must be a mapping of source name -> settings")
    out: dict = {}
    for name, body in value.items():
        key = ALIASES.get(name, name)
        if key not in SOURCES:
            raise ValueError(f"unknown source {name!r}; available: {', '.join(SOURCES)} (aliases: {', '.join(ALIASES)})")
        if key in out:
            raise ValueError(f"source {key!r} listed twice")
        out[key] = body if body is not None else {}
    return out


class TopicConfig(BaseModel):
    name: str
    db: str
    vault: str
    # Ordered mapping: first source has the highest priority for references/citations/lookups.
    sources: dict[str, SourceConfig] = Field(default_factory=lambda: {n: SourceConfig() for n in DEFAULT_SOURCES})
    search: SearchConfig
    snowball: SnowballConfig = Field(default_factory=SnowballConfig)
    llm: LLMSection
    categorize: CategorizeConfig
    relations: RelationsConfig
    fill_missing: FillMissingConfig = Field(default_factory=FillMissingConfig)
    update: UpdateConfig = Field(default_factory=UpdateConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    # set by loader: directory of the YAML file; relative paths resolve against it
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _legacy_source_settings(cls, data):
        """Old format: `sources: [names]` + `source_settings: {semantic_scholar_api_key, openalex_mailto,
        request_delay}` -> new per-source mapping."""
        if not isinstance(data, dict) or "source_settings" not in data:
            return data
        data = dict(data)
        legacy = data.pop("source_settings") or {}
        srcs = {k: dict(v or {}) for k, v in normalize_sources(data.get("sources")).items()}
        if legacy.get("semantic_scholar_api_key") and "semantic_scholar" in srcs:
            srcs["semantic_scholar"].setdefault("api_key", legacy["semantic_scholar_api_key"])
        if legacy.get("openalex_mailto") and "openalex" in srcs:
            srcs["openalex"].setdefault("mailto", legacy["openalex_mailto"])
        if legacy.get("request_delay"):
            for body in srcs.values():
                body.setdefault("max_rate", 1.0 / float(legacy["request_delay"]))
        data["sources"] = srcs
        return data

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, v):
        return normalize_sources(v)

    @property
    def enabled_sources(self) -> list[str]:
        return [name for name, sc in self.sources.items() if sc.enabled]

    @property
    def db_path(self) -> Path:
        p = Path(self.db).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def vault_path(self) -> Path:
        p = Path(self.vault).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def iter_categories(cats: list[Category], prefix: str = ""):
    """Yield (path, category) for the whole tree, depth first."""
    for c in cats:
        path = f"{prefix}/{c.id}" if prefix else c.id
        yield path, c
        yield from iter_categories(c.subcategories, path)


def load_config(path: str | Path) -> TopicConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)
    cfg = TopicConfig.model_validate(raw)
    # Config lives in <project>/configs/x.yaml -> resolve relative paths against project root
    cfg.base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    return cfg
