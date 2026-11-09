"""Topic configuration: one YAML file per research topic."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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


class SearchConfig(BaseModel):
    queries: list[str]
    year_from: int | None = None
    year_to: int | None = None
    limit_per_query: int = 100
    filter: FilterConfig = Field(default_factory=FilterConfig)


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


class SourceSettings(BaseModel):
    semantic_scholar_api_key: str | None = None
    openalex_mailto: str | None = None
    request_delay: float | None = None  # override per-source politeness delay


class TopicConfig(BaseModel):
    name: str
    db: str
    vault: str
    sources: list[str] = Field(default_factory=lambda: ["semantic_scholar", "openalex", "arxiv"])
    source_settings: SourceSettings = Field(default_factory=SourceSettings)
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
