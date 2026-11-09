"""Name -> PaperSource class. Add new sources here."""
from __future__ import annotations

from .arxiv import ArxivSource
from .base import PaperSource
from .openalex import OpenAlexSource
from .semantic_scholar import SemanticScholarSource

SOURCES: dict[str, type[PaperSource]] = {
    "semantic_scholar": SemanticScholarSource,
    "openalex": OpenAlexSource,
    "arxiv": ArxivSource,
}
ALIASES = {"s2": "semantic_scholar", "oa": "openalex"}


def make_source(name: str, settings=None) -> PaperSource:
    key = ALIASES.get(name, name)
    if key not in SOURCES:
        raise ValueError(f"unknown source {name!r}; available: {', '.join(SOURCES)}")
    return SOURCES[key](settings)


class SourceSet:
    """Ordered collection of configured sources (priority = config order)."""

    def __init__(self, names: list[str], settings=None):
        self.sources = [make_source(n, settings) for n in names]

    def __iter__(self):
        return iter(self.sources)

    def with_cap(self, cap: str) -> list[PaperSource]:
        return [s for s in self.sources if s.has(cap)]

    def get(self, name: str) -> PaperSource | None:
        key = ALIASES.get(name, name)
        return next((s for s in self.sources if s.name == key), None)

    def close(self):
        for s in self.sources:
            s.close()
