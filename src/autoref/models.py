"""Source-agnostic data records exchanged between sources, pipeline and DB."""
from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field


class ExternalIds(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None

    def any(self) -> bool:
        return any([self.doi, self.arxiv_id, self.s2_id, self.openalex_id])

    def merge(self, other: "ExternalIds") -> "ExternalIds":
        return ExternalIds(
            doi=self.doi or other.doi,
            arxiv_id=self.arxiv_id or other.arxiv_id,
            s2_id=self.s2_id or other.s2_id,
            openalex_id=self.openalex_id or other.openalex_id,
        )


class PaperRecord(BaseModel):
    title: str
    ids: ExternalIds = Field(default_factory=ExternalIds)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None  # ISO yyyy-mm-dd if known
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    source: str = ""  # which source produced this record


class CitationContext(BaseModel):
    text: str
    intents: list[str] = Field(default_factory=list)


class ReferenceRecord(BaseModel):
    """An edge citing -> cited; `cited` may be partial (stub)."""
    cited: PaperRecord
    contexts: list[CitationContext] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    is_influential: bool | None = None


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    return d or None


def normalize_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    a = aid.strip()
    a = re.sub(r"^(https?://arxiv\.org/(abs|pdf)/|arxiv:)", "", a, flags=re.I)
    a = re.sub(r"\.pdf$", "", a)
    a = re.sub(r"v\d+$", "", a)
    return a or None
