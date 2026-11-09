"""Semantic Scholar Graph API source. Best source for references + citation contexts/intents."""
from __future__ import annotations

from datetime import date

from ..models import CitationContext, ExternalIds, PaperRecord, ReferenceRecord, normalize_arxiv
from .base import PaperSource

API = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = ("paperId,externalIds,title,abstract,year,venue,publicationDate,authors,url,"
                "citationCount,openAccessPdf")
EDGE_PAPER_FIELDS = "paperId,externalIds,title,year,venue,publicationDate,authors,url,abstract,citationCount"


def _s2_id_param(ids: ExternalIds) -> str | None:
    if ids.s2_id:
        return ids.s2_id
    if ids.doi:
        return f"DOI:{ids.doi}"
    if ids.arxiv_id:
        return f"ARXIV:{normalize_arxiv(ids.arxiv_id)}"
    return None


def parse_paper(d: dict | None) -> PaperRecord | None:
    if not d or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    pdf = (d.get("openAccessPdf") or {}).get("url") or None
    arx = ext.get("ArXiv")
    if not pdf and arx:
        pdf = f"https://arxiv.org/pdf/{arx}"
    return PaperRecord(
        title=d["title"],
        ids=ExternalIds(doi=ext.get("DOI"), arxiv_id=arx, s2_id=d.get("paperId")),
        abstract=d.get("abstract"),
        authors=[a.get("name") for a in d.get("authors") or [] if a.get("name")],
        year=d.get("year"),
        publication_date=d.get("publicationDate"),
        venue=d.get("venue") or None,
        url=d.get("url") or (f"https://www.semanticscholar.org/paper/{d['paperId']}" if d.get("paperId") else None),
        pdf_url=pdf,
        citation_count=d.get("citationCount"),
        source="semantic_scholar",
    )


def _parse_edge(e: dict, key: str) -> ReferenceRecord | None:
    rec = parse_paper(e.get(key))
    if rec is None:
        return None
    intents = e.get("intents") or []
    return ReferenceRecord(
        cited=rec,
        contexts=[CitationContext(text=c, intents=intents) for c in (e.get("contexts") or [])],
        intents=intents,
        is_influential=e.get("isInfluential"),
    )


class SemanticScholarSource(PaperSource):
    name = "semantic_scholar"
    capabilities = frozenset({"search", "get", "references", "citations", "contexts", "abstract",
                              "pdf", "date_window"})
    min_delay = 1.1

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        key = getattr(settings, "semantic_scholar_api_key", None) if settings else None
        self._headers = {"x-api-key": key} if key else None
        if key and (settings is None or settings.request_delay is None):
            self.min_delay = 1.0

    def search(self, query, limit=100, year_from=None, year_to=None, date_from=None, date_to=None):
        params: dict = {"query": query, "fields": PAPER_FIELDS}
        if date_from or date_to:
            params["publicationDateOrYear"] = (
                f"{date_from.isoformat() if date_from else ''}:{date_to.isoformat() if date_to else ''}")
        elif year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        out: list[PaperRecord] = []
        offset = 0
        while len(out) < limit and offset < 1000:
            page = min(100, limit - len(out))
            r = self._get(f"{API}/paper/search", {**params, "offset": offset, "limit": page}, self._headers)
            if r.status_code == 404:
                break
            data = r.json()
            items = data.get("data") or []
            out.extend(p for p in (parse_paper(i) for i in items) if p)
            if "next" not in data or not items:
                break
            offset = data["next"]
        return out[:limit]

    def get(self, ids, title=None):
        pid = _s2_id_param(ids)
        if pid:
            r = self._get(f"{API}/paper/{pid}", {"fields": PAPER_FIELDS}, self._headers)
            if r.status_code != 404:
                return parse_paper(r.json())
        if title:
            r = self._get(f"{API}/paper/search/match", {"query": title, "fields": PAPER_FIELDS}, self._headers)
            if r.status_code != 404:
                data = (r.json().get("data") or [None])[0]
                return parse_paper(data)
        return None

    def _edges(self, paper: PaperRecord, kind: str, key: str, limit: int) -> list[ReferenceRecord]:
        pid = _s2_id_param(paper.ids)
        if not pid:
            found = self.get(paper.ids, paper.title)
            pid = found.ids.s2_id if found else None
        if not pid:
            return []
        fields = f"contexts,intents,isInfluential,{','.join(key + '.' + f for f in EDGE_PAPER_FIELDS.split(','))}"
        out: list[ReferenceRecord] = []
        offset = 0
        while len(out) < limit:
            page = min(1000, limit - len(out))
            r = self._get(f"{API}/paper/{pid}/{kind}", {"fields": fields, "offset": offset, "limit": page},
                          self._headers)
            if r.status_code == 404:
                break
            data = r.json()
            items = data.get("data") or []
            out.extend(x for x in (_parse_edge(e, key) for e in items) if x)
            if "next" not in data or not items:
                break
            offset = data["next"]
        return out

    def references(self, paper, limit=1000):
        return self._edges(paper, "references", "citedPaper", limit)

    def citations(self, paper, limit=1000, date_from: date | None = None):
        edges = self._edges(paper, "citations", "citingPaper", limit)
        if date_from:
            edges = [e for e in edges if _after(e.cited, date_from)]
        return edges

    def citation_contexts(self, citing, cited):
        for ref in self.references(citing):
            c = ref.cited
            if (cited.ids.s2_id and c.ids.s2_id == cited.ids.s2_id) or \
               (cited.ids.doi and c.ids.doi and c.ids.doi.lower() == cited.ids.doi.lower()):
                return ref.contexts
        return []


def _after(rec: PaperRecord, d: date) -> bool:
    if rec.publication_date:
        return rec.publication_date >= d.isoformat()
    return rec.year is not None and rec.year >= d.year
