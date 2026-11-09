"""OpenAlex source: huge coverage, references as IDs, abstracts as inverted index."""
from __future__ import annotations

import re
from datetime import date

from ..models import ExternalIds, PaperRecord, ReferenceRecord, normalize_arxiv, normalize_doi
from .base import PaperSource

API = "https://api.openalex.org"
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#v]+(?:v\d+)?)", re.I)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)", re.I)


def rebuild_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    return " ".join(w for _, w in sorted(positions)) or None


def _short_id(oid: str | None) -> str | None:
    return oid.rsplit("/", 1)[-1] if oid else None


def parse_work(w: dict | None) -> PaperRecord | None:
    if not w or not (w.get("title") or w.get("display_name")):
        return None
    doi = normalize_doi(w.get("doi"))
    arxiv = None
    if doi and (m := _ARXIV_DOI_RE.match(doi)):
        arxiv = m.group(1)
    for loc in w.get("locations") or []:
        if arxiv:
            break
        for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if u and (m := _ARXIV_RE.search(u)):
                arxiv = normalize_arxiv(m.group(1))
                break
    prim = w.get("primary_location") or {}
    venue = ((prim.get("source") or {}).get("display_name")) or None
    best = w.get("best_oa_location") or {}
    pdf = best.get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
    if arxiv and not pdf:
        pdf = f"https://arxiv.org/pdf/{arxiv}"
    return PaperRecord(
        title=w.get("title") or w.get("display_name"),
        ids=ExternalIds(doi=doi, arxiv_id=arxiv, openalex_id=_short_id(w.get("id"))),
        abstract=rebuild_abstract(w.get("abstract_inverted_index")),
        authors=[(a.get("author") or {}).get("display_name") for a in w.get("authorships") or []
                 if (a.get("author") or {}).get("display_name")],
        year=w.get("publication_year"),
        publication_date=w.get("publication_date"),
        venue=venue,
        url=(f"https://doi.org/{doi}" if doi else None) or prim.get("landing_page_url") or w.get("id"),
        pdf_url=pdf,
        citation_count=w.get("cited_by_count"),
        source="openalex",
    )


class OpenAlexSource(PaperSource):
    name = "openalex"
    capabilities = frozenset({"search", "get", "references", "citations", "abstract", "pdf", "date_window"})
    min_delay = 0.15

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        self._mailto = getattr(settings, "openalex_mailto", None) if settings else None

    def _params(self, **kw) -> dict:
        p = {k: v for k, v in kw.items() if v is not None}
        if self._mailto:
            p["mailto"] = self._mailto
        return p

    def _paged(self, filt: str | None, search: str | None, limit: int) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        cursor = "*"
        while cursor and len(out) < limit:
            r = self._get(f"{API}/works", self._params(
                search=search, filter=filt, cursor=cursor, **{"per-page": min(200, max(1, limit - len(out)))}))
            if r.status_code == 404:
                break
            data = r.json()
            results = data.get("results") or []
            out.extend(p for p in (parse_work(w) for w in results) if p)
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not results:
                break
        return out[:limit]

    @staticmethod
    def _date_filter(year_from, year_to, date_from, date_to) -> list[str]:
        f = []
        if date_from:
            f.append(f"from_publication_date:{date_from.isoformat()}")
        elif year_from:
            f.append(f"from_publication_date:{year_from}-01-01")
        if date_to:
            f.append(f"to_publication_date:{date_to.isoformat()}")
        elif year_to:
            f.append(f"to_publication_date:{year_to}-12-31")
        return f

    def search(self, query, limit=100, year_from=None, year_to=None, date_from=None, date_to=None):
        filt = self._date_filter(year_from, year_to, date_from, date_to)
        return self._paged(",".join(filt) or None, query, limit)

    def get(self, ids, title=None):
        key = None
        if ids.openalex_id:
            key = ids.openalex_id
        elif ids.doi:
            key = f"doi:{ids.doi}"
        elif ids.arxiv_id:
            key = f"doi:10.48550/arXiv.{normalize_arxiv(ids.arxiv_id)}"
        if key:
            r = self._get(f"{API}/works/{key}", self._params())
            if r.status_code != 404:
                return parse_work(r.json())
        if title:
            clean = re.sub(r"[,:|]", " ", title)
            res = self._paged(f"title.search:{clean}", None, 1)
            return res[0] if res else None
        return None

    def _work_json(self, paper: PaperRecord) -> dict | None:
        rec = paper if paper.ids.openalex_id else self.get(paper.ids, paper.title)
        if not rec or not rec.ids.openalex_id:
            return None
        r = self._get(f"{API}/works/{rec.ids.openalex_id}", self._params())
        return None if r.status_code == 404 else r.json()

    def references(self, paper, limit=1000):
        w = self._work_json(paper)
        if not w:
            return []
        ids = [_short_id(x) for x in (w.get("referenced_works") or [])][:limit]
        out: list[ReferenceRecord] = []
        for i in range(0, len(ids), 50):
            chunk = "|".join(ids[i:i + 50])
            out.extend(ReferenceRecord(cited=p) for p in self._paged(f"openalex_id:{chunk}", None, 50))
        return out

    def citations(self, paper, limit=1000, date_from: date | None = None):
        oid = paper.ids.openalex_id
        if not oid:
            rec = self.get(paper.ids, paper.title)
            oid = rec.ids.openalex_id if rec else None
        if not oid:
            return []
        filt = [f"cites:{oid}"]
        if date_from:
            filt.append(f"from_publication_date:{date_from.isoformat()}")
        return [ReferenceRecord(cited=p) for p in self._paged(",".join(filt), None, limit)]
