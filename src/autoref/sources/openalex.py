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
    rate_limit = 10.0  # documented: 10 requests/second (check current OpenAlex docs)

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        self._mailto = getattr(settings, "mailto", None) if settings else None
        self._api_key = getattr(settings, "api_key", None) if settings else None
        # credit budget as reported by the X-RateLimit-* response headers
        self.credits_remaining: int | None = None
        self.credits_limit: int | None = None
        self.credits_reset: int | None = None   # seconds until the budget resets
        self.credits_cost = 10                  # credits one list/search request costs

    def _params(self, **kw) -> dict:
        p = {k: v for k, v in kw.items() if v is not None}
        if self._mailto:
            p["mailto"] = self._mailto
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    def _get(self, url, params=None, headers=None):
        r = super()._get(url, params, headers)
        h = r.headers
        try:
            self.credits_remaining = int(h["x-ratelimit-remaining"])
            self.credits_limit = int(h["x-ratelimit-limit"])
            self.credits_reset = int(h["x-ratelimit-reset"])
            self.credits_cost = int(h.get("x-ratelimit-credits-used", self.credits_cost)) or self.credits_cost
        except (KeyError, ValueError):
            pass
        return r

    @property
    def budget_info(self):
        if self.credits_remaining is None:
            return None
        reset = f", resets in {self.credits_reset // 3600}h" if self.credits_reset else ""
        return f"{self.credits_remaining}/{self.credits_limit} credits{reset}"

    def _budget_exhausted(self) -> bool:
        if self.credits_remaining is not None and self.credits_remaining < self.credits_cost:
            self.stop_reason = f"OpenAlex daily budget used up ({self.budget_info})"
            return True
        return False

    PAGE_LIMIT = 10_000  # basic (page=N) paging cannot go past 10,000 results

    def _pages(self, filt: str | None, search: str | None, page_size: int, max_results: int | None):
        """Cursor paging needs an API key (OpenAlex answers 401 without one); without a key use
        page=N paging, which is limited to the first 10,000 results."""
        use_cursor = bool(self._api_key)
        cap = max_results
        if not use_cursor:
            cap = min(max_results or self.PAGE_LIMIT, self.PAGE_LIMIT)
        per = max(1, min(200, page_size, cap if cap is not None else page_size))  # constant: page=N offsets depend on it
        got, cursor, page_no = 0, "*", 1
        while cap is None or got < cap:
            if self._budget_exhausted():
                return
            r = self._get(f"{API}/works", self._params(
                search=search, filter=filt, **{"per-page": per},
                **({"cursor": cursor} if use_cursor else {"page": page_no})))
            if r.status_code == 404:
                break
            data = r.json()
            meta = data.get("meta") or {}
            self.last_total = meta.get("count", self.last_total)
            results = data.get("results") or []
            if cap is not None:
                results = results[: cap - got]
            got += len(results)
            recs = [p for p in (parse_work(w) for w in results) if p]
            if recs:
                yield recs
            if use_cursor:
                cursor = meta.get("next_cursor")
                if not cursor:
                    break
            else:
                page_no += 1
            if len(results) < per:
                break

    def _paged(self, filt: str | None, search: str | None, limit: int) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        for page in self._pages(filt, search, 200, limit):
            out.extend(page)
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

    def search_pages(self, query, page_size=100, max_results=None, *, year_from=None, year_to=None,
                     date_from=None, date_to=None):
        filt = self._date_filter(year_from, year_to, date_from, date_to)
        yield from self._pages(",".join(filt) or None, query, page_size, max_results)

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
