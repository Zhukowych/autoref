"""arXiv Atom API source: freshest preprints (cs.CV / cs.RO). No references -> enriched elsewhere."""
from __future__ import annotations

import re

import feedparser

from ..models import ExternalIds, PaperRecord, normalize_arxiv, normalize_doi
from .base import PaperSource

API = "https://export.arxiv.org/api/query"
_STOP = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "with", "to", "from", "by"}


def build_query(query: str, categories: list[str] | None = None) -> str:
    """Turn free text into arXiv syntax. Supports quoted phrases; other words AND-ed."""
    if any(tok in query for tok in ("ti:", "abs:", "all:", "cat:")):
        return query  # already arXiv syntax
    parts = []
    for phrase in re.findall(r'"([^"]+)"', query):
        parts.append(f'all:"{phrase}"')
    rest = re.sub(r'"[^"]+"', " ", query)
    parts += [f"all:{w}" for w in re.findall(r"[\w\-]+", rest) if w.lower() not in _STOP]
    q = " AND ".join(parts)
    if categories:
        q = f"({q}) AND ({' OR '.join('cat:' + c for c in categories)})"
    return q


def parse_feed(text: str) -> list[PaperRecord]:
    feed = feedparser.parse(text)
    out = []
    for e in feed.entries:
        aid = normalize_arxiv(e.get("id", ""))
        if not aid or not e.get("title"):
            continue
        pdf = next((l.get("href") for l in e.get("links", []) if l.get("type") == "application/pdf"), None)
        pub = (e.get("published") or "")[:10] or None
        out.append(PaperRecord(
            title=re.sub(r"\s+", " ", e.title).strip(),
            ids=ExternalIds(arxiv_id=aid, doi=normalize_doi(e.get("arxiv_doi"))),
            abstract=re.sub(r"\s+", " ", e.get("summary", "")).strip() or None,
            authors=[a.get("name") for a in e.get("authors", []) if a.get("name")],
            year=int(pub[:4]) if pub else None,
            publication_date=pub,
            venue=e.get("arxiv_journal_ref") or "arXiv",
            url=f"https://arxiv.org/abs/{aid}",
            pdf_url=pdf or f"https://arxiv.org/pdf/{aid}",
            source="arxiv",
        ))
    return out


class ArxivSource(PaperSource):
    name = "arxiv"
    capabilities = frozenset({"search", "get", "abstract", "pdf", "date_window"})
    min_delay = 3.1  # arXiv asks for >= 3 s between calls
    rate_limit = 1 / 3.0

    default_categories = ["cs.CV", "cs.RO", "cs.LG", "cs.AI", "eess.IV"]

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        self.categories = list(getattr(settings, "categories", None) or self.default_categories)

    def search(self, query, limit=100, year_from=None, year_to=None, date_from=None, date_to=None):
        q = build_query(query, self.categories)
        if date_from or date_to or year_from or year_to:
            lo = (date_from.strftime("%Y%m%d") if date_from else f"{year_from or 1991}0101") + "0000"
            hi = (date_to.strftime("%Y%m%d") if date_to else f"{year_to or 2100}1231") + "2359"
            q = f"({q}) AND submittedDate:[{lo} TO {hi}]"
        out: list[PaperRecord] = []
        start = 0
        while len(out) < limit:
            page = min(100, limit - len(out))
            r = self._get(API, {"search_query": q, "start": start, "max_results": page,
                                "sortBy": "submittedDate" if date_from else "relevance",
                                "sortOrder": "descending"})
            batch = parse_feed(r.text)
            out.extend(batch)
            if len(batch) < page:
                break
            start += page
        return out[:limit]

    def get(self, ids, title=None):
        if ids.arxiv_id:
            r = self._get(API, {"id_list": normalize_arxiv(ids.arxiv_id), "max_results": 1})
            res = parse_feed(r.text)
            return res[0] if res else None
        if title:
            clean = re.sub(r'["():]', " ", title)
            r = self._get(API, {"search_query": f'ti:"{clean.strip()}"', "max_results": 1})
            res = parse_feed(r.text)
            from ..models import normalize_title
            if res and normalize_title(res[0].title) == normalize_title(title):
                return res[0]
        return None
