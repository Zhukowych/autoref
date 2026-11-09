"""Data-access helpers: dedup, merge, membership, provenance."""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..models import PaperRecord, ReferenceRecord, normalize_arxiv, normalize_doi, normalize_title
from .schema import (
    Author, FieldProvenance, Paper, PaperAuthor, Reference, TopicPaper, utcnow,
)

MERGE_FIELDS = ["abstract", "year", "publication_date", "venue", "url", "pdf_url", "citation_count"]


def find_paper(s: Session, rec: PaperRecord) -> Paper | None:
    """Dedup order: DOI -> arXiv -> S2 -> OpenAlex -> normalized title (+year if both known)."""
    ids = rec.ids
    doi, arx = normalize_doi(ids.doi), normalize_arxiv(ids.arxiv_id)
    for col, val in ((Paper.doi, doi), (Paper.arxiv_id, arx), (Paper.s2_id, ids.s2_id),
                     (Paper.openalex_id, ids.openalex_id)):
        if val:
            p = s.scalar(select(Paper).where(col == val).limit(1))
            if p:
                return p
    nt = normalize_title(rec.title)
    if len(nt) < 12:  # too short to match safely
        return None
    cands = s.scalars(select(Paper).where(Paper.norm_title == nt)).all()
    for p in cands:
        if rec.year is None or p.year is None or abs(p.year - rec.year) <= 1:
            return p
    return None


def find_by_key(s: Session, key: str) -> Paper | None:
    """Lookup by numeric id, DOI, arXiv id, S2/OpenAlex id or title substring."""
    if key.isdigit():
        p = s.get(Paper, int(key))
        if p:
            return p
    doi, arx = normalize_doi(key), normalize_arxiv(key)
    p = s.scalar(select(Paper).where(or_(
        Paper.doi == doi, Paper.arxiv_id == arx, Paper.s2_id == key, Paper.openalex_id == key)).limit(1))
    if p:
        return p
    return s.scalar(select(Paper).where(Paper.title.ilike(f"%{key}%")).order_by(Paper.is_stub).limit(1))


def set_provenance(s: Session, paper: Paper, field: str, origin: str, model: str | None = None) -> None:
    fp = s.get(FieldProvenance, (paper.id, field))
    if fp:
        fp.origin, fp.model = origin, model
    else:
        s.add(FieldProvenance(paper_id=paper.id, field=field, origin=origin, model=model))


def _set_authors(s: Session, paper: Paper, names: list[str]) -> None:
    paper.authors.clear()
    s.flush()
    seen = set()
    for i, name in enumerate(n.strip() for n in names):
        if not name or name in seen:
            continue
        seen.add(name)
        a = s.scalar(select(Author).where(Author.name == name))
        if not a:
            a = Author(name=name)
            s.add(a)
            s.flush()
        paper.authors.append(PaperAuthor(author=a, position=i))


def upsert_paper(s: Session, rec: PaperRecord, *, stub: bool = False) -> tuple[Paper, bool]:
    """Insert or merge a record. Returns (paper, created). Never overwrites existing non-null
    fields except upgrading a stub's data; records provenance of every filled field."""
    p = find_paper(s, rec)
    created = p is None
    if created:
        p = Paper(title=rec.title.strip(), norm_title=normalize_title(rec.title), is_stub=stub, sources="")
        s.add(p)
        s.flush()
    merge_record(s, p, rec, stub=stub)
    return p, created


def merge_record(s: Session, p: Paper, rec: PaperRecord, *, stub: bool = False) -> list[str]:
    """Fill empty fields of `p` from `rec`. Returns names of fields that were filled."""
    filled: list[str] = []
    ids = rec.ids
    for attr, val in (("doi", normalize_doi(ids.doi)), ("arxiv_id", normalize_arxiv(ids.arxiv_id)),
                      ("s2_id", ids.s2_id), ("openalex_id", ids.openalex_id)):
        if val and not getattr(p, attr):
            setattr(p, attr, val)
    for f in MERGE_FIELDS:
        val = getattr(rec, f)
        if val not in (None, "") and getattr(p, f) in (None, ""):
            setattr(p, f, val)
            filled.append(f)
            if rec.source:
                set_provenance(s, p, f, rec.source)
    if rec.authors and not p.authors:
        _set_authors(s, p, rec.authors)
        filled.append("authors")
        if rec.source:
            set_provenance(s, p, "authors", rec.source)
    if p.is_stub and not stub:
        p.is_stub = False
        if len(rec.title) > len(p.title or ""):
            p.title = rec.title.strip()
            p.norm_title = normalize_title(rec.title)
    if rec.source and rec.source not in (p.sources or "").split(","):
        p.sources = ",".join(x for x in [p.sources, rec.source] if x)
    s.flush()
    return filled


def add_reference(s: Session, citing: Paper, ref: ReferenceRecord, source: str,
                  cited: Paper | None = None) -> tuple[Reference | None, Paper]:
    if cited is None:
        cited, _ = upsert_paper(s, ref.cited, stub=True)
    if cited.id == citing.id:
        return None, cited
    r = s.scalar(select(Reference).where(Reference.citing_id == citing.id, Reference.cited_id == cited.id))
    ctx = [c.model_dump() for c in ref.contexts]
    if r is None:
        r = Reference(citing_id=citing.id, cited_id=cited.id, raw_title=ref.cited.title,
                      contexts=ctx, intents=ref.intents, is_influential=ref.is_influential, source=source)
        s.add(r)
    else:
        if ctx and not r.contexts:
            r.contexts = ctx
        if ref.intents and not r.intents:
            r.intents = ref.intents
    s.flush()
    return r, cited


def membership(s: Session, topic: str, paper: Paper, *, origin: str = "search", depth: int = 0,
               run_id: int | None = None) -> tuple[TopicPaper, bool]:
    tp = s.get(TopicPaper, (topic, paper.id))
    if tp:
        if depth < tp.depth:
            tp.depth = depth
        return tp, False
    tp = TopicPaper(topic=topic, paper_id=paper.id, origin=origin, depth=depth,
                    added_run_id=run_id, added_at=utcnow())
    s.add(tp)
    s.flush()
    return tp, True


def to_record(p: Paper) -> PaperRecord:
    from ..models import ExternalIds
    return PaperRecord(
        title=p.title, abstract=p.abstract, year=p.year, publication_date=p.publication_date,
        venue=p.venue, url=p.url, pdf_url=p.pdf_url, authors=p.author_names,
        ids=ExternalIds(doi=p.doi, arxiv_id=p.arxiv_id, s2_id=p.s2_id, openalex_id=p.openalex_id),
    )
