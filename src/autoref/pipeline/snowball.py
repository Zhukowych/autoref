"""Reference / citation expansion (snowballing) by depth.

Expansion is level-based on the depth stored in `topic_papers`, and candidate selection reads the
*stored* edges. So re-running with a larger `--depth`, another `--direction` or a larger
`--max-per-paper` after everything was downloaded only fetches what is missing:
  * papers whose references/citations were already fetched cost no API calls;
  * more neighbours are promoted from edges already in the DB.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import and_, or_, select

from ..db import repo
from ..db.schema import Paper, Reference, TopicPaper, utcnow
from ..models import ReferenceRecord
from .context import Ctx, log, session
from .search import apply_filter


def _fetch_edges(ctx: Ctx, rec, kind: str, date_from: date | None = None) -> tuple[list[ReferenceRecord], str | None]:
    """Try sources with capability `kind` in config order until one returns edges."""
    for src in ctx.sources.with_cap(kind):
        try:
            edges = src.references(rec) if kind == "references" else src.citations(rec, date_from=date_from)
        except Exception as e:
            log.warning("%s %s failed for %r: %s", src.name, kind, rec.title[:60], e)
            continue
        if edges:
            return edges, src.name
    return [], None


def fetch_edges(ctx: Ctx, paper_id: int, direction: str, *, date_from: date | None = None,
                refresh: bool = False) -> int:
    """Download and store references and/or citations of one paper (skips what's already fetched,
    unless `refresh` or a `date_from` window is given). Returns number of API fetches done."""
    with session(ctx) as s:
        p = s.get(Paper, paper_id)
        rec = repo.to_record(p)
        need_refs = direction in ("references", "both") and (refresh or p.refs_fetched_at is None)
        need_cits = direction in ("citations", "both") and (refresh or date_from or p.cites_fetched_at is None)
    fetched = 0
    if need_refs:
        edges, src = _fetch_edges(ctx, rec, "references")
        fetched += 1
        with session(ctx) as s:
            p = s.get(Paper, paper_id)
            for e in edges:
                repo.add_reference(s, p, e, src or "")
            p.refs_fetched_at = utcnow()
    if need_cits:
        edges, src = _fetch_edges(ctx, rec, "citations", date_from)
        fetched += 1
        with session(ctx) as s:
            p = s.get(Paper, paper_id)
            for e in edges:
                citing, _ = repo.upsert_paper(s, e.cited, stub=True)  # e.cited holds the citing paper
                if citing.id == p.id:
                    continue
                repo.add_reference(s, citing, ReferenceRecord(cited=rec, contexts=e.contexts, intents=e.intents,
                                                              is_influential=e.is_influential), src or "", cited=p)
            p.cites_fetched_at = utcnow()
    return fetched


def stored_neighbours(s, paper_id: int, direction: str, limit: int, date_from: date | None = None) -> list[int]:
    """Top-`limit` neighbours per direction from stored edges, ranked by influence then citations."""
    out: list[int] = []
    for kind in (["references"] if direction == "references" else
                 ["citations"] if direction == "citations" else ["references", "citations"]):
        mine, other = ((Reference.citing_id, Reference.cited_id) if kind == "references"
                       else (Reference.cited_id, Reference.citing_id))
        q = select(other).join(Paper, Paper.id == other).where(mine == paper_id)
        if date_from is not None:  # update mode: only recently published neighbours
            q = q.where(or_(Paper.publication_date >= date_from.isoformat(),
                            and_(Paper.publication_date.is_(None), Paper.year >= date_from.year)))
        q = (q.order_by(Reference.is_influential.desc().nullslast(), Paper.citation_count.desc().nullslast())
             .limit(limit))
        out += [pid for pid in s.scalars(q) if pid != paper_id]
    return out


def expand_paper(ctx: Ctx, paper_id: int, level: int, direction: str, max_per_paper: int,
                 date_from: date | None = None, origin: str = "snowball", run_id: int | None = None,
                 refresh: bool = False) -> set[int]:
    """Fetch missing edges of one paper, then make its top neighbours topic candidates at depth
    level+1. Returns ids that are new to the topic."""
    fetch_edges(ctx, paper_id, direction, date_from=date_from, refresh=refresh)
    new: set[int] = set()
    with session(ctx) as s:
        for nid in stored_neighbours(s, paper_id, direction, max_per_paper, date_from):
            _, created = repo.membership(s, ctx.topic, s.get(Paper, nid), origin=origin, depth=level + 1,
                                         run_id=run_id)
            if created:
                new.add(nid)
    return new


def promote_included(ctx: Ctx, paper_ids: set[int]) -> None:
    """Included stubs are upgraded to full records (missing data is completed later by `enrich`)."""
    with session(ctx) as s:
        for pid in ctx.included_ids(s, list(paper_ids)):
            p = s.get(Paper, pid)
            if p.is_stub:
                p.is_stub = False


def run_snowball(ctx: Ctx, *, depth: int | None = None, direction: str | None = None,
                 max_per_paper: int | None = None, paper_ids: list[int] | None = None,
                 origin: str = "snowball", run_id: int | None = None, refresh: bool = False) -> dict:
    """Expand levels 0..depth-1.

    Without `paper_ids` the frontier of level L is every included paper stored at depth L, so a
    later `--depth 2` continues from the depth-1 papers of an earlier run. With `paper_ids`
    (incremental update) only those papers and the neighbours they bring in are expanded.
    """
    sc = ctx.cfg.snowball
    depth = sc.depth if depth is None else depth
    direction = direction or sc.direction
    max_per_paper = max_per_paper or sc.max_per_paper
    summary = {"expanded": 0, "candidates": 0, "included": 0}
    # resume: candidates promoted by an interrupted run were never scored -> score them now
    with session(ctx) as s:
        leftover = list(s.scalars(select(TopicPaper.paper_id).where(
            TopicPaper.topic == ctx.topic, TopicPaper.status == "candidate")))
    if leftover:
        summary["included"] += apply_filter(ctx, leftover)["included"]
        promote_included(ctx, set(leftover))
    current = set(paper_ids) if paper_ids is not None else None
    for level in range(depth):
        with session(ctx) as s:
            if current is None:
                frontier = list(s.scalars(select(TopicPaper.paper_id).where(
                    TopicPaper.topic == ctx.topic, TopicPaper.status == "included", TopicPaper.depth == level)))
            else:
                frontier = ctx.included_ids(s, list(current))
            depths = {pid: s.get(TopicPaper, (ctx.topic, pid)).depth for pid in frontier}
        if not frontier:
            if current is not None:
                break
            continue
        new: set[int] = set()
        with ctx.progress() as prog:
            task = prog.add_task(f"snowball level {level + 1}/{depth} ({direction})", total=len(frontier))
            for pid in frontier:
                new |= expand_paper(ctx, pid, depths[pid], direction, max_per_paper,
                                    origin=origin, run_id=run_id, refresh=refresh)
                prog.advance(task)
        summary["expanded"] += len(frontier)
        summary["candidates"] += len(new)
        summary["included"] += apply_filter(ctx, list(new))["included"]
        promote_included(ctx, new)
        if current is not None:
            current = new
    return summary
