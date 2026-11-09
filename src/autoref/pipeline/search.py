"""Harvest papers from sources + relevance gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field
from sqlalchemy import select

from ..db import repo
from ..db.schema import Paper, SyncRun, TopicPaper
from ..llm import FailureGuard, LLMError, llm_for
from .context import Ctx, log, render, session, short


@dataclass
class SearchStats:
    rows: list[dict] = field(default_factory=list)  # per (source, query)
    new_paper_ids: set[int] = field(default_factory=set)  # new to the topic
    errors: list[str] = field(default_factory=list)


def run_search(ctx: Ctx, *, source: str | None = None, queries: list[str] | None = None,
               limit: int | None = None, date_from: date | None = None, date_to: date | None = None,
               origin: str = "search", run_id: int | None = None,
               skip: set[tuple[str, str]] | None = None) -> SearchStats:
    """`skip` = (source, query) pairs already finished by an interrupted earlier run. Every finished
    pair is checkpointed into the SyncRun's details["done"] so the next run can skip it."""
    cfg = ctx.cfg.search
    queries = queries or cfg.queries
    limit = limit or cfg.limit_per_query
    srcs = [ctx.sources.get(source)] if source else ctx.sources.with_cap("search")
    if source and srcs[0] is None:
        raise ValueError(f"source {source!r} not in config sources {ctx.cfg.sources}")
    stats = SearchStats()
    with ctx.progress() as prog:
        task = prog.add_task("search", total=len(srcs) * len(queries))
        for src in srcs:
            for q in queries:
                if skip and (src.name, q) in skip:
                    stats.rows.append({"source": src.name, "query": q, "found": 0, "new": 0, "merged": 0,
                                       "error": "skipped: done in interrupted run"})
                    prog.advance(task)
                    continue
                prog.update(task, description=f"{src.name}: {q[:40]}")
                row = {"source": src.name, "query": q, "found": 0, "new": 0, "merged": 0}
                try:
                    recs = src.search(q, limit=limit, year_from=cfg.year_from, year_to=cfg.year_to,
                                      date_from=date_from, date_to=date_to)
                except Exception as e:  # keep going with other sources/queries
                    msg = f"{src.name} / {q!r}: {type(e).__name__}: {e}"
                    log.warning("search failed: %s", msg)
                    stats.errors.append(msg)
                    row["error"] = str(e)[:80]
                    recs = []
                with session(ctx) as s:
                    for rec in recs:
                        if cfg.year_from and rec.year and rec.year < cfg.year_from:
                            continue
                        p, created = repo.upsert_paper(s, rec)
                        _, new_member = repo.membership(s, ctx.topic, p, origin=origin, run_id=run_id)
                        row["found"] += 1
                        row["new" if created else "merged"] += 1
                        if new_member:
                            stats.new_paper_ids.add(p.id)
                    if run_id is not None and "error" not in row:  # checkpoint in the same transaction
                        r = s.get(SyncRun, run_id)
                        r.details = {**(r.details or {}), "done": [*(r.details or {}).get("done", []), [src.name, q]]}
                stats.rows.append(row)
                prog.advance(task)
    return stats


# ------------------------------------------------------------------ relevance gate
class RelevanceOut(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def keyword_score(text: str, keywords: list[str], needed: int) -> tuple[float, str]:
    t = text.lower()
    hits = [k for k in keywords if re.search(r"\b" + re.escape(k.lower()), t)]
    return min(1.0, len(hits) / max(1, needed)), ("matched: " + ", ".join(hits)) if hits else "no keyword match"


def apply_filter(ctx: Ctx, paper_ids: list[int] | None = None, *, rescore: bool = False) -> dict:
    """Score candidate topic papers and set status included/excluded (manual overrides kept)."""
    f = ctx.cfg.search.filter
    counts = {"included": 0, "excluded": 0}
    with session(ctx) as s:
        q = select(TopicPaper).where(TopicPaper.topic == ctx.topic, TopicPaper.manual.is_(False))
        if not rescore:
            q = q.where(TopicPaper.status == "candidate")
        if paper_ids is not None:
            q = q.where(TopicPaper.paper_id.in_(list(paper_ids)))
        tps = list(s.scalars(q))
    if not tps:
        return counts
    llm = llm_for(ctx.cfg, "relevance") if f.mode == "llm" else None
    if llm:
        sys_t, user_t = ctx.prompt("relevance")
        topic_desc = f.topic_description or "; ".join(ctx.cfg.search.queries)
    guard = FailureGuard("relevance")
    with ctx.progress() as prog:
        task = prog.add_task(f"relevance ({f.mode})", total=len(tps))
        for tp in tps:
            with session(ctx) as s:
                tp = s.merge(tp)
                p = s.get(Paper, tp.paper_id)
                if f.mode == "none":
                    score, reason = 1.0, "filter disabled"
                elif f.mode == "keywords":
                    score, reason = keyword_score(f"{p.title} {p.abstract or ''}", f.keywords, f.keywords_needed)
                else:
                    try:
                        out = llm.complete_json(
                            render(sys_t, topic=topic_desc),
                            render(user_t, title=p.title, year=p.year, abstract=short(p.abstract, 2500)),
                            RelevanceOut)
                        score, reason = out.relevance, out.reason
                        guard.ok()
                    except LLMError as e:
                        log.warning("relevance LLM failed for %s: %s", p.id, e)
                        guard.failed(e)
                        prog.advance(task)
                        continue  # stays candidate, retried next time
                tp.relevance, tp.relevance_reason = score, reason[:500]
                tp.status = "included" if score >= f.min_relevance else "excluded"
                counts[tp.status] += 1
            prog.advance(task)
    return counts
