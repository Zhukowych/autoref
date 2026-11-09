"""Harvest papers from sources + relevance gate."""
from __future__ import annotations

import math
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
    gathered: int = 0  # distinct papers touched by this run (new or already known)
    new_paper_ids: set[int] = field(default_factory=set)  # new to the topic
    errors: list[str] = field(default_factory=list)


def run_search(ctx: Ctx, *, source: str | None = None, queries: list[str] | None = None,
               limit: int | None = None, exhaustive: bool = False, total: int | None = None,
               date_from: date | None = None, date_to: date | None = None,
               origin: str = "search", run_id: int | None = None,
               skip: set[tuple[str, str]] | None = None) -> SearchStats:
    """Run queries on sources, store results page by page.

    Result count per (source, query): `limit` / `search.limit_per_query` if it is a number. If it is
    `all` (or `exhaustive=True`) pages are fetched until `search.stop` says the results are no longer
    relevant, or `stop.max_results` is reached.

    `total` / `search.total` caps the distinct papers gathered over the whole run: every (source, query)
    pair gets a fair share of what is left (unused share rolls over to the pairs after it) and pairs
    after the cap is reached are skipped.

    `skip` = (source, query) pairs already finished by an interrupted earlier run. Every finished
    pair is checkpointed into the SyncRun's details["done"] so the next run can skip it."""
    cfg = ctx.cfg.search
    queries = queries or cfg.queries
    max_results = None if exhaustive else (limit or cfg.limit_per_query)
    explore = max_results is None
    stop, flt = cfg.stop, cfg.filter
    cap = stop.max_results if explore else max_results
    page_size = stop.page_size if explore else 100
    use_yield = explore and bool(flt.keywords)
    if explore and not use_yield:
        ctx.console.print("[yellow]limit_per_query: all, but search.filter.keywords is empty -> cannot judge "
                          f"relevance; only stop.max_results ({cap}) ends a query[/]")
    srcs = [ctx.sources.get(source)] if source else ctx.sources.with_cap("search")
    if source and srcs[0] is None:
        raise ValueError(f"source {source!r} not enabled in config (sources: {', '.join(ctx.cfg.enabled_sources)})")
    stats = SearchStats()
    total = total if total is not None else cfg.total
    seen: set[int] = set()  # distinct paper ids touched in this run
    pairs_left = sum(1 for src in srcs for q in queries if not (skip and (src.name, q) in skip))
    with ctx.progress() as prog:
        task = prog.add_task("search", total=len(srcs) * len(queries))
        for src in srcs:
            for q in queries:
                if skip and (src.name, q) in skip:
                    stats.rows.append({"source": src.name, "query": q, "found": 0, "new": 0, "merged": 0,
                                       "note": "skipped: done in interrupted run"})
                    prog.advance(task)
                    continue
                row = {"source": src.name, "query": q, "found": 0, "new": 0, "merged": 0,
                       "total": None, "pages": 0, "note": ""}
                pair_cap, share_limited = cap, False
                if total is not None:
                    remaining = total - len(seen)
                    if remaining <= 0:
                        row["note"] = f"skipped: total of {total} papers reached"
                        stats.rows.append(row)
                        pairs_left -= 1
                        prog.advance(task)
                        continue
                    share = math.ceil(remaining / pairs_left)
                    if share < cap:
                        pair_cap, share_limited = share, True
                pairs_left -= 1
                low_pages = 0
                src.stop_reason = None
                try:
                    for page in src.search_pages(q, page_size=min(page_size, pair_cap), max_results=pair_cap,
                                                 year_from=cfg.year_from, year_to=cfg.year_to,
                                                 date_from=date_from, date_to=date_to):
                        row["pages"] += 1
                        n = relevant = 0
                        with session(ctx) as s:  # committed per page: an interruption loses at most one page
                            for rec in page:
                                if cfg.year_from and rec.year and rec.year < cfg.year_from:
                                    continue
                                p, created = repo.upsert_paper(s, rec)
                                _, new_member = repo.membership(s, ctx.topic, p, origin=origin, run_id=run_id)
                                n += 1
                                seen.add(p.id)
                                row["found"] += 1
                                row["new" if created else "merged"] += 1
                                if new_member:
                                    stats.new_paper_ids.add(p.id)
                                if use_yield and keyword_score(f"{rec.title} {rec.abstract or ''}",
                                                               flt.keywords, flt.keywords_needed)[0] >= flt.min_relevance:
                                    relevant += 1
                        prog.update(task, description=f"{src.name}: {q[:32]} ({row['found']} found)")
                        if use_yield and n:
                            low_pages = low_pages + 1 if relevant / n < stop.min_yield else 0
                            if low_pages >= stop.patience:
                                row["note"] = f"stopped: <{stop.min_yield:.0%} relevant for {stop.patience} pages"
                                break
                    if src.stop_reason:
                        row["note"] = src.stop_reason
                    elif share_limited and not row["note"] and row["found"] >= pair_cap:
                        row["note"] = f"share of total {total} ({pair_cap})"
                    elif explore and not row["note"] and row["found"] >= min(cap, getattr(src, "SEARCH_CAP", cap)):
                        row["note"] = f"hit cap ({min(cap, getattr(src, 'SEARCH_CAP', cap))})"
                except Exception as e:  # keep going with other sources/queries
                    msg = f"{src.name} / {q!r}: {type(e).__name__}: {e}"
                    log.warning("search failed: %s", msg)
                    stats.errors.append(msg)
                    row["error"] = str(e)[:80]
                row["total"] = getattr(src, "last_total", None)
                if run_id is not None and "error" not in row:  # checkpoint finished pair
                    with session(ctx) as s:
                        r = s.get(SyncRun, run_id)
                        r.details = {**(r.details or {}), "done": [*(r.details or {}).get("done", []), [src.name, q]]}
                stats.rows.append(row)
                prog.advance(task)
    stats.gathered = len(seen)
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
