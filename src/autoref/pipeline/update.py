"""Full pipeline run + incremental update of new papers since the last successful run."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from sqlalchemy import or_, select

from ..db.schema import Paper, SyncRun, TopicPaper, utcnow
from .categorize import run_categorize
from .context import Ctx, log, session
from .enrich import run_enrich
from .relations import run_relations
from .search import apply_filter, run_search
from .snowball import expand_paper, promote_included, run_snowball


class LockedError(RuntimeError):
    pass


@contextmanager
def topic_lock(ctx: Ctx):
    path = ctx.cfg.db_path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockedError(f"another autoref run holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def start_run(ctx: Ctx, kind: str, window_from: date | None = None, window_to: date | None = None,
              details: dict | None = None) -> int:
    with session(ctx) as s:
        r = SyncRun(topic=ctx.topic, kind=kind, window_from=window_from.isoformat() if window_from else None,
                    window_to=window_to.isoformat() if window_to else None, details=details or {})
        s.add(r)
        s.flush()
        return r.id


def finish_run(ctx: Ctx, run_id: int, *, status: str, n_new: int = 0, details: dict | None = None,
               error: str | None = None):
    with session(ctx) as s:
        r = s.get(SyncRun, run_id)
        merged = {**(r.details or {}), **(details or {})}  # keep "done" checkpoints
        r.status, r.n_new, r.details, r.error = status, n_new, merged, error
        r.finished_at = utcnow()


def resume_point(ctx: Ctx, kind: str, window_to: date | None = None) -> list:
    """(source, query) pairs finished by the latest run of `kind` if that run was interrupted
    (failed/running). For updates the window must match, otherwise nothing is skipped."""
    with session(ctx) as s:
        r = s.scalar(select(SyncRun).where(SyncRun.topic == ctx.topic, SyncRun.kind == kind)
                     .order_by(SyncRun.id.desc()).limit(1))
        if r is None or r.status not in ("failed", "running"):
            return []
        if window_to is not None and r.window_to != window_to.isoformat():
            return []
        return [list(x) for x in (r.details or {}).get("done", [])]


def last_success(ctx: Ctx, s) -> SyncRun | None:
    return s.scalar(select(SyncRun).where(SyncRun.topic == ctx.topic, SyncRun.status == "ok",
                                          SyncRun.kind.in_(["search", "update"]))
                    .order_by(SyncRun.started_at.desc()).limit(1))


def compute_window(ctx: Ctx, since: date | None = None, today: date | None = None) -> tuple[date, date]:
    """[last successful run start - overlap, today]. Failed runs never advance the window."""
    today = today or utcnow().date()
    if since:
        return since, today
    with session(ctx) as s:
        last = last_success(ctx, s)
    if last is None:
        return today - timedelta(days=ctx.cfg.update.default_since_days), today
    return last.started_at.date() - timedelta(days=ctx.cfg.update.overlap_days), today


def core_paper_ids(ctx: Ctx, s) -> list[int]:
    u = ctx.cfg.update
    return list(s.scalars(select(TopicPaper.paper_id).where(
        TopicPaper.topic == ctx.topic, TopicPaper.status == "included",
        or_(TopicPaper.core.is_(True), TopicPaper.relevance >= u.core_min_relevance))))


def process_new(ctx: Ctx, new_ids: set[int], run_id: int | None) -> dict:
    """Gate + enrich + snowball + categorize + relate only for `new_ids` (and their new neighbours)."""
    out: dict = {}
    sc = ctx.cfg.snowball
    with session(ctx) as s:  # leftovers from an interrupted earlier run
        new_ids = set(new_ids) | set(s.scalars(select(TopicPaper.paper_id).where(
            TopicPaper.topic == ctx.topic, TopicPaper.status == "candidate")))
        # included earlier but never expanded (run stopped before snowball)
        not_fetched = []
        if sc.direction in ("references", "both"):
            not_fetched.append(Paper.refs_fetched_at.is_(None))
        if sc.direction in ("citations", "both"):
            not_fetched.append(Paper.cites_fetched_at.is_(None))
        new_ids |= set(s.scalars(select(TopicPaper.paper_id).join(Paper, Paper.id == TopicPaper.paper_id).where(
            TopicPaper.topic == ctx.topic, TopicPaper.status == "included", TopicPaper.depth < sc.depth,
            or_(*not_fetched))))
    out["filter"] = apply_filter(ctx, list(new_ids))
    promote_included(ctx, new_ids)
    out["snowball"] = run_snowball(ctx, paper_ids=list(new_ids), origin="snowball", run_id=run_id)
    out["enrich"] = run_enrich(ctx)        # only not-yet-enriched papers
    out["categorize"] = run_categorize(ctx)  # only not-yet-categorized papers
    out["relations"] = run_relations(ctx)    # only unlabeled edges
    return out


def run_update(ctx: Ctx, *, since: date | None = None, export: bool = True, restart: bool = False) -> dict:
    with topic_lock(ctx):
        lo, hi = compute_window(ctx, since)
        done = resume_point(ctx, "update", window_to=hi) if since is None and not restart else []
        run_id = start_run(ctx, "update", lo, hi, details={"done": done})
        try:
            ctx.console.print(f"[bold]update window[/] {lo} → {hi}"
                              + (f" (resuming, {len(done)} queries already done)" if done else ""))
            stats = run_search(ctx, date_from=lo, date_to=hi, origin="update", run_id=run_id,
                               skip={tuple(x) for x in done})
            new_ids = set(stats.new_paper_ids)
            n_cit = 0
            if ctx.cfg.update.track_citations_of_core:
                with session(ctx) as s:
                    core = core_paper_ids(ctx, s)
                with ctx.progress() as prog:
                    task = prog.add_task("new citations of core papers", total=len(core))
                    for pid in core:
                        found = expand_paper(ctx, pid, 0, "citations", ctx.cfg.snowball.max_per_paper,
                                             date_from=lo, origin="citation", run_id=run_id)
                        n_cit += len(found)
                        new_ids |= found
                        prog.advance(task)
            details = {"search": stats.rows, "search_errors": stats.errors, "from_citations": n_cit}
            details.update(process_new(ctx, new_ids, run_id))
            if export:
                from ..obsidian import export_vault
                details["export"] = export_vault(ctx)
            with session(ctx) as s:
                n_inc = len(ctx.included_ids(s, list(new_ids)))
            details["new_included"] = n_inc
            status = "failed" if stats.errors and not stats.rows else ("partial" if stats.errors else "ok")
            finish_run(ctx, run_id, status=status, n_new=len(new_ids), details=details,
                       error="; ".join(stats.errors)[:2000] or None)
            details["status"], details["n_new"] = status, len(new_ids)
            return details
        except BaseException as e:
            finish_run(ctx, run_id, status="failed", error=f"{type(e).__name__}: {e}")
            raise


def run_all(ctx: Ctx, *, export: bool = True, restart: bool = False) -> dict:
    """Full pipeline: search → filter → snowball → enrich → categorize → relate → export."""
    with topic_lock(ctx):
        done = [] if restart else resume_point(ctx, "search")
        run_id = start_run(ctx, "search", details={"done": done})
        try:
            out: dict = {}
            if done:
                ctx.console.print(f"[bold]resuming[/]: {len(done)} source/query pairs already done")
            st = run_search(ctx, run_id=run_id, skip={tuple(x) for x in done})
            out["search"] = {"rows": st.rows, "new": len(st.new_paper_ids), "errors": st.errors}
            out["filter"] = apply_filter(ctx)
            out["snowball"] = run_snowball(ctx)
            out["enrich"] = run_enrich(ctx)
            out["categorize"] = run_categorize(ctx)
            out["relations"] = run_relations(ctx)
            if export:
                from ..obsidian import export_vault
                out["export"] = export_vault(ctx)
            status = "partial" if st.errors else "ok"
            finish_run(ctx, run_id, status=status, n_new=len(st.new_paper_ids), details={"errors": st.errors},
                       error="; ".join(st.errors)[:2000] or None)
            return out
        except BaseException as e:
            finish_run(ctx, run_id, status="failed", error=f"{type(e).__name__}: {e}")
            raise
