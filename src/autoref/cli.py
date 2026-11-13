"""autoref command-line interface."""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sqlalchemy import func, select

from .config import load_config

app = typer.Typer(help="autoref — literature-review CLI: harvest, categorize, relate, export to Obsidian.",
                  no_args_is_help=True, pretty_exceptions_show_locals=False)
export_app = typer.Typer(help="Export the topic database.", no_args_is_help=True)
llm_app = typer.Typer(help="LLM backend utilities.", no_args_is_help=True)
sources_app = typer.Typer(help="Paper source utilities.", no_args_is_help=True)
schedule_app = typer.Typer(help="Periodic `autoref update` via systemd user timer.", no_args_is_help=True)
app.add_typer(export_app, name="export")
app.add_typer(llm_app, name="llm")
app.add_typer(sources_app, name="sources")
app.add_typer(schedule_app, name="schedule")

console = Console()
ConfigOpt = Annotated[Path, typer.Option("--config", "-c", envvar="AUTOREF_CONFIG", exists=True, dir_okay=False,
                                         help="Topic config YAML (or env AUTOREF_CONFIG).")]
DryRun = Annotated[bool, typer.Option("--dry-run", help="Do everything but do not commit DB changes.")]
Total = Annotated[Optional[int], typer.Option("--total", min=1, help="Stop after this many distinct papers "
                                              "were gathered in this run (overrides search.total).")]
Restart = Annotated[bool, typer.Option("--restart", help="Ignore checkpoints of an interrupted run; search everything again.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(message)s",
                        handlers=[RichHandler(console=console, show_path=False)])
    if not verbose:
        for noisy in ("httpx", "httpcore", "openai", "anthropic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _ctx(config: Path, dry_run: bool = False):
    from .pipeline.context import Ctx
    return Ctx(load_config(config), console=console, dry_run=dry_run)


def _dict_table(title: str, d: dict) -> Table:
    t = Table(title=title, show_header=False)
    for k, v in d.items():
        t.add_row(str(k), json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v))
    return t


def _search_table(rows: list[dict]) -> Table:
    t = Table(title="search results")
    for col in ("source", "query", "found", "new", "merged", "total", "note"):
        t.add_column(col)
    for r in rows:
        total = r.get("total")
        t.add_row(r["source"], r["query"], str(r["found"]), str(r["new"]), str(r["merged"]),
                  f"{total:,}" if isinstance(total, int) else "", r.get("error") or r.get("note", ""))
    return t


# ====================================================================== setup
@app.command()
def init(topic: str,
         from_template: Annotated[str, typer.Option(
             help="Packaged template (generic | mde | uav_metric) or path to any topic YAML to copy.")] = "generic",
         configs_dir: Annotated[Path, typer.Option(help="Where to write the config.")] = Path("configs")):
    """Create configs/<topic>.yaml from a template and an empty SQLite DB."""
    path = configs_dir / f"{topic}.yaml"
    if path.exists():
        console.print(f"[red]{path} already exists[/]")
        raise typer.Exit(1)
    src = Path(from_template).expanduser()
    if src.suffix in (".yaml", ".yml") and src.is_file():
        text = src.read_text(encoding="utf-8")
    else:
        tpl = resources.files("autoref.templates").joinpath(f"{from_template}.yaml")
        if not tpl.is_file():
            names = sorted(p.name.removesuffix(".yaml") for p in resources.files("autoref.templates").iterdir()
                           if p.name.endswith(".yaml"))
            console.print(f"[red]no template {from_template!r}[/]; use one of {', '.join(names)} or a path to a .yaml file")
            raise typer.Exit(1)
        text = tpl.read_text(encoding="utf-8")
    text = text.replace("__TOPIC__", topic)
    if from_template != "generic":
        import re
        text = re.sub(r"^name: .*$", f"name: {topic}", text, count=1, flags=re.M)
        text = re.sub(r"^db: .*$", f"db: data/{topic}.sqlite", text, count=1, flags=re.M)
        text = re.sub(r"^vault: .*$", f"vault: vault/{topic}", text, count=1, flags=re.M)
    configs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    ctx = _ctx(path)
    console.print(f"[green]created[/] {path}\n[green]database[/] {ctx.cfg.db_path}\n"
                  f"Edit queries/categories, then: autoref llm test -c {path}")


@llm_app.command("test")
def llm_test(config: ConfigOpt,
             task: Annotated[Optional[str], typer.Option(help="categorize | relations | fill | relevance")] = None):
    """Send a tiny prompt + JSON request to each configured backend; report latency and JSON support."""
    from pydantic import BaseModel
    from .llm import llm_for

    class Ping(BaseModel):
        answer: str
        number: int

    cfg = load_config(config)
    tasks = [task] if task else ["categorize", "relations", "fill", "relevance"]
    t = Table(title="LLM backends")
    for col in ("task", "provider:model", "text", "json", "latency"):
        t.add_column(col)
    seen = set()
    for tk in tasks:
        lc = cfg.llm.for_task(tk)
        key = (lc.provider, lc.model, lc.base_url)
        try:
            client = llm_for(cfg, tk)
            if key in seen:
                t.add_row(tk, client.model_id, "(same as above)", "", "")
                continue
            seen.add(key)
            t0 = time.time()
            txt = client.complete("Be terse.", "Reply with the single word: pong")
            js = client.complete_json("Be terse.", "Return answer='pong' and number=42.", Ping)
            t.add_row(tk, client.model_id, txt.strip()[:30], "ok" if js.number == 42 else f"wrong: {js}",
                      f"{time.time() - t0:.1f}s")
        except Exception as e:
            t.add_row(tk, f"{lc.provider}:{lc.model}", f"[red]{type(e).__name__}[/]", str(e)[:80], "")
    console.print(t)


@sources_app.command("test")
def sources_test(config: ConfigOpt, query: str = "monocular depth estimation"):
    """Query each configured source once; show results count, capabilities and API-key status."""
    cfg = load_config(config)
    from .sources import SourceSet
    t = Table(title="sources (priority order)")
    for col in ("source", "capabilities", "key/e-mail", "max rate", "budget", "result", "time"):
        t.add_column(col)
    for src in SourceSet(cfg.sources):
        sc = cfg.sources[src.name]
        key = {"semantic_scholar": bool(sc.api_key), "openalex": bool(sc.api_key)}.get(src.name)
        t0 = time.time()
        try:
            res = src.search(query, limit=3)
            r = f"{len(res)} hits: {res[0].title[:50] if res else '-'}"
        except Exception as e:
            r = f"[red]{type(e).__name__}: {str(e)[:60]}[/]"
        t.add_row(src.name, ", ".join(sorted(src.capabilities)), "n/a" if key is None else ("yes" if key else "no"),
                  f"{1 / src.min_delay:.2f}/s", src.budget_info or "", r, f"{time.time() - t0:.1f}s")
    console.print(t)


# ====================================================================== pipeline
@app.command()
def search(config: ConfigOpt,
           source: Annotated[Optional[str], typer.Option(help="Only this source.")] = None,
           query: Annotated[Optional[list[str]], typer.Option(help="Override config queries (repeatable).")] = None,
           limit: Annotated[Optional[int], typer.Option(help="Max results per query and source.")] = None,
           all_results: Annotated[bool, typer.Option("--all", help="Page until results stop being relevant "
                                                     "(search.stop) instead of a fixed limit.")] = False,
           no_filter: Annotated[bool, typer.Option("--no-filter", help="Skip relevance gate.")] = False,
           total: Total = None, restart: Restart = False, dry_run: DryRun = False):
    """Run config queries on sources, dedup/merge into DB, then apply the relevance gate."""
    from .pipeline.search import apply_filter, run_search
    from .pipeline.update import finish_run, resume_point, start_run
    ctx = _ctx(config, dry_run)
    full = not (source or query or dry_run)  # only full searches are checkpointed/resumed
    done = resume_point(ctx, "search") if full and not restart else []
    run_id = start_run(ctx, "search", details={"done": done}) if full else None
    if done:
        console.print(f"[bold]resuming[/]: {len(done)} source/query pairs already done")
    try:
        st = run_search(ctx, source=source, queries=query, limit=limit, exhaustive=all_results, total=total, run_id=run_id,
                        skip={tuple(x) for x in done})
    except BaseException as e:
        if run_id:
            finish_run(ctx, run_id, status="failed", error=str(e))
        raise
    console.print(_search_table(st.rows))
    console.print(f"gathered [bold]{st.gathered}[/] distinct papers")
    if run_id:
        finish_run(ctx, run_id, status="partial" if st.errors else "ok", n_new=len(st.new_paper_ids))
    if not no_filter:
        console.print(_dict_table("relevance gate", apply_filter(ctx)))
    ctx.close()


@app.command()
def refilter(config: ConfigOpt, dry_run: DryRun = False):
    """Re-score ALL non-manual topic papers with the current filter settings."""
    from .pipeline.search import apply_filter
    ctx = _ctx(config, dry_run)
    console.print(_dict_table("relevance gate", apply_filter(ctx, rescore=True)))


@app.command()
def snowball(config: ConfigOpt,
             depth: Annotated[Optional[int], typer.Option(help="Hops (default from config).")] = None,
             direction: Annotated[Optional[str], typer.Option(help="references | citations | both")] = None,
             max_per_paper: Annotated[Optional[int], typer.Option(
                 help="Neighbours promoted per paper (raise later to pull more from stored edges, no API calls).")] = None,
             refresh: Annotated[bool, typer.Option(help="Re-download edges even if already fetched.")] = False,
             dry_run: DryRun = False):
    """Fetch reference lists / citing papers of included papers; build the citation graph.

    Re-runnable after everything was downloaded: a larger --depth continues from papers stored at the
    previous depth; already-fetched papers cost no API calls."""
    from .pipeline.snowball import run_snowball
    ctx = _ctx(config, dry_run)
    console.print(_dict_table("snowball", run_snowball(ctx, depth=depth, direction=direction,
                                                       max_per_paper=max_per_paper, refresh=refresh)))
    ctx.close()


@app.command()
def enrich(config: ConfigOpt,
           fields: Annotated[Optional[str], typer.Option(help="Comma list, e.g. abstract,year,venue")] = None,
           no_llm: Annotated[bool, typer.Option("--no-llm", help="Sources/PDF only, no LLM.")] = False,
           force: Annotated[bool, typer.Option(help="Retry papers already enriched.")] = False,
           dry_run: DryRun = False):
    """Fill missing fields: other sources → PDF text → LLM (flagged in provenance)."""
    from .pipeline.enrich import run_enrich
    ctx = _ctx(config, dry_run)
    st = run_enrich(ctx, fields=fields.split(",") if fields else None, use_llm=False if no_llm else None, force=force)
    console.print(_dict_table("enrich", st))
    ctx.close()


@app.command()
def categorize(config: ConfigOpt,
               force: Annotated[bool, typer.Option(help="Re-categorize everything.")] = False,
               paper: Annotated[Optional[str], typer.Option(help="Only this paper (id/doi/arxiv/title).")] = None,
               limit: Annotated[Optional[int], typer.Option()] = None,
               dry_run: DryRun = False):
    """Assign config categories (tree) to included papers with an LLM.

    Processes papers never categorized or categorized with a different category tree (the tree's
    scheme id changes whenever you edit it, or set `categorize.scheme`). Resumable; --force redoes all."""
    from .pipeline.categorize import run_categorize
    ctx = _ctx(config, dry_run)
    ids = [_resolve(ctx, paper)] if paper else None
    console.print(_dict_table("categorize", run_categorize(ctx, force=force or bool(paper), paper_ids=ids, limit=limit)))


@app.command()
def relate(config: ConfigOpt,
           force: Annotated[bool, typer.Option(help="Re-label existing edges.")] = False,
           pair: Annotated[Optional[tuple[str, str]], typer.Option(help="CITING CITED (ids/doi/title).")] = None,
           limit: Annotated[Optional[int], typer.Option()] = None,
           dry_run: DryRun = False):
    """Label citation edges between included papers: label, concrete relation detail, difference.

    Processes unlabeled edges and edges labeled under a different relation-label set (stale)."""
    from .pipeline.relations import run_relations
    ctx = _ctx(config, dry_run)
    pr = (_resolve(ctx, pair[0]), _resolve(ctx, pair[1])) if pair else None
    console.print(_dict_table("relations", run_relations(ctx, force=force or bool(pair), pair=pr, limit=limit)))
    ctx.close()


@export_app.command("obsidian")
def export_obsidian(config: ConfigOpt,
                    vault: Annotated[Optional[Path], typer.Option(help="Override vault path.")] = None,
                    force: Annotated[bool, typer.Option(help="Rewrite all notes, not only changed.")] = False,
                    canvas: Annotated[Optional[list[str]], typer.Option(help="Category id to export as .canvas.")] = None):
    """Write/update the Obsidian vault (paper notes, categories, index, inbox, canvas)."""
    from .obsidian import export_vault
    ctx = _ctx(config)
    st = export_vault(ctx, vault=vault, force=force, canvas=canvas)
    console.print(_dict_table(f"export → {vault or ctx.cfg.vault_path}", st))


@app.command()
def run(config: ConfigOpt, no_export: Annotated[bool, typer.Option("--no-export")] = False,
        total: Total = None, restart: Restart = False):
    """Full pipeline: search → filter → snowball → enrich → categorize → relate → export. Resumable."""
    from .pipeline.update import run_all
    ctx = _ctx(config)
    out = run_all(ctx, export=not no_export, restart=restart, total=total)
    console.print(_search_table(out.pop("search")["rows"]))
    for k, v in out.items():
        console.print(_dict_table(k, v))
    ctx.close()


# ====================================================================== maintenance
@app.command()
def update(config: ConfigOpt,
           since: Annotated[Optional[str], typer.Option(help="YYYY-MM-DD; default = last successful run - overlap")] = None,
           no_export: Annotated[bool, typer.Option("--no-export")] = False, total: Total = None,
           restart: Restart = False):
    """Incremental: fetch papers published since the last run (+ new citations of core papers)."""
    from .pipeline.update import LockedError, run_update
    ctx = _ctx(config)
    try:
        out = run_update(ctx, since=date.fromisoformat(since) if since else None, export=not no_export,
                         restart=restart, total=total)
    except LockedError as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(2)
    console.print(_search_table(out.pop("search")))
    for k in ("search_errors", "from_citations", "filter", "snowball", "enrich", "categorize", "relations", "export"):
        if k in out and out[k]:
            v = out[k]
            console.print(_dict_table(k, v) if isinstance(v, dict) else f"{k}: {v}")
    console.print(f"[bold green]{out['n_new']} new papers[/], {out['new_included']} included — status {out['status']}")
    ctx.close()


@schedule_app.command("install")
def schedule_install(config: ConfigOpt, every: Annotated[str, typer.Option(help="e.g. 3d, 12h")] = "3d"):
    """Install a systemd user timer running `autoref update` periodically."""
    from . import schedule
    cfg = load_config(config)
    console.print(schedule.install(cfg.name, config, every))


@schedule_app.command("remove")
def schedule_remove(config: ConfigOpt):
    """Remove the timer for this topic."""
    from . import schedule
    console.print(schedule.remove(load_config(config).name))


@schedule_app.command("status")
def schedule_status(config: ConfigOpt):
    """Show the timer's next run and recent update runs."""
    from . import schedule
    from .db.schema import SyncRun
    from .pipeline.context import session
    ctx = _ctx(config)
    console.print(schedule.status(ctx.topic))
    t = Table(title="recent runs")
    for col in ("id", "kind", "window", "started", "status", "new", "error"):
        t.add_column(col)
    with session(ctx) as s:
        for r in s.scalars(select(SyncRun).where(SyncRun.topic == ctx.topic).order_by(SyncRun.id.desc()).limit(10)):
            t.add_row(str(r.id), r.kind, f"{r.window_from or ''} → {r.window_to or ''}",
                      f"{r.started_at:%Y-%m-%d %H:%M}", r.status, str(r.n_new), (r.error or "")[:50])
    console.print(t)


# ====================================================================== browsing
def _resolve(ctx, key: str) -> int:
    from .db import repo
    from .pipeline.context import session
    with session(ctx) as s:
        p = repo.find_by_key(s, key)
        if not p:
            console.print(f"[red]no paper matches {key!r}[/]")
            raise typer.Exit(1)
        return p.id


@app.command()
def stats(config: ConfigOpt):
    """Counts: papers by status, categories, relation labels, missing fields, LLM-filled fields."""
    from .db.schema import (CategoryAssignment, FieldProvenance, Paper, Reference, Relation, SyncRun,
                            TopicPaper)
    from .pipeline.context import session
    ctx = _ctx(config)
    with session(ctx) as s:
        st = dict(s.execute(select(TopicPaper.status, func.count()).where(TopicPaper.topic == ctx.topic)
                            .group_by(TopicPaper.status)).all())
        inc = ctx.included_ids(s)
        d = {
            "topic": ctx.topic, "db": str(ctx.cfg.db_path),
            "papers in DB (all topics, incl. stubs)": s.scalar(select(func.count()).select_from(Paper)),
            "stubs": s.scalar(select(func.count()).select_from(Paper).where(Paper.is_stub.is_(True))),
            "topic: included": st.get("included", 0), "topic: excluded": st.get("excluded", 0),
            "topic: candidate": st.get("candidate", 0),
            "core papers": s.scalar(select(func.count()).select_from(TopicPaper).where(
                TopicPaper.topic == ctx.topic, TopicPaper.core.is_(True))),
            "citation edges (all)": s.scalar(select(func.count()).select_from(Reference)),
            "included missing abstract": s.scalar(select(func.count()).select_from(Paper).where(
                Paper.id.in_(inc), Paper.abstract.is_(None))),
            "LLM-filled fields": s.scalar(select(func.count()).select_from(FieldProvenance).where(
                FieldProvenance.origin.like("llm%"), FieldProvenance.paper_id.in_(inc))),
            "uncategorized": s.scalar(select(func.count()).select_from(TopicPaper).where(
                TopicPaper.topic == ctx.topic, TopicPaper.status == "included", TopicPaper.categorized_at.is_(None))),
        }
        from .pipeline import categorize as _cat, relations as _rel
        d["category scheme (current)"] = ctx.cfg.categorize.scheme_id
        d["to (re)categorize"] = _cat.stale_count(ctx, s)
        d["relation scheme (current)"] = ctx.cfg.relations.scheme_id
        d["edges to (re)label"] = _rel.stale_count(ctx, s)
        d["depth levels"] = dict(s.execute(select(TopicPaper.depth, func.count()).where(
            TopicPaper.topic == ctx.topic, TopicPaper.status == "included").group_by(TopicPaper.depth)).all())
        last = s.scalar(select(SyncRun).where(SyncRun.topic == ctx.topic).order_by(SyncRun.id.desc()).limit(1))
        if last:
            d["last run"] = f"{last.kind} {last.started_at:%Y-%m-%d %H:%M} {last.status} (+{last.n_new})"
        console.print(_dict_table("stats", d))
        cat = Table(title="categories")
        cat.add_column("category")
        cat.add_column("papers", justify="right")
        for path, n in s.execute(select(CategoryAssignment.category_path, func.count()).where(
                CategoryAssignment.topic == ctx.topic).group_by(CategoryAssignment.category_path)
                .order_by(CategoryAssignment.category_path)).all():
            c = ctx.cfg.categorize.get(path.split("/")[-1])
            cat.add_row(f"{'  ' * path.count('/')}{c.label if c else path}", str(n))
        console.print(cat)
        rel = Table(title="relations")
        rel.add_column("label")
        rel.add_column("edges", justify="right")
        for lab, n in s.execute(select(Relation.label, func.count()).where(Relation.topic == ctx.topic)
                                .group_by(Relation.label)).all():
            rel.add_row(lab, str(n))
        console.print(rel)


@app.command("list")
def list_papers(config: ConfigOpt,
                category: Annotated[Optional[str], typer.Option(help="Category id (includes subcategories).")] = None,
                year_from: Annotated[Optional[int], typer.Option()] = None,
                source: Annotated[Optional[str], typer.Option()] = None,
                status: Annotated[str, typer.Option(help="included | excluded | candidate | all")] = "included",
                new: Annotated[bool, typer.Option("--new", help="Only papers within inbox_days.")] = False,
                sort: Annotated[str, typer.Option(help="citations | year | relevance | added")] = "year",
                limit: Annotated[int, typer.Option()] = 50):
    """Table of topic papers with filters."""
    from datetime import timedelta
    from .db.schema import CategoryAssignment, Paper, TopicPaper, utcnow
    from .pipeline.context import session
    ctx = _ctx(config)
    with session(ctx) as s:
        q = select(Paper, TopicPaper).join(TopicPaper, TopicPaper.paper_id == Paper.id).where(TopicPaper.topic == ctx.topic)
        if status != "all":
            q = q.where(TopicPaper.status == status)
        if year_from:
            q = q.where(Paper.year >= year_from)
        if source:
            q = q.where(Paper.sources.like(f"%{source}%"))
        if new:
            q = q.where(TopicPaper.added_at >= utcnow() - timedelta(days=ctx.cfg.export.inbox_days))
        if category:
            path = ctx.cfg.categorize.path_of(category)
            if not path:
                console.print(f"[red]unknown category {category}[/]")
                raise typer.Exit(1)
            sub = select(CategoryAssignment.paper_id).where(
                CategoryAssignment.topic == ctx.topic,
                (CategoryAssignment.category_path == path) | CategoryAssignment.category_path.like(f"{path}/%"))
            q = q.where(Paper.id.in_(sub))
        order = {"citations": Paper.citation_count.desc().nullslast(), "year": Paper.year.desc().nullslast(),
                 "relevance": TopicPaper.relevance.desc().nullslast(), "added": TopicPaper.added_at.desc()}[sort]
        rows = s.execute(q.order_by(order).limit(limit)).all()
        t = Table(title=f"{len(rows)} papers")
        for col in ("id", "year", "cit.", "rel.", "title", "first author", "origin"):
            t.add_column(col)
        for p, tp in rows:
            t.add_row(str(p.id), str(p.year or ""), str(p.citation_count or ""),
                      f"{tp.relevance:.2f}" if tp.relevance is not None else "", p.title[:80],
                      p.author_names[0] if p.authors else "", tp.origin)
        console.print(t)


@app.command()
def show(config: ConfigOpt, key: str):
    """Full record of a paper: metadata, provenance, categories, references, relations."""
    from .db import repo
    from .db.schema import CategoryAssignment, FieldProvenance, Paper, Reference, Relation, TopicPaper
    from .pipeline.context import session
    ctx = _ctx(config)
    with session(ctx) as s:
        p = repo.find_by_key(s, key)
        if not p:
            console.print(f"[red]no paper matches {key!r}[/]")
            raise typer.Exit(1)
        tp = s.get(TopicPaper, (ctx.topic, p.id))
        console.print(f"[bold]{p.title}[/]  (id {p.id}{', stub' if p.is_stub else ''})")
        console.print(", ".join(p.author_names))
        info = {"year": p.year, "venue": p.venue, "url": p.url, "pdf": p.pdf_url, "doi": p.doi, "arxiv": p.arxiv_id,
                "s2": p.s2_id, "openalex": p.openalex_id, "citations": p.citation_count, "sources": p.sources}
        if tp:
            info.update({"status": tp.status, "relevance": tp.relevance, "reason": tp.relevance_reason,
                         "origin": tp.origin, "depth": tp.depth, "core": tp.core, "added": tp.added_at})
        console.print(_dict_table("metadata", {k: v for k, v in info.items() if v not in (None, "")}))
        console.print(f"[bold]Abstract[/]: {p.abstract or '-'}\n")
        prov = {fp.field: f"{fp.origin} {fp.model or ''}" for fp in
                s.scalars(select(FieldProvenance).where(FieldProvenance.paper_id == p.id))}
        if prov:
            console.print(_dict_table("provenance", prov))
        for ca in s.scalars(select(CategoryAssignment).where(CategoryAssignment.paper_id == p.id,
                                                             CategoryAssignment.topic == ctx.topic)):
            console.print(f"category [cyan]{ca.category_path}[/] {ca.confidence:.2f} — {ca.rationale}")
        for r in s.scalars(select(Relation).where(Relation.topic == ctx.topic, Relation.citing_id == p.id)):
            console.print(f"→ [magenta]{r.label}[/] {s.get(Paper, r.cited_id).title[:70]}\n   {r.relation_detail}\n"
                          f"   diff: {r.difference_summary}")
        for r in s.scalars(select(Relation).where(Relation.topic == ctx.topic, Relation.cited_id == p.id)):
            console.print(f"← [magenta]{r.label}[/] {s.get(Paper, r.citing_id).title[:70]}\n   {r.relation_detail}")
        n_ref = s.scalar(select(func.count()).select_from(Reference).where(Reference.citing_id == p.id))
        n_cit = s.scalar(select(func.count()).select_from(Reference).where(Reference.cited_id == p.id))
        console.print(f"\nreferences stored: {n_ref}, cited-by stored: {n_cit}")


@app.command()
def mark(config: ConfigOpt, key: str,
         core: Annotated[bool, typer.Option("--core", help="Mark as core (tracked for new citations).")] = False,
         not_core: Annotated[bool, typer.Option("--not-core")] = False,
         include: Annotated[bool, typer.Option("--include")] = False,
         exclude: Annotated[bool, typer.Option("--exclude")] = False):
    """Manually include/exclude a paper or mark it as core. Manual status is never overridden."""
    from .db.schema import Paper, TopicPaper
    from .db import repo
    from .pipeline.context import session
    ctx = _ctx(config)
    pid = _resolve(ctx, key)
    with session(ctx) as s:
        p = s.get(Paper, pid)
        tp, _ = repo.membership(s, ctx.topic, p, origin="manual")
        if include or exclude:
            tp.status, tp.manual = ("included" if include else "excluded"), True
            if include:
                p.is_stub = False
        if core or not_core:
            tp.core = core
        console.print(f"{p.title[:80]} → status={tp.status} core={tp.core}")


def entry():
    """Console entry point: friendly messages for expected configuration errors."""
    from .llm import LLMError
    from .sources import SourceError
    try:
        app()
    except (LLMError, SourceError) as e:
        console.print(f"[red]error:[/] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    entry()
