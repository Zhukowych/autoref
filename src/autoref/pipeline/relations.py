"""Label citation edges between included papers (citation contexts + abstracts -> LLM)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model
from sqlalchemy import select
from sqlalchemy.orm import aliased

from ..db import repo
from ..db.schema import Paper, Reference, Relation, TopicPaper
from ..llm import FailureGuard, LLMError, llm_for
from ..models import CitationContext
from .context import Ctx, log, render, session, short


def build_schema(labels: list[str]) -> type[BaseModel]:
    Label = Literal[tuple(labels)]  # type: ignore[valid-type]
    return create_model(
        "RelationOut",
        label=(Label, ...),
        relation_detail=(str, ...),
        difference_summary=(str, ...),
        evidence=(list[str], Field(default_factory=list)),
        confidence=(float, Field(0.5, ge=0, le=1)),
    )


def labels_block(cfg) -> str:
    return "\n".join(f"- {l.relation_label} ({l.label}): {l.description.strip()}" for l in cfg.relations.labels)


def candidate_edges(ctx: Ctx, s, *, force: bool, pair: tuple[int, int] | None = None) -> list[int]:
    a, b = aliased(TopicPaper), aliased(TopicPaper)
    q = (select(Reference.id)
         .join(a, (a.paper_id == Reference.citing_id) & (a.topic == ctx.topic) & (a.status == "included"))
         .join(b, (b.paper_id == Reference.cited_id) & (b.topic == ctx.topic) & (b.status == "included")))
    if pair:
        q = q.where(Reference.citing_id == pair[0], Reference.cited_id == pair[1])
    q = q.add_columns(Reference.citing_id, Reference.cited_id)
    rows = s.execute(q).all()
    if force:
        return [r[0] for r in rows]
    # done = labeled with the CURRENT label set; edges labeled under an older set are stale -> redone
    done = set(s.execute(select(Relation.citing_id, Relation.cited_id).where(
        Relation.topic == ctx.topic, Relation.scheme == ctx.cfg.relations.scheme_id)).all())
    return [rid for rid, a_id, b_id in rows if (a_id, b_id) not in done]


def _fill_contexts(ctx: Ctx, ref_ids: list[int]) -> None:
    """Edges found via OpenAlex lack citation sentences; fetch them from S2 (one call per citing paper)."""
    s2 = ctx.sources.get("semantic_scholar")
    if not s2:
        return
    with session(ctx) as s:
        by_citing: dict[int, list[int]] = {}
        for rid in ref_ids:
            r = s.get(Reference, rid)
            if not r.contexts and r.source != "semantic_scholar":
                by_citing.setdefault(r.citing_id, []).append(rid)
    for citing_id, rids in by_citing.items():
        with session(ctx) as s:
            rec = repo.to_record(s.get(Paper, citing_id))
        try:
            edges = s2.references(rec)
        except Exception as e:
            log.info("S2 contexts fetch failed for %s: %s", citing_id, e)
            continue
        with session(ctx) as s:
            citing = s.get(Paper, citing_id)
            for e in edges:
                if e.contexts or e.intents:
                    repo.add_reference(s, citing, e, "semantic_scholar")


def run_relations(ctx: Ctx, *, force: bool = False, pair: tuple[int, int] | None = None,
                  limit: int | None = None) -> dict:
    rc = ctx.cfg.relations
    Schema = build_schema([l.relation_label for l in rc.labels])
    llm = llm_for(ctx.cfg, "relations")
    sys_t, user_t = ctx.prompt("relations")
    system = render(sys_t, labels=labels_block(ctx.cfg))
    with session(ctx) as s:
        todo = candidate_edges(ctx, s, force=force, pair=pair)
    if limit:
        todo = todo[:limit]
    _fill_contexts(ctx, todo)
    stats = {"labeled": 0, "failed": 0}
    guard = FailureGuard("relations")
    with ctx.progress() as prog:
        task = prog.add_task(f"relations ({llm.model_id})", total=len(todo))
        for rid in todo:
            with session(ctx) as s:
                r = s.get(Reference, rid)
                citing, cited = s.get(Paper, r.citing_id), s.get(Paper, r.cited_id)
                contexts = [CitationContext.model_validate(c).text for c in (r.contexts or [])]
                user = render(
                    user_t,
                    citing_title=citing.title, citing_year=citing.year, citing_abstract=short(citing.abstract, 2500),
                    cited_title=cited.title, cited_year=cited.year, cited_abstract=short(cited.abstract, 2500),
                    intents=", ".join(r.intents or []) or "(none)",
                    contexts="\n".join(f"- {c}" for c in contexts[:12]) or "(no citation sentences available)",
                )
                key = (r.citing_id, r.cited_id)
            try:
                out = llm.complete_json(system, user, Schema)
            except LLMError as e:
                log.warning("relation failed for %s: %s", key, e)
                guard.failed(e)
                stats["failed"] += 1
                prog.advance(task)
                continue
            guard.ok()
            with session(ctx) as s:
                rel = s.scalar(select(Relation).where(Relation.topic == ctx.topic, Relation.citing_id == key[0],
                                                      Relation.cited_id == key[1]))
                if rel is None:
                    rel = Relation(topic=ctx.topic, citing_id=key[0], cited_id=key[1])
                    s.add(rel)
                rel.label, rel.relation_detail, rel.scheme = out.label, out.relation_detail, rc.scheme_id
                rel.difference_summary, rel.evidence = out.difference_summary, out.evidence
                rel.confidence, rel.model = out.confidence, llm.model_id
            stats["labeled"] += 1
            prog.advance(task)
    return stats


def stale_count(ctx: Ctx, s) -> int:
    return len(candidate_edges(ctx, s, force=False))
