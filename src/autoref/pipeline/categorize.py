"""LLM categorization into the config's category tree."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model
from sqlalchemy import delete, or_, select

from ..config import iter_categories
from ..db.schema import CategoryAssignment, Paper, TopicPaper, utcnow
from ..llm import FailureGuard, LLMError, llm_for
from .context import Ctx, log, render, session, short


def build_schema(cat_ids: list[str]) -> type[BaseModel]:
    CatId = Literal[tuple(cat_ids)]  # type: ignore[valid-type]
    Item = create_model("CategoryChoice", id=(CatId, ...), confidence=(float, Field(0.8, ge=0, le=1)))
    return create_model("CategorizeOut", categories=(list[Item], ...), rationale=(str, ""))


def categories_block(cfg) -> str:
    lines = []
    for path, c in iter_categories(cfg.categorize.categories):
        indent = "  " * path.count("/")
        lines.append(f"{indent}- {c.id} | {c.label} | {c.description.strip()}")
    return "\n".join(lines)


def run_categorize(ctx: Ctx, *, force: bool = False, paper_ids: list[int] | None = None,
                   limit: int | None = None) -> dict:
    cc = ctx.cfg.categorize
    cat_ids = [c.id for _, c in iter_categories(cc.categories)]
    scheme = cc.scheme_id
    Schema = build_schema(cat_ids)
    llm = llm_for(ctx.cfg, "categorize")
    sys_t, user_t = ctx.prompt("categorize")
    system = render(sys_t, mode=cc.mode, categories=categories_block(ctx.cfg))
    with session(ctx) as s:
        q = select(TopicPaper.paper_id).where(TopicPaper.topic == ctx.topic, TopicPaper.status == "included")
        if not force:  # never categorized, or categorized with a different category tree (stale)
            q = q.where(or_(TopicPaper.categorized_scheme.is_(None), TopicPaper.categorized_scheme != scheme))
        if paper_ids is not None:
            q = q.where(TopicPaper.paper_id.in_(paper_ids))
        todo = list(s.scalars(q))[:limit] if limit else list(s.scalars(q))
    stats = {"categorized": 0, "failed": 0, "uncategorized": 0}
    guard = FailureGuard("categorize")
    with ctx.progress() as prog:
        task = prog.add_task(f"categorize ({llm.model_id})", total=len(todo))
        for pid in todo:
            with session(ctx) as s:
                p = s.get(Paper, pid)
                user = render(user_t, title=p.title, year=p.year, venue=p.venue, abstract=short(p.abstract, 4000))
            try:
                out = llm.complete_json(system, user, Schema)
            except LLMError as e:
                log.warning("categorize failed for %s: %s", pid, e)
                guard.failed(e)
                stats["failed"] += 1
                prog.advance(task)
                continue
            guard.ok()
            chosen = sorted(out.categories, key=lambda c: c.confidence, reverse=True)
            chosen = [c for c in chosen if c.confidence >= cc.min_confidence]
            if cc.mode == "single":
                chosen = chosen[:1]
            with session(ctx) as s:
                s.execute(delete(CategoryAssignment).where(
                    CategoryAssignment.topic == ctx.topic, CategoryAssignment.paper_id == pid))
                seen = set()
                for c in chosen:
                    if c.id in seen:
                        continue
                    seen.add(c.id)
                    s.add(CategoryAssignment(topic=ctx.topic, paper_id=pid, category_id=c.id,
                                             category_path=cc.path_of(c.id), confidence=c.confidence,
                                             rationale=out.rationale[:1000], model=llm.model_id, scheme=scheme))
                tp = s.get(TopicPaper, (ctx.topic, pid))
                tp.categorized_at, tp.categorized_scheme = utcnow(), scheme
            stats["categorized" if chosen else "uncategorized"] += 1
            prog.advance(task)
    return stats


def stale_count(ctx: Ctx, s) -> int:
    from sqlalchemy import func
    scheme = ctx.cfg.categorize.scheme_id
    return s.scalar(select(func.count()).select_from(TopicPaper).where(
        TopicPaper.topic == ctx.topic, TopicPaper.status == "included",
        or_(TopicPaper.categorized_scheme.is_(None), TopicPaper.categorized_scheme != scheme)))
