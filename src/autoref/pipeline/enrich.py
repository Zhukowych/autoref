"""Fill missing metadata: other sources -> PDF text -> LLM (from text only, flagged)."""
from __future__ import annotations

import io

import httpx
from pydantic import BaseModel
from sqlalchemy import select

from ..db import repo
from ..db.schema import Paper, utcnow
from ..llm import LLMError, llm_for
from ..sources.base import USER_AGENT
from .context import Ctx, log, render, session

MAX_PDF_BYTES = 25 * 1024 * 1024


def missing_fields(p: Paper, fields: list[str]) -> list[str]:
    out = []
    for f in fields:
        if f == "authors":
            if not p.authors:
                out.append(f)
        elif getattr(p, f, None) in (None, ""):
            out.append(f)
    return out


def pdf_text(url: str, pages: int = 3) -> str | None:
    try:
        with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            with c.stream("GET", url) as r:
                r.raise_for_status()
                if "pdf" not in r.headers.get("content-type", "pdf"):
                    return None
                buf = io.BytesIO()
                for chunk in r.iter_bytes():
                    buf.write(chunk)
                    if buf.tell() > MAX_PDF_BYTES:
                        return None
        from pypdf import PdfReader
        reader = PdfReader(buf)
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages[:pages])
        return text.strip() or None
    except Exception as e:
        log.info("pdf fetch failed %s: %s", url, e)
        return None


class FillOut(BaseModel):
    abstract: str | None = None
    abstract_is_summary: bool = False
    year: int | None = None
    venue: str | None = None
    authors: list[str] | None = None


def run_enrich(ctx: Ctx, *, fields: list[str] | None = None, use_llm: bool | None = None,
               paper_ids: list[int] | None = None, force: bool = False) -> dict:
    fm = ctx.cfg.fill_missing
    fields = fields or fm.fields
    use_llm = fm.allow_llm if use_llm is None else use_llm
    stats = {"checked": 0, "from_sources": 0, "from_llm": 0, "still_missing": 0}
    if not fm.enabled:
        return stats
    with session(ctx) as s:
        ids = ctx.included_ids(s, paper_ids)
        todo = [pid for pid in ids
                if (force or s.get(Paper, pid).enriched_at is None) and missing_fields(s.get(Paper, pid), fields)]
        # papers with nothing missing are simply marked done
        for pid in set(ids) - set(todo):
            p = s.get(Paper, pid)
            if p.enriched_at is None and not missing_fields(p, fields):
                p.enriched_at = utcnow()
    llm = llm_for(ctx.cfg, "fill") if use_llm else None
    if llm:
        sys_t, user_t = ctx.prompt("fill")
    with ctx.progress() as prog:
        task = prog.add_task("enrich", total=len(todo))
        for pid in todo:
            stats["checked"] += 1
            with session(ctx) as s:
                p = s.get(Paper, pid)
                rec = repo.to_record(p)
            # 1) other sources
            for src in ctx.sources.with_cap("get"):
                with session(ctx) as s:
                    if not missing_fields(s.get(Paper, pid), fields):
                        break
                try:
                    found = src.get(rec.ids, rec.title)
                except Exception as e:
                    log.info("%s.get failed: %s", src.name, e)
                    continue
                if found:
                    with session(ctx) as s:
                        p = s.get(Paper, pid)
                        if repo.merge_record(s, p, found):
                            stats["from_sources"] += 1
            # 2) PDF text -> LLM
            with session(ctx) as s:
                p = s.get(Paper, pid)
                still = missing_fields(p, fields)
                pdf_url = p.pdf_url or (f"https://arxiv.org/pdf/{p.arxiv_id}" if p.arxiv_id else None)
                title = p.title
            if still and llm and fm.allow_pdf and pdf_url:
                text = pdf_text(pdf_url)
                if text:
                    try:
                        out = llm.complete_json(render(sys_t), render(
                            user_t, title=title, fields=", ".join(still), text=text[:12000]), FillOut)
                    except LLMError as e:
                        log.warning("fill LLM failed for %s: %s", pid, e)
                        out = None
                    if out:
                        with session(ctx) as s:
                            p = s.get(Paper, pid)
                            for f in still:
                                val = getattr(out, f, None)
                                if val in (None, "", []):
                                    continue
                                if f == "authors":
                                    repo._set_authors(s, p, val)
                                else:
                                    setattr(p, f, val)
                                origin = "llm_summary" if f == "abstract" and out.abstract_is_summary else "llm"
                                repo.set_provenance(s, p, f, origin, llm.model_id)
                                stats["from_llm"] += 1
            with session(ctx) as s:
                p = s.get(Paper, pid)
                p.enriched_at = utcnow()
                if missing_fields(p, fields):
                    stats["still_missing"] += 1
            prog.advance(task)
    return stats
