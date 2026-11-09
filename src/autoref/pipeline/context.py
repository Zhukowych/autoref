"""Shared runtime context for pipeline steps + prompt templates."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from string import Template

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from sqlalchemy import select

from ..config import TopicConfig
from ..db.schema import Paper, TopicPaper, make_session_factory
from ..sources import SourceSet

log = logging.getLogger("autoref")


@dataclass
class Ctx:
    cfg: TopicConfig
    console: Console = field(default_factory=Console)
    dry_run: bool = False
    _sources: SourceSet | None = None

    def __post_init__(self):
        self.Session = make_session_factory(self.cfg.db_path)

    @property
    def topic(self) -> str:
        return self.cfg.name

    @property
    def sources(self) -> SourceSet:
        if self._sources is None:
            self._sources = SourceSet(self.cfg.sources, self.cfg.source_settings)
        return self._sources

    def close(self):
        if self._sources:
            self._sources.close()

    def progress(self) -> Progress:
        return Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), MofNCompleteColumn(),
                        TimeElapsedColumn(), console=self.console, transient=False)

    def prompt(self, name: str) -> tuple[Template, Template]:
        """Load prompt `name` (system, user). Project-level `prompts/<name>.txt` overrides package default."""
        override = self.cfg.base_dir / "prompts" / f"{name}.txt"
        text = override.read_text(encoding="utf-8") if override.exists() else \
            resources.files("autoref.prompts").joinpath(f"{name}.txt").read_text(encoding="utf-8")
        sys_part, _, user_part = text.partition("=== USER ===")
        sys_part = sys_part.replace("=== SYSTEM ===", "").strip()
        return Template(sys_part), Template(user_part.strip())

    def included_ids(self, s, paper_ids: list[int] | None = None) -> list[int]:
        q = select(TopicPaper.paper_id).where(TopicPaper.topic == self.topic, TopicPaper.status == "included")
        if paper_ids is not None:
            q = q.where(TopicPaper.paper_id.in_(paper_ids))
        return list(s.scalars(q))


def render(t: Template, **kw) -> str:
    return t.safe_substitute({k: ("" if v is None else v) for k, v in kw.items()})


def short(text: str | None, n: int = 3000) -> str:
    if not text:
        return "(not available)"
    return text if len(text) <= n else text[:n] + " …"


@contextmanager
def session(ctx: Ctx):
    s = ctx.Session()
    try:
        yield s
        if ctx.dry_run:
            s.rollback()
        else:
            s.commit()
    except BaseException:
        s.rollback()
        raise
    finally:
        s.close()


def paper_label(p: Paper) -> str:
    first = p.author_names[0].split()[-1] if p.authors else "Anon"
    return f"{first} {p.year or 'n.d.'} – {p.title[:80]}"
