"""SQLAlchemy ORM schema (SQLite)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    inspect, text,
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
    create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    norm_title: Mapped[str] = mapped_column(String, index=True)
    abstract: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    publication_date: Mapped[str | None] = mapped_column(String)
    venue: Mapped[str | None] = mapped_column(String)
    url: Mapped[str | None] = mapped_column(String)
    pdf_url: Mapped[str | None] = mapped_column(String)
    doi: Mapped[str | None] = mapped_column(String, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String, index=True)
    s2_id: Mapped[str | None] = mapped_column(String, index=True)
    openalex_id: Mapped[str | None] = mapped_column(String, index=True)
    citation_count: Mapped[int | None] = mapped_column(Integer)
    is_stub: Mapped[bool] = mapped_column(Boolean, default=True)
    sources: Mapped[str] = mapped_column(String, default="")  # comma-separated
    refs_fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    cites_fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    enriched_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    authors: Mapped[list["PaperAuthor"]] = relationship(
        back_populates="paper", order_by="PaperAuthor.position", cascade="all, delete-orphan")

    @property
    def author_names(self) -> list[str]:
        return [pa.author.name for pa in self.authors]


class Author(Base):
    __tablename__ = "authors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)


class PaperAuthor(Base):
    __tablename__ = "paper_authors"
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("authors.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    paper: Mapped[Paper] = relationship(back_populates="authors")
    author: Mapped[Author] = relationship(lazy="joined")


class Reference(Base):
    __tablename__ = "paper_references"
    __table_args__ = (UniqueConstraint("citing_id", "cited_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    citing_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    cited_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    raw_title: Mapped[str | None] = mapped_column(Text)
    contexts: Mapped[list] = mapped_column(JSON, default=list)
    intents: Mapped[list] = mapped_column(JSON, default=list)
    is_influential: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TopicPaper(Base):
    """Membership of a paper in a topic + per-topic state."""
    __tablename__ = "topic_papers"
    topic: Mapped[str] = mapped_column(String, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String, default="candidate")  # candidate|included|excluded
    relevance: Mapped[float | None] = mapped_column(Float)
    relevance_reason: Mapped[str | None] = mapped_column(Text)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    origin: Mapped[str] = mapped_column(String, default="search")  # search|snowball|update|citation|manual
    core: Mapped[bool] = mapped_column(Boolean, default=False)
    manual: Mapped[bool] = mapped_column(Boolean, default=False)  # status set by user, never overridden
    categorized_at: Mapped[datetime | None] = mapped_column(DateTime)
    categorized_scheme: Mapped[str | None] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    added_run_id: Mapped[int | None] = mapped_column(Integer)
    export_hash: Mapped[str | None] = mapped_column(String)
    note_name: Mapped[str | None] = mapped_column(String)


class CategoryAssignment(Base):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("topic", "paper_id", "category_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String, index=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    category_id: Mapped[str] = mapped_column(String)
    category_path: Mapped[str] = mapped_column(String)
    scheme: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (UniqueConstraint("topic", "citing_id", "cited_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String, index=True)
    citing_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    cited_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String)
    scheme: Mapped[str | None] = mapped_column(String)
    relation_detail: Mapped[str | None] = mapped_column(Text)
    difference_summary: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class FieldProvenance(Base):
    __tablename__ = "field_provenance"
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    field: Mapped[str] = mapped_column(String, primary_key=True)
    origin: Mapped[str] = mapped_column(String)  # semantic_scholar|openalex|arxiv|pdf|llm|manual
    model: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class SyncRun(Base):
    __tablename__ = "sync_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)  # search|update
    window_from: Mapped[str | None] = mapped_column(String)
    window_to: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String, default="running")  # running|ok|failed
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)


def make_session_factory(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _add_missing_columns(engine) -> None:
    """Tiny forward migration: add columns that exist in the models but not yet in an older DB."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing:
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {col.name} "
                                      f"{col.type.compile(engine.dialect)}"))
