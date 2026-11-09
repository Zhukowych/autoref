"""Interrupt the pipeline at various points and check that the next run resumes."""
import pytest

from autoref.db import repo
from autoref.db.schema import Paper, SyncRun, TopicPaper
from autoref.models import ExternalIds, PaperRecord, ReferenceRecord
from autoref.pipeline import snowball as snow_mod
from autoref.pipeline.context import session
from autoref.pipeline.update import process_new, run_all
from autoref.sources.base import PaperSource


class Boom(KeyboardInterrupt):
    pass


class FakeSearch(PaperSource):
    name = "fake"
    capabilities = frozenset({"search", "references"})
    min_delay = 0
    calls: list = []
    explode_on: str | None = None

    def search(self, query, limit=100, **kw):
        FakeSearch.calls.append(query)
        if query == FakeSearch.explode_on:
            raise Boom()
        return [PaperRecord(title=f"monocular depth uav {query} {i}", abstract="depth uav", year=2024,
                            ids=ExternalIds(doi=f"10.7/{query}/{i}"), source="fake") for i in range(2)]

    def references(self, paper, limit=1000):
        return [ReferenceRecord(cited=PaperRecord(title=f"ref depth uav of {paper.title}", abstract="depth uav",
                                                  ids=ExternalIds(doi=f"10.8/{paper.ids.doi}")))]


class Srcs:
    def __init__(self):
        self.src = FakeSearch()

    def with_cap(self, cap):
        return [self.src] if cap in self.src.capabilities else []

    def get(self, name):
        return self.src

    def close(self):
        pass


@pytest.fixture
def rctx(ctx):
    ctx._sources = Srcs()
    ctx.cfg.search.queries = ["q1", "q2", "q3"]
    ctx.cfg.search.filter.mode = "none"
    FakeSearch.calls = []
    FakeSearch.explode_on = None
    return ctx


def test_run_resumes_search_at_query_level(rctx):
    FakeSearch.explode_on = "q2"
    with pytest.raises(Boom):
        run_all(rctx, export=False)
    with session(rctx) as s:
        r = s.query(SyncRun).one()
        assert r.status == "failed" and r.details["done"] == [["fake", "q1"]]
    FakeSearch.explode_on = None
    FakeSearch.calls = []
    run_all(rctx, export=False)
    assert FakeSearch.calls == ["q2", "q3"]  # q1 not repeated
    with session(rctx) as s:
        assert s.query(TopicPaper).filter_by(status="included", depth=0).count() == 6
        last = s.query(SyncRun).order_by(SyncRun.id.desc()).first()
        assert last.status == "ok"
    FakeSearch.calls = []
    run_all(rctx, export=False)  # after success: fresh full search
    assert FakeSearch.calls == ["q1", "q2", "q3"]


def test_run_restart_ignores_checkpoint(rctx):
    FakeSearch.explode_on = "q2"
    with pytest.raises(Boom):
        run_all(rctx, export=False)
    FakeSearch.explode_on, FakeSearch.calls = None, []
    run_all(rctx, export=False, restart=True)
    assert FakeSearch.calls == ["q1", "q2", "q3"]


def test_snowball_scores_candidates_left_by_interrupted_run(rctx, monkeypatch):
    run_all(rctx, export=False)
    with session(rctx) as s:
        p, _ = repo.upsert_paper(s, PaperRecord(title="orphan depth uav candidate", ids=ExternalIds(doi="10.6/x")),
                                 stub=True)
        tp, _ = repo.membership(s, rctx.topic, p, origin="snowball", depth=1)  # stays "candidate"
        pid = p.id
    snow_mod.run_snowball(rctx)
    with session(rctx) as s:
        assert s.get(TopicPaper, (rctx.topic, pid)).status == "included"
        assert s.get(Paper, pid).is_stub is False


def test_update_expands_papers_included_but_never_snowballed(rctx):
    from autoref.pipeline.search import apply_filter, run_search
    run_search(rctx)
    apply_filter(rctx)  # ...then the run died before snowball
    with session(rctx) as s:
        assert s.query(Paper).filter(Paper.refs_fetched_at.isnot(None)).count() == 0
    process_new(rctx, set(), None)
    with session(rctx) as s:
        assert s.query(Paper).filter(Paper.refs_fetched_at.isnot(None)).count() == 6
