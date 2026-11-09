"""`limit_per_query: all` — page until results stop being relevant."""
import httpx
import pytest
import respx
import yaml

from autoref.config import SourceConfig, TopicConfig
from autoref.db.schema import SyncRun, TopicPaper
from autoref.models import ExternalIds, PaperRecord
from autoref.pipeline.context import session
from autoref.pipeline.search import run_search
from autoref.sources.arxiv import ArxivSource
from autoref.sources.base import PaperSource
from autoref.sources.openalex import OpenAlexSource
from autoref.sources.semantic_scholar import SemanticScholarSource

REL, JUNK = True, False


class PagedSource(PaperSource):
    """Serves pages described by `plan` (True = relevant record, False = unrelated)."""
    name = "paged"
    capabilities = frozenset({"search"})
    min_delay = 0
    plan: list = []
    plans: dict = {}          # per-query override of `plan`
    served = 0
    last_total = 777

    def search_pages(self, query, page_size=100, max_results=None, **kw):
        got = 0
        for flags in self.plans.get(query, self.plan):
            if max_results is not None and got >= max_results:
                return
            n = min(len(flags), page_size, (max_results - got) if max_results is not None else 10 ** 9)
            recs = [PaperRecord(title=(f"monocular depth uav study {query} {got + i}" if rel else f"gardening tips volume {query} {got + i}"),
                                ids=ExternalIds(doi=f"10.1/{query}/{got + i}"), year=2024, source="paged")
                    for i, rel in enumerate(flags[:n])]
            PagedSource.served += 1
            got += n
            yield recs


class Srcs:
    def __init__(self):
        self.src = PagedSource()

    def with_cap(self, cap):
        return [self.src]

    def get(self, name):
        return self.src

    def close(self):
        pass


@pytest.fixture
def sctx(ctx):
    ctx._sources = Srcs()
    ctx.cfg.search.queries = ["q"]
    ctx.cfg.search.stop.page_size = 50
    ctx.cfg.search.stop.min_yield = 0.2
    ctx.cfg.search.stop.patience = 2
    PagedSource.served = 0
    PagedSource.plans = {}
    return ctx


def page(n_rel, n=50):
    return [REL] * n_rel + [JUNK] * (n - n_rel)


def test_stops_after_patience_low_yield_pages(sctx):
    PagedSource.plan = [page(50), page(40), page(5), page(3), page(50)]  # 100%, 80%, 10%, 6% -> stop
    st = run_search(sctx, exhaustive=True)
    [row] = st.rows
    assert PagedSource.served == 4 and row["found"] == 200 and row["pages"] == 4
    assert row["note"].startswith("stopped: <20% relevant for 2 pages") and row["total"] == 777


def test_one_low_page_does_not_stop(sctx):
    PagedSource.plan = [page(50), page(2), page(45), page(1), page(40)]  # low pages never consecutive
    st = run_search(sctx, exhaustive=True)
    assert PagedSource.served == 5 and st.rows[0]["found"] == 250 and st.rows[0]["note"] == ""


def test_hard_cap(sctx):
    sctx.cfg.search.stop.max_results = 120
    PagedSource.plan = [page(50)] * 10
    st = run_search(sctx, exhaustive=True)
    assert st.rows[0]["found"] == 120 and st.rows[0]["note"] == "hit cap (120)"


def test_numeric_limit_is_exact_and_ignores_yield(sctx):
    sctx.cfg.search.limit_per_query = 60
    PagedSource.plan = [page(0), page(0), page(0)]       # all junk, would stop in explore mode
    st = run_search(sctx)
    assert st.rows[0]["found"] == 60 and st.rows[0]["note"] == ""
    st = run_search(sctx, limit=10)                       # explicit --limit beats the config
    assert st.rows[0]["found"] == 10


def test_all_in_config_means_explore(sctx):
    sctx.cfg.search.limit_per_query = None
    PagedSource.plan = [page(50), page(1), page(1), page(50)]
    st = run_search(sctx)
    assert st.rows[0]["found"] == 150 and st.rows[0]["note"].startswith("stopped")


def test_all_string_is_accepted_in_yaml(cfg_path):
    raw = yaml.safe_load(cfg_path.read_text())
    for v in ("all", "ALL", None):
        raw["search"]["limit_per_query"] = v
        assert TopicConfig.model_validate(raw).search.limit_per_query is None
    raw["search"]["limit_per_query"] = 25
    assert TopicConfig.model_validate(raw).search.limit_per_query == 25


def test_no_keywords_means_only_cap_applies(sctx):
    sctx.cfg.search.filter.keywords = []
    sctx.cfg.search.stop.max_results = 100
    PagedSource.plan = [page(0)] * 5
    st = run_search(sctx, exhaustive=True)
    assert st.rows[0]["found"] == 100 and st.rows[0]["note"].startswith("hit cap")


def test_pages_are_committed_and_pair_is_checkpointed(sctx):
    PagedSource.plan = [page(50), page(50), page(0), page(0)]
    with session(sctx) as s:
        r = SyncRun(topic=sctx.topic, kind="search")
        s.add(r)
        s.flush()
        rid = r.id
    run_search(sctx, exhaustive=True, run_id=rid)
    with session(sctx) as s:
        assert s.get(SyncRun, rid).details["done"] == [["paged", "q"]]
        assert s.query(TopicPaper).count() == 200


def test_error_midway_keeps_earlier_pages_and_is_not_checkpointed(sctx):
    class Failing(PagedSource):
        def search_pages(self, query, **kw):
            yield from list(PagedSource.search_pages(self, query, **kw))[:1]
            raise RuntimeError("boom")
    sctx._sources.src = Failing()
    PagedSource.plan = [page(50), page(50)]
    with session(sctx) as s:
        r = SyncRun(topic=sctx.topic, kind="search")
        s.add(r)
        s.flush()
        rid = r.id
    st = run_search(sctx, exhaustive=True, run_id=rid)
    assert st.rows[0]["found"] == 50 and "boom" in st.rows[0]["error"] and len(st.errors) == 1
    with session(sctx) as s:
        assert not s.get(SyncRun, rid).details.get("done")  # will be retried on resume


# ------------------------------------------------------------------ search.total
def test_total_is_split_fairly_over_queries(sctx):
    sctx.cfg.search.queries = ["q1", "q2", "q3"]
    sctx.cfg.search.limit_per_query = 100
    PagedSource.plan = [page(50)] * 5
    st = run_search(sctx, total=30)
    assert [r["found"] for r in st.rows] == [10, 10, 10] and st.gathered == 30
    assert all(r["note"] == "share of total 30 (10)" for r in st.rows)


def test_unused_share_rolls_over_to_later_queries(sctx):
    sctx.cfg.search.queries = ["q1", "q2", "q3"]
    sctx.cfg.search.limit_per_query = 100
    PagedSource.plan = [page(50)] * 5
    PagedSource.plans = {"q1": [page(4, n=4)]}            # q1 only has 4 results
    st = run_search(sctx, total=30)
    assert [r["found"] for r in st.rows] == [4, 13, 13] and st.gathered == 30


def test_pairs_after_total_are_skipped(sctx):
    sctx.cfg.search.queries = ["q1", "q2", "q3"]
    sctx.cfg.search.limit_per_query = 100
    PagedSource.plan = [page(50)]
    st = run_search(sctx, total=2)
    assert [r["found"] for r in st.rows] == [1, 1, 0] and st.rows[2]["note"] == "skipped: total of 2 papers reached"
    assert st.gathered == 2 and PagedSource.served == 2    # the third query was never sent


def test_total_counts_known_papers_so_reruns_stay_short(sctx):
    sctx.cfg.search.queries = ["q1", "q2"]
    sctx.cfg.search.limit_per_query = 100
    PagedSource.plan = [page(50)] * 3
    first = run_search(sctx, total=20)
    again = run_search(sctx, total=20)                     # everything already in the DB: all "merged"
    assert first.gathered == again.gathered == 20
    assert sum(r["new"] for r in again.rows) == 0 and sum(r["merged"] for r in again.rows) == 20


def test_total_from_config_and_explicit_override(sctx):
    sctx.cfg.search.queries = ["q1", "q2"]
    sctx.cfg.search.limit_per_query = 100
    sctx.cfg.search.total = 8
    PagedSource.plan = [page(50)] * 3
    assert run_search(sctx).gathered == 8
    assert run_search(sctx, total=4).gathered == 4        # CLI --total beats the config


def test_total_works_with_explore_mode_and_skipped_pairs(sctx):
    sctx.cfg.search.queries = ["q1", "q2"]
    sctx.cfg.search.limit_per_query = None
    PagedSource.plan = [page(50)] * 5
    st = run_search(sctx, total=20, skip={("paged", "q1")})   # q1 done by an interrupted run -> q2 gets it all
    assert [r["found"] for r in st.rows] == [0, 20] and st.gathered == 20


def test_total_validation(cfg_path):
    raw = yaml.safe_load(cfg_path.read_text())
    raw["search"]["total"] = 0
    with pytest.raises(Exception):
        TopicConfig.model_validate(raw)
    raw["search"]["total"] = 150
    assert TopicConfig.model_validate(raw).search.total == 150


# ------------------------------------------------------------------ real sources
def oa_page(ids, cursor):
    return httpx.Response(200, json={
        "meta": {"count": 12345, "next_cursor": cursor},
        "results": [{"id": f"https://openalex.org/W{i}", "title": f"Paper number {i} about depth",
                     "publication_year": 2023} for i in ids]})


def oa_src(**cfg):
    src = OpenAlexSource(SourceConfig(**cfg) if cfg else None)
    src.min_delay = src._delay = 0
    return src


@respx.mock
def test_openalex_with_key_uses_cursor_paging_and_reports_total():
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[
        oa_page(range(0, 2), "c2"), oa_page(range(2, 4), "c3"), oa_page([], None)])
    src = oa_src(api_key="K")
    pages = list(src.search_pages("depth", page_size=2))
    assert [len(p) for p in pages] == [2, 2] and src.last_total == 12345
    q = route.calls[1].request.url.params
    assert q["cursor"] == "c2" and q["api_key"] == "K" and "page" not in q


@respx.mock
def test_openalex_without_key_uses_page_paging_not_cursor():
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[
        oa_page(range(0, 2), "ignored"), oa_page(range(2, 4), "ignored"), oa_page([5], "ignored")])
    src = oa_src()
    pages = list(src.search_pages("depth", page_size=2))
    assert [len(p) for p in pages] == [2, 2, 1]                       # short page ends it
    params = [c.request.url.params for c in route.calls]
    assert [p["page"] for p in params] == ["1", "2", "3"]
    assert all("cursor" not in p and "api_key" not in p and p["per-page"] == "2" for p in params)


@respx.mock
def test_openalex_page_mode_keeps_page_size_constant_and_trims_to_cap():
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[
        oa_page(range(0, 4), "x"), oa_page(range(4, 8), "x")])
    src = oa_src()
    out = [r for pg in src.search_pages("depth", page_size=4, max_results=6) for r in pg]
    assert len(out) == 6                                              # 4 + 2, trimmed
    assert [c.request.url.params["per-page"] for c in route.calls] == ["4", "4"]  # offsets stay aligned


@respx.mock
def test_openalex_stops_when_daily_budget_is_used_up():
    low = {"x-ratelimit-remaining": "5", "x-ratelimit-limit": "1000", "x-ratelimit-reset": "7200",
           "x-ratelimit-credits-used": "10"}
    resp = oa_page(range(0, 2), "c2")
    resp.headers.update(low)
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[resp, oa_page(range(2, 4), "c3")])
    src = oa_src()
    pages = list(src.search_pages("depth", page_size=2))
    assert len(pages) == 1 and route.call_count == 1                 # 5 credits left < 10 per request
    assert "daily budget used up" in src.stop_reason and src.budget_info == "5/1000 credits, resets in 2h"


@respx.mock
def test_openalex_search_wrapper_trims_and_limits_page_size():
    route = respx.get("https://api.openalex.org/works").mock(side_effect=[oa_page(range(0, 3), "c2")])
    src = oa_src()
    assert len(src.search("depth", limit=3)) == 3
    assert route.calls[0].request.url.params["per-page"] == "3" and route.call_count == 1


@respx.mock
def test_s2_paging_stops_at_1000_cap():
    def s2_page(request):
        off = int(request.url.params["offset"])
        n = int(request.url.params["limit"])
        return httpx.Response(200, json={"total": 50000, "offset": off, "next": off + n, "data": [
            {"paperId": f"p{off + i}", "title": f"Depth paper {off + i}", "year": 2022} for i in range(n)]})
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(side_effect=s2_page)
    src = SemanticScholarSource()
    src.min_delay = src._delay = 0
    pages = list(src.search_pages("depth", page_size=100))   # max_results=None -> source cap
    assert sum(len(p) for p in pages) == 1000 and src.last_total == 50000
    assert src.SEARCH_CAP == 1000


@respx.mock
def test_arxiv_pages_and_total():
    def atom(start, n, total):
        entries = "".join(
            f'<entry><id>http://arxiv.org/abs/2401.{start + i:05d}v1</id><published>2024-01-01T00:00:00Z</published>'
            f'<title>Depth paper {start + i}</title><summary>s</summary><author><name>A</name></author></entry>'
            for i in range(n))
        return httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
                                        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
                                        f'<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>')
    respx.get("https://export.arxiv.org/api/query").mock(side_effect=[atom(0, 3, 8), atom(3, 3, 8), atom(6, 2, 8)])
    src = ArxivSource()
    src.min_delay = src._delay = 0
    pages = list(src.search_pages("depth", page_size=3))
    assert [len(p) for p in pages] == [3, 3, 2] and src.last_total == 8   # short last page ends it
