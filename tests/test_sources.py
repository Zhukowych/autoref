import httpx
import respx

from autoref.models import ExternalIds, PaperRecord
from autoref.sources.arxiv import ArxivSource, build_query, parse_feed
from autoref.sources.base import SourceError
from autoref.sources.openalex import OpenAlexSource, parse_work, rebuild_abstract
from autoref.sources.semantic_scholar import SemanticScholarSource, parse_paper

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2401.10891v2</id><published>2024-01-19T18:59:57Z</published>
<title>Depth Anything: Unleashing the Power of
 Large-Scale Unlabeled Data</title><summary>  We present Depth Anything. </summary>
<author><name>Lihe Yang</name></author><author><name>Bingyi Kang</name></author>
<link href="http://arxiv.org/pdf/2401.10891v2" rel="related" type="application/pdf"/>
</entry></feed>"""


def test_arxiv_parse_and_query():
    [p] = parse_feed(ATOM)
    assert p.ids.arxiv_id == "2401.10891"
    assert p.title.startswith("Depth Anything: Unleashing")
    assert p.authors == ["Lihe Yang", "Bingyi Kang"] and p.year == 2024
    q = build_query('"monocular depth" UAV of', ["cs.CV"])
    assert q == '(all:"monocular depth" AND all:UAV) AND (cat:cs.CV)'


def test_openalex_parse():
    w = {"id": "https://openalex.org/W1", "title": "Digging Into Self-Supervised Monocular Depth Estimation",
         "doi": "https://doi.org/10.1109/ICCV.2019.00393", "publication_year": 2019,
         "abstract_inverted_index": {"Per-pixel": [0], "depth": [1, 3], "ground-truth": [2]},
         "authorships": [{"author": {"display_name": "Clément Godard"}}],
         "primary_location": {"source": {"display_name": "ICCV"}},
         "locations": [{"landing_page_url": "https://arxiv.org/abs/1806.01260v4"}], "cited_by_count": 3000}
    p = parse_work(w)
    assert p.ids.openalex_id == "W1" and p.ids.doi == "10.1109/iccv.2019.00393"
    assert p.ids.arxiv_id == "1806.01260"
    assert p.abstract == "Per-pixel depth ground-truth depth"
    assert p.venue == "ICCV" and p.pdf_url == "https://arxiv.org/pdf/1806.01260"
    assert rebuild_abstract(None) is None


def test_s2_parse():
    p = parse_paper({"paperId": "abc", "title": "T", "externalIds": {"ArXiv": "2101.1", "DOI": "10.1/x"},
                     "authors": [{"name": "A"}], "year": 2021})
    assert p.ids.s2_id == "abc" and p.pdf_url == "https://arxiv.org/pdf/2101.1"


@respx.mock
def test_s2_references_with_contexts():
    respx.get("https://api.semanticscholar.org/graph/v1/paper/abc/references").mock(return_value=httpx.Response(
        200, json={"data": [{"contexts": ["we use the encoder of [4]"], "intents": ["methodology"],
                             "isInfluential": True,
                             "citedPaper": {"paperId": "def", "title": "ResNet", "year": 2016}}]}))
    src = SemanticScholarSource()
    src.min_delay = 0
    [ref] = src.references(PaperRecord(title="x", ids=ExternalIds(s2_id="abc")))
    assert ref.cited.title == "ResNet" and ref.contexts[0].text.startswith("we use")
    assert ref.intents == ["methodology"] and ref.is_influential


@respx.mock
def test_circuit_breaker(monkeypatch):
    route = respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(429))
    src = OpenAlexSource()
    src.min_delay = 0
    monkeypatch.setattr(OpenAlexSource._get_retrying.retry, "wait", lambda *_: 0)
    for _ in range(2):
        try:
            src.search("x", limit=1)
        except httpx.HTTPStatusError:
            pass
    n = route.call_count
    try:
        src.search("x", limit=1)
        raise AssertionError("expected SourceError")
    except SourceError:
        pass
    assert route.call_count == n  # no more HTTP once disabled
