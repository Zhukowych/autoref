"""Re-running steps after everything was downloaded: new category tree, new labels, deeper snowball."""
from autoref.config import Category
from autoref.db import repo
from autoref.db.schema import CategoryAssignment, Relation, TopicPaper
from autoref.models import ExternalIds, PaperRecord, ReferenceRecord
from autoref.obsidian.export import export_vault
from autoref.pipeline import categorize as cat_mod
from autoref.pipeline.categorize import run_categorize
from autoref.pipeline.context import session
from autoref.pipeline.relations import run_relations
from autoref.pipeline.snowball import run_snowball
from autoref.sources.base import PaperSource

from test_pipeline import _seed


def test_new_category_tree_recategorizes_only_stale(ctx):
    _seed(ctx)
    ctx.cfg.search.filter.mode = "none"
    from autoref.pipeline.search import apply_filter
    apply_filter(ctx)
    assert run_categorize(ctx)["categorized"] == 2
    assert run_categorize(ctx)["categorized"] == 0  # nothing stale

    # user designs a better tree: mde_self_supervised moves under a new parent
    cats = ctx.cfg.categorize.categories
    ss = next(c for c in cats[0].subcategories if c.id == "mde_self_supervised")
    cats[0].subcategories.remove(ss)
    cats.append(Category(id="learning", label="Learning paradigm", description="x", subcategories=[ss]))
    with session(ctx) as s:
        assert cat_mod.stale_count(ctx, s) == 2
    assert run_categorize(ctx)["categorized"] == 2
    with session(ctx) as s:
        ca = s.query(CategoryAssignment).first()
        assert ca.category_path == "learning/mde_self_supervised"
        assert ca.scheme == ctx.cfg.categorize.scheme_id
        assert cat_mod.stale_count(ctx, s) == 0

    # explicit scheme name: editing descriptions no longer invalidates
    ctx.cfg.categorize.scheme = "v2"
    run_categorize(ctx)
    cats[-1].description = "reworded"
    with session(ctx) as s:
        assert cat_mod.stale_count(ctx, s) == 0

    # removed categories are hidden in export
    cats.pop()  # drop the "learning" subtree entirely
    export_vault(ctx)
    note = next((ctx.cfg.vault_path / "Papers").glob("Smith*.md")).read_text()
    assert "mde_self_supervised" not in note


def test_new_relation_labels_relabel(ctx):
    _seed(ctx)
    ctx.cfg.search.filter.mode = "none"
    from autoref.pipeline.search import apply_filter
    apply_filter(ctx)
    assert run_relations(ctx)["labeled"] == 1
    assert run_relations(ctx)["labeled"] == 0
    ctx.cfg.relations.labels[0].description = "sharper definition"
    assert run_relations(ctx)["labeled"] == 1
    with session(ctx) as s:
        assert s.query(Relation).one().scheme == ctx.cfg.relations.scheme_id


class GraphSource(PaperSource):
    """Fake source: paper N cites papers N*10+1 .. N*10+3."""
    name = "graph"
    capabilities = frozenset({"references"})
    min_delay = 0
    calls: list = []

    def search(self, *a, **k):
        return []

    def references(self, paper, limit=1000):
        GraphSource.calls.append(paper.title)
        n = int(paper.title.split()[-1])
        return [ReferenceRecord(cited=PaperRecord(title=f"depth monocular paper {n * 10 + i}", abstract="depth uav",
                                                  citation_count=10 - i, ids=ExternalIds(doi=f"10.9/{n * 10 + i}")))
                for i in (1, 2, 3)]


def test_deeper_snowball_after_first_run(ctx):
    class Srcs:
        def with_cap(self, cap):
            return [GraphSource()] if cap == "references" else []
        def close(self):
            pass
    ctx._sources = Srcs()
    GraphSource.calls = []
    ctx.cfg.search.filter.mode = "none"
    with session(ctx) as s:
        p, _ = repo.upsert_paper(s, PaperRecord(title="depth monocular paper 1", ids=ExternalIds(doi="10.9/1")))
        tp, _ = repo.membership(s, ctx.topic, p)
        tp.status = "included"

    r1 = run_snowball(ctx, depth=1, max_per_paper=1)
    assert r1["included"] == 1 and GraphSource.calls == ["depth monocular paper 1"]

    # more neighbours from STORED edges: no API call
    r2 = run_snowball(ctx, depth=1, max_per_paper=3)
    assert r2["included"] == 2 and len(GraphSource.calls) == 1

    # deeper: continues from the depth-1 papers; root is not re-downloaded
    r3 = run_snowball(ctx, depth=2, max_per_paper=3)
    assert len(GraphSource.calls) == 4 and r3["included"] == 9
    with session(ctx) as s:
        depths = sorted(tp.depth for tp in s.query(TopicPaper).filter_by(status="included"))
        assert depths == [0] + [1] * 3 + [2] * 9
