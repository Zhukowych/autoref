from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from autoref.config import TopicConfig, load_config
from autoref.db import repo
from autoref.db.schema import CategoryAssignment, Relation, SyncRun, TopicPaper, utcnow
from autoref.models import CitationContext, ExternalIds, PaperRecord, ReferenceRecord
from autoref.obsidian.export import NOTES_MARKER, export_vault
from autoref.pipeline.categorize import run_categorize
from autoref.pipeline.context import session
from autoref.pipeline.relations import run_relations
from autoref.pipeline.search import apply_filter, keyword_score
from autoref.pipeline.update import compute_window, finish_run, start_run


def rec(title, **kw):
    return PaperRecord(title=title, source="openalex", **kw)


def test_config_tree_and_paths(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg.categorize.path_of("mde_metric") == "mde/mde_metric"
    assert cfg.categorize.get("map_fusion").label.startswith("Depth fusion")
    assert cfg.relations.get("uses_component").label == "Uses component"


def test_config_rejects_duplicate_ids(cfg_path):
    import yaml
    raw = yaml.safe_load(cfg_path.read_text())
    raw["categorize"]["categories"].append({"id": "mde", "label": "x", "description": "y"})
    with pytest.raises(ValidationError):
        TopicConfig.model_validate(raw)


def test_dedup_merge(ctx):
    with session(ctx) as s:
        a, c1 = repo.upsert_paper(s, rec("Depth Anything: Unleashing the Power", ids=ExternalIds(arxiv_id="2401.10891v2")))
        b, c2 = repo.upsert_paper(s, PaperRecord(title="Depth anything - unleashing the power", year=2024,
                                                 abstract="abs", source="arxiv",
                                                 ids=ExternalIds(doi="10.1/X")))
        c, c3 = repo.upsert_paper(s, rec("other", ids=ExternalIds(doi="https://doi.org/10.1/x")))
        assert c1 and not c2 and not c3 and a.id == b.id == c.id
        assert a.abstract == "abs" and a.doi == "10.1/x" and set(a.sources.split(",")) == {"openalex", "arxiv"}


def test_keyword_score():
    assert keyword_score("Monocular depth for UAV", ["depth", "uav", "lidar"], 2)[0] == 1.0
    assert keyword_score("Monocular depth", ["depth", "uav"], 2)[0] == 0.5


def _seed(ctx):
    with session(ctx) as s:
        a, _ = repo.upsert_paper(s, rec("Self-supervised aerial depth for drones", year=2024, abstract="UAV depth",
                                        authors=["Ann Smith"], ids=ExternalIds(doi="10.1/a")))
        b, _ = repo.upsert_paper(s, rec("Digging into self-supervised monocular depth", year=2019,
                                        abstract="Monodepth2", authors=["Clément Godard"], ids=ExternalIds(doi="10.1/b")))
        repo.add_reference(s, a, ReferenceRecord(cited=rec(b.title, ids=ExternalIds(doi="10.1/b")),
                                                 contexts=[CitationContext(text="we use the encoder of [3]")],
                                                 intents=["methodology"]), "semantic_scholar")
        for p in (a, b):
            repo.membership(s, ctx.topic, p)
        return a.id, b.id


def test_full_llm_flow_and_export(ctx):
    a, b = _seed(ctx)
    ctx.cfg.search.filter.mode = "llm"
    assert apply_filter(ctx) == {"included": 2, "excluded": 0}
    assert run_categorize(ctx)["categorized"] == 2
    assert run_relations(ctx) == {"labeled": 1, "failed": 0}
    assert run_relations(ctx) == {"labeled": 0, "failed": 0}  # cached
    with session(ctx) as s:
        ca = s.query(CategoryAssignment).filter_by(paper_id=a).one()
        assert ca.category_path == "mde/mde_self_supervised"
        rel = s.query(Relation).one()
        assert (rel.citing_id, rel.cited_id, rel.label) == (a, b, "uses_component")
        assert "ResNet encoder" in rel.relation_detail and rel.evidence == ["we use [3] encoder"]
    # relation prompt carried the citation sentence and the label definitions
    sys_p, user_p = next(c for c in ctx_calls() if "CITING paper" in c[1])
    assert "we use the encoder of [3]" in user_p and "uses_component (Uses component)" in sys_p

    st = export_vault(ctx)
    assert st["written"] == 2
    vault = ctx.cfg.vault_path
    note_a = next((vault / "Papers").glob("Smith 2024*.md"))
    text = note_a.read_text()
    assert "uses_component:\n- '[[Godard 2019 - Digging into self-supervised monocular depth]]'" in text
    assert "uses_component:: [[Godard 2019" in text and "Difference: adds UAV data" in text
    assert "cat/mde/mde_self_supervised" in text and "status/new" in text
    note_b = next((vault / "Papers").glob("Godard 2019*.md")).read_text()
    assert "## Cited by" in note_b and "Smith 2024" in note_b
    assert (vault / "Categories" / "Self-supervised MDE.md").exists()

    # user notes survive re-export; unchanged notes are not rewritten
    note_a.write_text(text + "my precious notes\n")
    assert export_vault(ctx)["written"] == 0
    st = export_vault(ctx, force=True)
    assert st["written"] == 2
    assert note_a.read_text().endswith(NOTES_MARKER + "\nmy precious notes\n")

    export_vault(ctx, canvas=["mde"])
    import json
    canvas = json.loads((vault / "Monocular Depth Estimation.canvas").read_text())
    assert len(canvas["nodes"]) == 2 and canvas["edges"][0]["label"].startswith("Uses component")


def ctx_calls():
    from conftest import FakeLLM
    return FakeLLM.calls


def test_update_window(ctx):
    today = date(2026, 11, 9)
    lo, hi = compute_window(ctx, today=today)
    assert (lo, hi) == (today - timedelta(days=30), today)
    rid = start_run(ctx, "update")
    finish_run(ctx, rid, status="ok")
    with session(ctx) as s:
        s.get(SyncRun, rid).started_at = utcnow().replace(year=2026, month=11, day=1)
    assert compute_window(ctx, today=today)[0] == date(2026, 10, 29)  # minus 3-day overlap
    bad = start_run(ctx, "update")
    finish_run(ctx, bad, status="failed")
    assert compute_window(ctx, today=today)[0] == date(2026, 10, 29)  # failed run does not advance
    assert compute_window(ctx, since=date(2026, 1, 1), today=today)[0] == date(2026, 1, 1)


def test_manual_status_respected(ctx):
    a, b = _seed(ctx)
    with session(ctx) as s:
        tp = s.get(TopicPaper, (ctx.topic, a))
        tp.status, tp.manual = "excluded", True
    ctx.cfg.search.filter.mode = "none"
    apply_filter(ctx, rescore=True)
    with session(ctx) as s:
        assert s.get(TopicPaper, (ctx.topic, a)).status == "excluded"
        assert s.get(TopicPaper, (ctx.topic, b)).status == "included"
