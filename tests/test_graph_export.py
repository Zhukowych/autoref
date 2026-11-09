"""graph.json for the paper-map plugin + plugin installation."""
import json

from autoref.obsidian.export import GRAPH_FILE, PLUGIN_ID, export_vault
from autoref.pipeline.categorize import run_categorize
from autoref.pipeline.relations import run_relations
from autoref.pipeline.search import apply_filter
from test_pipeline import _seed


def test_export_writes_graph_data_and_plugin(ctx):
    a, b = _seed(ctx)
    ctx.cfg.search.filter.mode = "llm"
    apply_filter(ctx)
    run_categorize(ctx)
    run_relations(ctx)
    export_vault(ctx)
    vault = ctx.cfg.vault_path
    data = json.loads((vault / GRAPH_FILE).read_text(encoding="utf-8"))
    assert {"categories", "papers", "relations", "relation_labels"} <= set(data)
    by_id = {p["id"]: p for p in data["papers"]}
    assert set(by_id) == {a, b}
    assert by_id[a]["primary"] == "mde_self_supervised" and by_id[b]["inbound"] == 1
    (rel,) = data["relations"]
    assert (rel["source"], rel["target"], rel["label"]) == (a, b, "uses_component")
    assert rel["detail"] and rel["evidence"] == ["we use [3] encoder"]
    assert all(p["note"].startswith("Papers/") for p in data["papers"])
    plug = vault / ".obsidian" / "plugins" / PLUGIN_ID
    assert all((plug / f).exists() for f in ("main.js", "manifest.json", "styles.css"))
    assert export_vault(ctx)["plugin"] == "ok"  # second export: nothing to copy
