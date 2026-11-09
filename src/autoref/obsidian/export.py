"""Obsidian vault writer: paper notes with typed links, category MOCs, index, inbox, canvas."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import yaml
from sqlalchemy import select

from ..config import iter_categories
from ..db.schema import CategoryAssignment, FieldProvenance, Paper, Relation, TopicPaper, utcnow
from ..pipeline.context import Ctx, session

# notes written before the rename to autoref
LEGACY_NOTES_MARKER = "<!-- liblib:notes — everything below this line is yours and is kept on re-export -->"
NOTES_MARKER = "<!-- autoref:notes — everything below this line is yours and is kept on re-export -->"
PAPERS_DIR = "Papers"
CATS_DIR = "Categories"
GRAPH_FILE = ".autoref/graph.json"          # read by the bundled "autoref paper map" plugin
PLUGIN_ID = "autoref-graph"
PLUGIN_SRC = Path(__file__).parent / "plugin"


def safe_name(text: str, maxlen: int = 90) -> str:
    t = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", text)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    return t[:maxlen].rstrip()


def base_note_name(p: Paper) -> str:
    first = p.author_names[0].split()[-1] if p.authors else "Anon"
    words = p.title.split()
    title = " ".join(words[:10])
    return safe_name(f"{first} {p.year or 'nd'} - {title}")


def cat_tag(path: str) -> str:
    return "cat/" + re.sub(r"[^\w/\-]", "_", path)


def yaml_block(data: dict) -> str:
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def _one_line(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class VaultExporter:
    def __init__(self, ctx: Ctx, vault: Path | None = None):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.vault = vault or ctx.cfg.vault_path
        self.labels = {l.relation_label: l for l in self.cfg.relations.labels}

    # ---------------------------------------------------------------- names
    def _assign_names(self, s, tps: list[TopicPaper]) -> dict[int, str]:
        taken = {tp.note_name for tp in tps if tp.note_name}
        names = {}
        for tp in tps:
            if not tp.note_name:
                p = s.get(Paper, tp.paper_id)
                base, n, i = base_note_name(p), None, 1
                n = base
                while n in taken:
                    i += 1
                    n = f"{base} ({i})"
                tp.note_name = n
                taken.add(n)
            names[tp.paper_id] = tp.note_name
        return names

    # ---------------------------------------------------------------- export
    def export(self, *, force: bool = False, canvas: list[str] | None = None) -> dict:
        stats = {"written": 0, "unchanged": 0, "categories": 0, "canvas": 0}
        (self.vault / PAPERS_DIR).mkdir(parents=True, exist_ok=True)
        (self.vault / CATS_DIR).mkdir(parents=True, exist_ok=True)
        topic = self.ctx.topic
        new_since = utcnow() - timedelta(days=self.cfg.export.inbox_days)
        with session(self.ctx) as s:
            tps = list(s.scalars(select(TopicPaper).where(TopicPaper.topic == topic,
                                                          TopicPaper.status == "included")))
            names = self._assign_names(s, tps)
            ids = set(names)
            cats = defaultdict(list)
            for ca in s.scalars(select(CategoryAssignment).where(CategoryAssignment.topic == topic)):
                # categories removed from the current tree are hidden until the paper is re-categorized
                if ca.paper_id in ids and self.cfg.categorize.get(ca.category_id):
                    cats[ca.paper_id].append(ca)
            rels_out, rels_in = defaultdict(list), defaultdict(list)
            for r in s.scalars(select(Relation).where(Relation.topic == topic)):
                if r.citing_id in ids and r.cited_id in ids:
                    rels_out[r.citing_id].append(r)
                    rels_in[r.cited_id].append(r)
            prov = defaultdict(dict)
            for fp in s.scalars(select(FieldProvenance).where(FieldProvenance.paper_id.in_(ids))):
                prov[fp.paper_id][fp.field] = fp

            for tp in tps:
                p = s.get(Paper, tp.paper_id)
                content = self.render_paper(p, tp, names, cats[p.id], rels_out[p.id], rels_in[p.id],
                                            prov[p.id], is_new=tp.added_at >= new_since)
                h = hashlib.sha1(content.encode()).hexdigest()
                path = self.vault / PAPERS_DIR / f"{names[p.id]}.md"
                if not force and tp.export_hash == h and path.exists():
                    stats["unchanged"] += 1
                    continue
                write_preserving_notes(path, content)
                tp.export_hash = h
                stats["written"] += 1

            stats["categories"] = self.write_categories(s, tps, names, cats)
            self.write_index(s, tps, names, cats, rels_out)
            self.write_inbox(s, tps, names, new_since)
            self.write_graph(s, tps, names, cats, rels_out, new_since)
            stats["plugin"] = self.install_plugin()
            for cat_id in (canvas or []) + list(self.cfg.export.canvas_categories):
                self.write_canvas(s, cat_id, names, cats, rels_out)
                stats["canvas"] += 1
        return stats

    # ---------------------------------------------------------------- paper note
    def render_paper(self, p: Paper, tp: TopicPaper, names, cats, rels_out, rels_in, prov, *, is_new: bool) -> str:
        cc = self.cfg.categorize
        tags = [cat_tag(cc.path_of(c.category_id)) for c in cats]  # path from the current tree
        if is_new:
            tags.append("status/new")
        if tp.core:
            tags.append("status/core")
        fm: dict = {
            "title": p.title,
            "authors": p.author_names,
            "year": p.year,
            "venue": p.venue,
            "url": p.url,
            "doi": p.doi,
            "arxiv": p.arxiv_id,
            "pdf": p.pdf_url,
            "citations": p.citation_count,
            "categories": [cc.get(c.category_id).label if cc.get(c.category_id) else c.category_id for c in cats],
            "relevance": round(tp.relevance, 2) if tp.relevance is not None else None,
            "origin": tp.origin,
            "depth": tp.depth,
            "added": tp.added_at.date(),
            "autoref_id": p.id,
            "tags": tags,
        }
        for field, fp in prov.items():
            if fp.origin.startswith("llm"):
                fm[f"{field}_source"] = fp.origin
        typed = defaultdict(list)
        for r in rels_out:
            typed[r.label].append(f"[[{names[r.cited_id]}]]")
        for label in self.labels:
            if typed.get(label):
                fm[label] = typed[label]
        fm = {k: v for k, v in fm.items() if v not in (None, [], "")}

        lines = [yaml_block(fm), f"# {p.title}", ""]
        meta = " · ".join(x for x in [", ".join(p.author_names[:6]) + (" et al." if len(p.author_names) > 6 else ""),
                                      str(p.year or ""), p.venue or ""] if x)
        lines += [meta, ""]
        links = " | ".join(f"[{k}]({v})" for k, v in (("page", p.url), ("pdf", p.pdf_url)) if v)
        if links:
            lines += [links, ""]
        lines += ["## Abstract", ""]
        abstract_note = ""
        if "abstract" in prov and prov["abstract"].origin.startswith("llm"):
            abstract_note = f"> [!warning] Abstract generated by LLM ({prov['abstract'].origin}, {prov['abstract'].model})\n\n"
        lines += [abstract_note + (p.abstract or "_No abstract available._"), ""]
        if cats:
            lines += ["## Categories", ""]
            for c in sorted(cats, key=lambda c: -(c.confidence or 0)):
                cat = cc.get(c.category_id)
                label = cat.label if cat else c.category_id
                lines.append(f"- [[{CATS_DIR}/{safe_name(label)}|{label}]] ({c.confidence:.2f})")
            if cats[0].rationale:
                lines += ["", f"> {_one_line(cats[0].rationale)}"]
            lines.append("")
        if rels_out:
            lines += ["## Relations (this paper → cited)", ""]
            for r in sorted(rels_out, key=lambda r: r.label):
                lines += self._relation_lines(r, names[r.cited_id], r.label)
            lines.append("")
        if rels_in:
            lines += ["## Cited by (in this vault)", ""]
            for r in sorted(rels_in, key=lambda r: r.label):
                lab = self.labels.get(r.label)
                lines.append(f"- [[{names[r.citing_id]}]] — {lab.label if lab else r.label}: {_one_line(r.relation_detail)}")
            lines.append("")
        lines += ["## Notes", NOTES_MARKER, ""]
        return "\n".join(lines)

    def _relation_lines(self, r: Relation, target: str, label: str) -> list[str]:
        lab = self.labels.get(label)
        out = [f"- {label}:: [[{target}]] — {_one_line(r.relation_detail)}"]
        if r.difference_summary:
            out.append(f"\t- Difference: {_one_line(r.difference_summary)}")
        for ev in (r.evidence or [])[:3]:
            out.append(f"\t- Evidence: “{_one_line(ev)}”")
        out.append(f"\t- _{lab.label if lab else label}, confidence {r.confidence or 0:.2f}, {r.model}_")
        return out

    # ---------------------------------------------------------------- categories
    def write_categories(self, s, tps, names, cats) -> int:
        n = 0
        by_cat = defaultdict(list)
        for pid, cas in cats.items():
            for ca in cas:
                by_cat[ca.category_id].append(pid)
        parents = {}
        for path, c in iter_categories(self.cfg.categorize.categories):
            parts = path.split("/")
            parents[c.id] = parts[-2] if len(parts) > 1 else None
        cc = self.cfg.categorize
        for path, c in iter_categories(cc.categories):
            lines = [yaml_block({"category_id": c.id, "path": path, "tags": ["autoref/category"]}),
                     f"# {c.label}", "", c.description.strip(), ""]
            if parents[c.id]:
                par = cc.get(parents[c.id])
                lines += [f"Parent: [[{CATS_DIR}/{safe_name(par.label)}|{par.label}]]", ""]
            if c.subcategories:
                lines += ["## Subcategories", ""]
                lines += [f"- [[{CATS_DIR}/{safe_name(sc.label)}|{sc.label}]] ({len(by_cat.get(sc.id, []))})"
                          for sc in c.subcategories]
                lines.append("")
            pids = sorted(by_cat.get(c.id, []), key=lambda pid: -(s.get(Paper, pid).year or 0))
            lines += [f"## Papers ({len(pids)})", ""]
            lines += [f"- [[{names[pid]}]]" for pid in pids]
            lines += ["", "## Dataview (includes subcategories)", "", "```dataview",
                      "TABLE year, venue, citations, relevance", f"FROM #{cat_tag(path)}",
                      "SORT year DESC", "```", ""]
            (self.vault / CATS_DIR / f"{safe_name(c.label)}.md").write_text("\n".join(lines), encoding="utf-8")
            n += 1
        return n

    # ---------------------------------------------------------------- index / inbox
    def write_index(self, s, tps, names, cats, rels_out):
        cc = self.cfg.categorize
        n_rel = sum(len(v) for v in rels_out.values())
        lines = [f"# {self.cfg.name}", "",
                 f"Papers: **{len(tps)}** · relations: **{n_rel}** · exported {utcnow():%Y-%m-%d %H:%M} UTC", "",
                 "## Categories", ""]
        for path, c in iter_categories(cc.categories):
            indent = "  " * path.count("/")
            count = sum(1 for cas in cats.values() if any(ca.category_id == c.id for ca in cas))
            lines.append(f"{indent}- [[{CATS_DIR}/{safe_name(c.label)}|{c.label}]] ({count})")
        lines += ["", "## Relation types", ""]
        lines += [f"- `{l.relation_label}` — **{l.label}**: {l.description.strip()}" for l in self.cfg.relations.labels]
        lines += ["", "## Most cited", "", "```dataview", "TABLE year, citations, categories",
                  f'FROM "{PAPERS_DIR}"', "SORT citations DESC", "LIMIT 30", "```", "",
                  "## Uncategorized", ""]
        lines += [f"- [[{names[tp.paper_id]}]]" for tp in tps if not cats.get(tp.paper_id)]
        (self.vault / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_inbox(self, s, tps, names, new_since):
        new = sorted((tp for tp in tps if tp.added_at >= new_since), key=lambda tp: tp.added_at, reverse=True)
        lines = [f"# Inbox — papers added in the last {self.cfg.export.inbox_days} days", "",
                 "```dataview", "TABLE added, year, categories, origin", f'FROM "{PAPERS_DIR}"',
                 "WHERE contains(tags, \"status/new\")", "SORT added DESC", "```", "", "## Static list", ""]
        lines += [f"- {tp.added_at:%Y-%m-%d} [[{names[tp.paper_id]}]] ({tp.origin})" for tp in new]
        (self.vault / "Inbox.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- graph data + plugin
    def write_graph(self, s, tps, names, cats, rels_out, new_since) -> None:
        """Data for the paper-map plugin: category tree, one primary category per paper, relations
        with the full LLM analysis."""
        cc = self.cfg.categorize
        tree = []
        for path, c in iter_categories(cc.categories):
            parts = path.split("/")
            tree.append({"id": c.id, "label": c.label, "description": _one_line(c.description),
                         "parent": parts[-2] if len(parts) > 1 else None, "top": parts[0], "depth": len(parts) - 1})
        depth = {c["id"]: c["depth"] for c in tree}
        inbound = defaultdict(int)
        for rs in rels_out.values():
            for r in rs:
                inbound[r.cited_id] += 1
        papers = []
        for tp in tps:
            p = s.get(Paper, tp.paper_id)
            mine = sorted(cats.get(p.id, []), key=lambda c: (-(c.confidence or 0), -depth.get(c.category_id, 0)))
            papers.append({
                "id": p.id, "note": f"{PAPERS_DIR}/{names[p.id]}.md", "name": names[p.id], "title": p.title,
                "year": p.year, "venue": p.venue, "url": p.url, "authors": p.author_names[:4],
                "n_authors": len(p.author_names), "citations": p.citation_count, "relevance": tp.relevance,
                "primary": mine[0].category_id if mine else None,
                "categories": [{"id": c.category_id, "confidence": c.confidence} for c in mine],
                "rationale": _one_line(mine[0].rationale) if mine else None,
                "inbound": inbound.get(p.id, 0), "new": tp.added_at >= new_since,
                "abstract": _one_line(p.abstract)[:900] or None,
            })
        relations = [{
            "id": r.id, "source": r.citing_id, "target": r.cited_id, "label": r.label,
            "detail": _one_line(r.relation_detail), "difference": _one_line(r.difference_summary),
            "evidence": [_one_line(e) for e in (r.evidence or [])[:4]], "confidence": r.confidence, "model": r.model,
        } for rs in rels_out.values() for r in rs]
        data = {"topic": self.cfg.name, "generated": utcnow().isoformat(timespec="seconds"),
                "categories": tree,
                "relation_labels": [{"id": l.relation_label, "label": l.label, "description": _one_line(l.description)}
                                    for l in self.cfg.relations.labels],
                "papers": papers, "relations": relations}
        path = self.vault / GRAPH_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def install_plugin(self) -> str:
        """Copy the bundled plugin into <vault>/.obsidian/plugins/. Enabling it stays the user's choice."""
        if not PLUGIN_SRC.is_dir():
            return "missing"
        dest = self.vault / ".obsidian" / "plugins" / PLUGIN_ID
        dest.mkdir(parents=True, exist_ok=True)
        changed = False
        for f in PLUGIN_SRC.iterdir():
            if f.is_file() and (not (dest / f.name).exists() or (dest / f.name).read_bytes() != f.read_bytes()):
                shutil.copyfile(f, dest / f.name)
                changed = True
        return "updated" if changed else "ok"

    # ---------------------------------------------------------------- canvas
    def write_canvas(self, s, cat_id: str, names, cats, rels_out):
        cat = self.cfg.categorize.get(cat_id)
        if not cat:
            raise ValueError(f"unknown category {cat_id!r}")
        sub_ids = {c.id for p, c in iter_categories([cat])}
        pids = [pid for pid, cas in cats.items() if any(ca.category_id in sub_ids for ca in cas)]
        by_year = defaultdict(list)
        for pid in pids:
            by_year[s.get(Paper, pid).year or 0].append(pid)
        nodes, node_id = [], {}
        for xi, year in enumerate(sorted(by_year)):
            for yi, pid in enumerate(by_year[year]):
                nid = f"p{pid}"
                node_id[pid] = nid
                nodes.append({"id": nid, "type": "file", "file": f"{PAPERS_DIR}/{names[pid]}.md",
                              "x": xi * 500, "y": yi * 320, "width": 400, "height": 260})
        colors = {l: str(i % 6 + 1) for i, l in enumerate(self.labels)}
        edges = []
        for pid in pids:
            for r in rels_out.get(pid, []):
                if r.cited_id in node_id:
                    lab = self.labels.get(r.label)
                    edges.append({"id": f"r{r.id}", "fromNode": node_id[r.citing_id], "fromSide": "left",
                                  "toNode": node_id[r.cited_id], "toSide": "right", "color": colors.get(r.label, "1"),
                                  "label": f"{lab.label if lab else r.label}: {_one_line(r.relation_detail)[:120]}"})
        path = self.vault / f"{safe_name(cat.label)}.canvas"
        path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=1), encoding="utf-8")


def write_preserving_notes(path: Path, content: str) -> None:
    """Write generated content; keep whatever the user wrote below NOTES_MARKER."""
    user = ""
    if path.exists():
        old = path.read_text(encoding="utf-8")
        for marker in (NOTES_MARKER, LEGACY_NOTES_MARKER):
            if marker in old:
                user = old.split(marker, 1)[1].lstrip("\n")
                break
    if user:
        content = content.rstrip("\n") + "\n" + user
    path.write_text(content, encoding="utf-8")


def export_vault(ctx: Ctx, *, vault: Path | None = None, force: bool = False, canvas: list[str] | None = None) -> dict:
    return VaultExporter(ctx, vault).export(force=force, canvas=canvas)
