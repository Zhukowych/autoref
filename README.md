# autoref

Literature-review CLI. Harvests papers from **Semantic Scholar, OpenAlex and arXiv** into a local SQLite DB
(title, abstract, authors, year, venue, links, full reference lists + citation sentences), screens them for
relevance, **categorizes** them into your category tree with an LLM, **labels citation edges** (how one paper
uses another, in the papers' own terms, and how they differ) and exports an **Obsidian vault** with typed links.
Topic-agnostic: one YAML config per research topic. Re-run `autoref update` (or a systemd timer) to pull new papers.

## Install

```bash
cd autoref
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
source .venv/bin/activate          # or call .venv/bin/autoref
```

Keys (all optional, via env or the config):

| env | used for |
|---|---|
| `S2_API_KEY` | Semantic Scholar. **Recommended** — the unauthenticated pool is often rate-limited (429). Free: https://www.semanticscholar.org/product/api |
| `OPENALEX_MAILTO` | OpenAlex polite pool |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | API LLMs |

Without S2, references come from OpenAlex (no citation sentences — relation labeling then relies on abstracts).
Requests are spaced by a per-source politeness delay (S2 1.0–1.1 s, OpenAlex 0.15 s, arXiv 3.1 s). On HTTP 429/5xx or
network errors a request is retried up to 5 times with exponential backoff (2, 4, 8, 16, 32 s, capped at 60 s, random
jitter) and never earlier than the server's `Retry-After`; if the server asks for more than 300 s, that request
is abandoned. After a 429 the spacing between requests doubles (up to 8×) and shrinks again after 20 successes.
A source whose requests fail 2 times in a row (all retries used up) is switched off for the rest of the run.

## Quick start

```bash
autoref init mde_uav --from-template mde        # -> configs/mde_uav.yaml (thesis example)
$EDITOR configs/mde_uav.yaml                   # queries, categories, relation labels, LLMs
autoref llm test -c configs/mde_uav.yaml        # check LLM backends
autoref sources test -c configs/mde_uav.yaml    # check sources
autoref run -c configs/mde_uav.yaml             # everything; safe to re-run
```

`configs/mde_uav_mapping.yaml` is a ready config for the thesis topic. Relative paths (`db`, `vault`) resolve
against the project root (the folder containing `configs/`). Use `export AUTOREF_CONFIG=configs/x.yaml` to skip `-c`.

## Commands

Typical order: `search → snowball → enrich → categorize → relate → export obsidian` (`run` = all of them),
then `update` periodically. Every step is idempotent — it only processes what is not done yet (`--force` redoes).
Most writing commands accept `--dry-run`. `-v` before the command enables debug logs.

**Setup**
- `init <topic> [--from-template mde|generic]` — write `configs/<topic>.yaml` and create the empty DB.
- `llm test [--task categorize|relations|fill|relevance]` — tiny text + JSON request to each configured backend; shows latency and whether JSON output works.
- `sources test [--query ...]` — one search per source; shows capabilities and API-key status.

**Pipeline**
- `search [--source s2] [--query "..."]... [--limit N] [--no-filter]` — run queries on sources, normalize, dedup (DOI → arXiv id → S2/OpenAlex id → normalized title+year), merge fields from several sources, add to topic; then relevance gate (`search.filter`: `none` / `keywords` / `llm`). Excluded papers are kept (marked) so they aren't fetched again.
- `refilter` — re-score all (non-manual) papers after changing filter settings.
- `snowball [--depth N] [--direction references|citations|both] [--max-per-paper N] [--refresh]` — fetch reference lists (and/or citing papers) of included papers. *All* edges are stored (unknown papers as stubs); the top `max_per_paper` neighbours (by influence/citations) become candidates, pass the relevance gate, and are expanded again for depth > 1.
- `enrich [--fields abstract,year] [--no-llm] [--force]` — fill missing fields: other sources → open-access/arXiv PDF text → LLM extraction **from that text only** (never invented from a title). Every filled field gets a provenance row; LLM-filled ones are flagged in Obsidian (`abstract_source: llm`).
- `categorize [--force] [--paper KEY] [--limit N]` — LLM assigns category ids from the tree (`single`/`multi` mode); ids are validated (invalid output is retried). Stored as path, e.g. `mde/mde_metric`.
- `relate [--force] [--pair CITING CITED] [--limit N]` — for every citation edge between two included papers: citation sentences + S2 intents + both abstracts → LLM returns `label`, `relation_detail` (concrete: "uses DPT ViT-L encoder from X as backbone"), `difference_summary`, `evidence`, `confidence`.
- `export obsidian [--vault PATH] [--force] [--canvas CATEGORY_ID]...` — write/refresh the vault (only changed notes are rewritten).
- `run [--no-export]` — the whole pipeline.

**Interruptions (Ctrl+C, crash, network loss)** — just run the same command again.
Each query / paper / edge is committed as soon as it is done. `run`, `search` and `update` checkpoint finished
source×query pairs and skip them on the next run (`--restart` ignores the checkpoint); unscored candidates and
included-but-unexpanded papers are picked up automatically; an interrupted `update` keeps its date window.

**Re-running after everything is downloaded**
- *Better category tree:* edit `categorize.categories` and run `autoref categorize`. Each tree has a scheme id
  (hash of tree + mode, or your own `categorize.scheme: v2`); papers categorized under another scheme are
  stale and redone. The step is resumable: an interrupted run continues where it stopped. Categories no longer in the
  tree are hidden in the export. With an explicit `scheme`, editing descriptions does *not* trigger a
  recategorization; bump the name when you want one.
- *New relation labels/definitions:* same mechanism via `relations.scheme`; `autoref relate` redoes stale edges.
- *Deeper / wider graph:* `autoref snowball --depth 2` continues from the papers stored at depth 1
  (already-fetched papers cost no API calls); `--max-per-paper 60` promotes more neighbours from edges already
  in the DB (no API calls); `--direction citations` adds citing papers; `--refresh` re-downloads edges.
- *New relevance settings:* `autoref refilter`.
- `autoref stats` shows the current scheme ids, how many papers/edges are stale, and papers per depth level.
- Then `autoref export obsidian`.

**Maintenance**
- `update [--since YYYY-MM-DD] [--no-export]` — incremental: searches only the window *last successful run − overlap_days → today*, plus new papers citing your **core** papers (`core` flag or relevance ≥ `update.core_min_relevance`); then filter/snowball/enrich/categorize/relate only what's new and export. Failed runs don't advance the window; a lock file prevents overlapping runs.
- `schedule install --every 3d | remove | status` — systemd user timer running `autoref update` (prints a crontab line if systemd is missing). Put API keys for the timer in `~/.config/autoref/env`. `status` also lists recent runs.

**Browsing**
- `stats` — papers by status, stubs, edges, missing/LLM-filled fields, per-category and per-relation counts, last run.
- `list [--category ID] [--year-from Y] [--source S] [--status ...] [--new] [--sort citations|year|relevance|added]`
- `show KEY` — KEY = id, DOI, arXiv id, S2/OpenAlex id or title fragment. Metadata, provenance, categories + rationale, relations in/out.
- `mark KEY --include|--exclude|--core|--not-core` — manual decisions are never overridden by the filter.

## Config

See `configs/mde_uav_mapping.yaml`. Key parts:

```yaml
llm:                       # per task: default + optional categorize / relations / fill / relevance
  default:   {provider: openai_compat, base_url: http://localhost:11434/v1, model: qwen2.5:14b}
  relations: {provider: anthropic, model: claude-sonnet-5-5, api_key: "${ANTHROPIC_API_KEY}"}
categorize:
  categories:              # tree of any depth; id = key, label = display name, description = LLM instruction
    - id: mde
      label: Monocular Depth Estimation
      description: ...
      subcategories: [{id: mde_metric, label: Metric depth, description: ...}]
relations:
  labels:                  # relation_label = key, label = display name, description = LLM definition
    - {relation_label: uses_component, label: Uses component, description: ...}
```

LLM providers: `openai_compat` (OpenAI, Ollama `/v1`, LM Studio, vLLM, llama.cpp, OpenRouter — set `base_url`),
`anthropic`, `ollama` (native API with schema-constrained output). `${VAR}` / `${VAR:-default}` are expanded.

Prompts live in `src/autoref/prompts/*.txt`; copy one to `<project>/prompts/<name>.txt` to override it.

## Obsidian vault

```
vault/<topic>/
  index.md            overview, category tree with counts, relation-type legend
  Inbox.md            papers added in the last `export.inbox_days` days
  Papers/<Author> <Year> - <Title>.md
  Categories/<Label>.md   MOC per category (static list + Dataview query incl. subcategories)
  <Category>.canvas   optional: papers by year, edges labeled with relation + detail
```

Each paper note has frontmatter properties (`authors`, `year`, `url`, `doi`, `categories`, `tags: [cat/mde/mde_metric, status/new]`, …)
and **typed link properties** per relation label, e.g. `uses_component: ["[[Ranftl 2021 - Vision Transformers…]]"]`.
The body has the abstract, categories, `## Relations` with Dataview inline fields
(`- uses_component:: [[X]] — uses X's encoder …`, nested *Difference* and *Evidence*), `## Cited by`, and `## Notes`.
**Everything you write below the marker under `## Notes` is preserved on re-export.**

Edge types in Obsidian: the core graph view shows links but not their types. Recommended community plugins:
- **Dataview** — tables/queries in `index.md`, `Inbox.md`, category notes.
- **Extended Graph** or **Juggl** — colour/label graph edges by frontmatter property (`uses_component`, `builds_on_idea`, …).
- **Breadcrumbs** — treat the typed properties as hierarchies/relations.
- Canvas (core) — `--canvas mde` writes a canvas with labeled, coloured edges.

## Extending

- New source: subclass `autoref.sources.base.PaperSource` (`search`, optionally `get`, `references`, `citations`), declare `capabilities`, register in `sources/registry.py`.
- New LLM backend: subclass `autoref.llm.base.LLMClient`, implement `_chat(system, user, json_schema=None)`, register in `llm/registry.py`. JSON validation/retry is shared.

## Data model (SQLite)

`papers`, `authors`, `paper_authors`, `paper_references` (citing → cited, contexts, intents), `topic_papers`
(per-topic status/relevance/depth/core/origin), `categories`, `relations`, `field_provenance`, `sync_runs`.
One DB can serve several topics.

## Tests

```bash
.venv/bin/pytest
```
