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

## Quick start

```bash
autoref init mde_uav --from-template uav_metric # -> configs/mde_uav.yaml (thesis example)
$EDITOR configs/mde_uav.yaml                   # queries, categories, relation labels, LLMs
autoref llm test -c configs/mde_uav.yaml        # check LLM backends
autoref sources test -c configs/mde_uav.yaml    # check sources
autoref run -c configs/mde_uav.yaml             # everything; safe to re-run
```

`configs/mde_uav_mapping.yaml` is a ready config for the thesis topic. Relative paths (`db`, `vault`) resolve
against the project root (the folder containing `configs/`). Use `export AUTOREF_CONFIG=configs/x.yaml` to skip `-c`.

## Tutorial: run on a GPU server

Full setup on a Linux machine with an NVIDIA GPU (planned for an A100), from nothing to an Obsidian vault.
Uses `configs/uav_metric_mapping.yaml`. Run everything as a normal user.

What runs where:

- **LLM**: `Qwen/Qwen3-14B-AWQ` (official 4-bit build by the Qwen team, ~10 GB of weights) served by vLLM on
  `http://127.0.0.1:8002/v1` under the name `qwen3-14b`, capped at ~16 GB of GPU memory. One model does all four
  LLM tasks: relevance screening, categorization, relation labeling, filling missing metadata.
- **autoref**: its own Python environment in `~/autoref`, talks to the LLM over the OpenAI-compatible API and to
  Semantic Scholar / OpenAlex / arXiv over the internet. Data: `~/autoref/data/*.sqlite`, vault:
  `~/autoref/vault/uav_metric_mapping/`.

### 1. Check the machine

```bash
nvidia-smi                       # driver works, note GPU index and memory
python3 --version                # 3.11–3.13 (vLLM supports 3.10–3.13)
sudo apt install -y git tmux python3-venv   # Debian/Ubuntu; skip what is already installed
```

### 2. Install the LLM server (vLLM) in its own environment

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip uv
nvidia-smi | grep "CUDA Version"   # pick the install line that matches
```

The default vLLM wheel is built for CUDA 13. Install the variant that matches the driver's "CUDA Version":

```bash
# driver CUDA 13.x
uv pip install vllm --torch-backend=cu130

# driver CUDA 12.9 (or newer 12.x): vLLM's CUDA 12.9 wheel index (keeps a few releases behind, hence the pin)
uv pip install vllm==0.22.0 \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

```bash
python -c "import vllm, torch; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())"
deactivate
```

`ImportError: libcudart.so.13` when starting vLLM means the CUDA 13 wheel was installed on a CUDA 12 driver:
delete the environment (`rm -rf ~/llm/vllm-env`) and repeat this step with the CUDA 12.9 line.

### 3. Start the model

The model downloads from Hugging Face (`Qwen/Qwen3-14B-AWQ`) on first start into `~/.cache/huggingface`.
Run it inside tmux so it survives logout.

```bash
tmux new -s llm
source ~/llm/vllm-env/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B-AWQ \
  --served-model-name qwen3-14b \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.2
# 0.2 × 80 GB = 16 GB (weights + cache). On a 40 GB card use 0.4. Detach: Ctrl-b d. Reattach: tmux attach -t llm
```

Qwen3 "thinks" before answering by default; the config turns that off per request
(`extra_body: {chat_template_kwargs: {enable_thinking: false}}` under `llm.default`), which keeps answers short
and fast. `--reasoning-parser qwen3` keeps any thinking text out of the answer if it is switched back on.

Check from another shell (wait for "Application startup complete" first):

```bash
curl -s http://127.0.0.1:8002/v1/models
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"Reply with {\"ok\": true}"}],"chat_template_kwargs":{"enable_thinking":false}}'
nvidia-smi                       # the vllm process should use about 16 GB
```

If startup fails with out-of-memory, lower `--max-model-len` to 6144 or raise `--gpu-memory-utilization` to 0.25.

### 4. Install autoref

```bash
cd ~
git clone https://github.com/Zhukowych/autoref.git
cd autoref
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
source .venv/bin/activate
autoref --help
```

Optional — reuse papers already harvested on another computer (run on that computer):

```bash
rsync -av data/ <user>@<server>:~/autoref/data/
```

### 5. Keys

```bash
cat >> ~/.bashrc <<'EOF'
export S2_API_KEY=...              # Semantic Scholar key
export OPENALEX_API_KEY=...        # https://openalex.org/settings/api
export OPENALEX_MAILTO=you@example.org
export AUTOREF_CONFIG=~/autoref/configs/uav_metric_mapping.yaml   # so -c can be omitted
EOF
source ~/.bashrc
```

The LLM part of `configs/uav_metric_mapping.yaml` already matches step 3, nothing to edit:

```yaml
llm:
  default:
    provider: openai_compat
    base_url: ${AUTOREF_LLM_URL:-http://localhost:8002/v1}   # vLLM address from step 3
    model: ${AUTOREF_LLM_MODEL:-qwen3-14b}                   # = --served-model-name
    max_tokens: 1024
    timeout: 300
    extra_body:
      chat_template_kwargs: {enable_thinking: false}          # Qwen3 answers directly, no <think> block
```

All tasks (`relevance`, `categorize`, `relations`, `fill`) fall back to `default`.

### 6. Check everything

```bash
cd ~/autoref && source .venv/bin/activate
pytest -q                                   # offline tests
autoref sources test                        # each source answers, key status, OpenAlex budget
autoref llm test                            # all four LLM tasks return valid JSON
autoref llm test --task relations
```

### 7. Run the pipeline

```bash
tmux new -s autoref
cd ~/autoref && source .venv/bin/activate
autoref run --total 300        # first pass, capped: search → filter → snowball → enrich → categorize → relate → export
autoref stats                  # check numbers and categories before the big run
autoref run                    # full run; resumes where it stopped if interrupted
# Detach: Ctrl-b d. Reattach: tmux attach -t autoref
```

Steps can also be run one at a time: `autoref search`, `autoref snowball`, `autoref enrich`,
`autoref categorize`, `autoref relate`, `autoref export obsidian`.

### 8. Mark key papers and look around

```bash
autoref list --sort citations --limit 40
autoref mark "Depth Anything 3" --core      # core papers: update tracks new papers citing them
autoref mark "MoGe-2" --core
autoref mark "AerialMetric" --core
autoref show "AerialMetric"
```

### 9. Keep it up to date

```bash
autoref update                         # new papers since the last run
autoref schedule install --every 3d    # systemd user timer
autoref schedule status
loginctl enable-linger $USER           # let the timer run while you are logged out
```

The vLLM server from step 3 must be running when `update` runs.

### 10. Open the vault

Copy the vault to the computer with Obsidian, open it as a vault, then enable the plugin
(Settings → Community plugins → "Autoref paper map"):

```bash
rsync -av <user>@<server>:~/autoref/vault/uav_metric_mapping/ ~/Obsidian/uav_metric_mapping/   # run on your computer
```

## Keys and sources

Keys (all optional, via env or the config):

| env | used for |
|---|---|
| `S2_API_KEY` | Semantic Scholar. **Recommended** — the unauthenticated pool is often rate-limited (429). Free: https://www.semanticscholar.org/product/api |
| `OPENALEX_API_KEY` | OpenAlex. **Recommended** (free, https://openalex.org/settings/api). OpenAlex now meters usage: without a key you share ~1000 credits/day per IP (a search page costs 10, so ~100 pages/day) and cursor paging is refused (`page=N` paging, max 10,000 results, is used instead); a key gives $1/day (~1000 pages). Single-record lookups are free. autoref stops a search cleanly when the budget is used up; `autoref sources test` shows what is left. |
| `OPENALEX_MAILTO` | OpenAlex polite pool |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | API LLMs |

Without S2, references come from OpenAlex (no citation sentences — relation labeling then relies on abstracts).
Sources are configured as an ordered mapping (first = highest priority for references/citations/lookups):

```yaml
sources:
  semantic_scholar:            # alias: s2
    api_key: ${S2_API_KEY:-}
    max_rate: 1.0              # max requests per second
  openalex:                    # alias: oa
    api_key: ${OPENALEX_API_KEY:-}
    mailto: ${OPENALEX_MAILTO:-}
    max_rate: 6
  arxiv:
    max_rate: 0.32
    # categories: [cs.CV, cs.RO]   # optional arXiv category filter
    # enabled: false               # keep the entry but skip the source
```

`max_rate` is the maximum number of requests per second for that source (spacing = 1/max_rate; shared by all
calls of the process). Without it the built-in defaults apply: S2 1.0 req/s with a key (about 0.9 without), OpenAlex
~6.7, arXiv ~0.32. A value above the source's documented limit (S2 1, OpenAlex 10, arXiv 1 per 3 s) logs a warning.
The old format (`sources: [names]` plus `source_settings:`) still loads. Requests are spaced as above. On HTTP 429/5xx or
network errors a request is retried up to 5 times with exponential backoff (2, 4, 8, 16, 32 s, capped at 60 s, random
jitter) and never earlier than the server's `Retry-After`; if the server asks for more than 300 s, that request
is abandoned. After a 429 the spacing between requests doubles (up to 8×) and shrinks again after 20 successes.
A source whose requests fail 2 times in a row (all retries used up) is switched off for the rest of the run.

## Commands

Typical order: `search → snowball → enrich → categorize → relate → export obsidian` (`run` = all of them),
then `update` periodically. Every step is idempotent — it only processes what is not done yet (`--force` redoes).
Most writing commands accept `--dry-run`. `-v` before the command enables debug logs.

**Setup**
- `init <topic> [--from-template generic|mde|uav_metric|<path.yaml>]` — write `configs/<topic>.yaml` and create the
  empty DB. A path copies any existing topic config; `name`, `db` and `vault` are set to the new topic.
  `uav_metric` = the thesis setup (`configs/uav_metric_mapping.yaml`), `mde` = the older, broader MDE config.
- `llm test [--task categorize|relations|fill|relevance]` — tiny text + JSON request to each configured backend; shows latency and whether JSON output works.
- `sources test [--query ...]` — one search per source; shows capabilities, API-key/e-mail status and the effective max rate.

**Pipeline**
- `search [--source s2] [--query "..."]... [--limit N | --all] [--total N] [--no-filter]` — run queries on sources, normalize, dedup (DOI → arXiv id → S2/OpenAlex id → normalized title+year), merge fields from several sources, add to topic; then relevance gate (`search.filter`: `none` / `keywords` / `llm`). Excluded papers are kept (marked) so they aren't fetched again.
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

**How many results per query?** `search.limit_per_query: 200` fetches at most 200 per query and source (best match
first; paging is automatic). Set it to `all` (or use `autoref search --all`) to keep paging until results stop being
relevant instead: pages are fetched `stop.page_size` at a time and a page counts as low-yield when fewer than
`stop.min_yield` of its results pass your `filter.keywords`; after `stop.patience` low-yield pages in a row the query
ends (hard cap: `stop.max_results`). Without `filter.keywords` there is nothing to judge relevance by, so only the cap
applies. The search table shows each query's `total` (matches the source reports) and why it stopped. Limits of the
sources themselves: Semantic Scholar's relevance search ends at 1000 results per query, OpenAlex without a key at
10,000 (and its daily budget, see above), arXiv pages freely. To reach beyond that, add queries and use `snowball`.
Every page is committed as it arrives, so an interruption loses at most one page.

**Capping a whole run:** `search.total: 500` (or `--total 500` on `search`, `run`, `update`) stops a run once 500
distinct papers have been gathered across *all* queries and sources — handy for quick local test runs. The budget is
shared fairly: each query/source pair may take at most its share of what is left (`ceil(remaining / pairs left)`),
a pair that finds less leaves its remainder to the pairs after it, and pairs after the cap are skipped. Papers already
in the DB count too, so a rerun stops just as early as a first run. Fewer than `total` can come back when a source has
few matches or results overlap. It applies per run: the next `search` may gather more (use a smaller `total` for
`update` only if you accept missing new papers).

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
  # each entry also takes: max_tokens, temperature, timeout, max_retries, json_mode, and for openai_compat
  # extra_body (extra request fields, e.g. {chat_template_kwargs: {enable_thinking: false}} for Qwen3 on vLLM)
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

### Paper map plugin (category halos)

`autoref export obsidian` also writes `.autoref/graph.json` and copies the bundled plugin (plain JS, no build step)
to `<vault>/.obsidian/plugins/autoref-graph/`. Enable it once: *Settings → Community plugins → turn off Restricted
mode → enable "Autoref paper map"*. Open it with the orbit icon in the ribbon or the command "Open paper map".

- Papers are grouped by **primary category** (highest-confidence assignment): every top-level category is a
  translucent halo, sub-categories are denser clouds inside it, papers drift slowly within their cloud.
  Dot size = how often the paper is cited inside the vault.
- Relations stay quiet until you hover a paper: its lines light up in the relation colour with a flow from the
  citing to the cited paper. **Click a line (or a relation in the side panel) to read the LLM's analysis**: label,
  how the papers are related, how they differ, evidence sentences, confidence and model.
- Click a paper for its card (categories, abstract, cites / cited by); double-click opens its note.
  Left panel: search, hide/show categories (shift-click = show only this one), filter relation types, minimum
  confidence, lines at rest on/off, drift on/off. Esc clears the selection.
- The map reloads itself when a later export changes `graph.json`.

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
