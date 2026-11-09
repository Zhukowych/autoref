# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`autoref`: literature-review CLI (Typer). Harvests papers from Semantic Scholar / OpenAlex / arXiv into SQLite,
screens relevance, LLM-categorizes into a user-defined category tree, LLM-labels citation edges, exports an
Obsidian vault. Topic-agnostic: one YAML config per topic (`configs/*.yaml`; `configs/mde_uav_mapping.yaml` is the
thesis topic). README.md documents every CLI command and config key in detail — read it before changing CLI behavior.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                   # all tests (offline: fake LLM + respx-mocked HTTP)
.venv/bin/pytest tests/test_pipeline.py::test_name # single test
.venv/bin/autoref -v run -c configs/mde_uav_mapping.yaml   # full pipeline; -v = debug logs
.venv/bin/autoref llm test -c ... / sources test -c ...    # check backends
```

No linter/formatter configured. `AUTOREF_CONFIG` env var replaces `-c`.

## Architecture

- **Flow**: `cli.py` → `pipeline/*` steps (`search → snowball → enrich → categorize → relate → export`, `update`
  = incremental run over a date window, guarded by a lock file and recorded in `sync_runs`). Every step must stay
  **idempotent/resumable**: it selects only unprocessed rows; `--force` redoes. Most commands support `--dry-run`.
- **`pipeline/context.py`**: `Ctx` holds config, session factory, lazily-built `SourceSet`. Use the `session(ctx)`
  context manager — it commits, or rolls back when `ctx.dry_run`. `ctx.prompt(name)` loads
  `prompts/<name>.txt` split on `=== SYSTEM ===` / `=== USER ===`, `string.Template` `$vars`; a project-level
  `<base_dir>/prompts/<name>.txt` overrides the packaged one.
- **Config (`config.py`)**: pydantic models, `${VAR}` / `${VAR:-default}` env expansion. `base_dir` = parent of
  `configs/` (relative `db`/`vault` resolve against it). LLM config is per task (`categorize`, `relations`, `fill`,
  `relevance`) falling back to `default`.
- **Scheme ids**: `categorize.scheme_id` / `relations.scheme_id` = explicit `scheme` or hash of the tree/labels.
  Rows stamped with a different scheme are "stale" and reprocessed on next run. Changing the hashed inputs
  therefore triggers recategorization/relabeling of everything.
- **Sources (`sources/`)**: subclass `PaperSource`; declare `capabilities` (search, get, references, citations,
  contexts, …), return source-agnostic `PaperRecord`/`ReferenceRecord` (`models.py`); register in
  `sources/registry.py`. Base class (`_get`) handles politeness delay (adaptive: doubles after 429, decays on success) and its own retry loop (exponential backoff + jitter, honors `Retry-After`); `SourceSet` disables a source after
  repeated failures (circuit breaker).
- **LLMs (`llm/`)**: subclass `LLMClient`, implement only `_chat(system, user, *, json_schema=None)`, register in
  `llm/registry.py` (`PROVIDERS`, clients cached in `_cache`). `complete_json` does shared JSON extraction
  (fences, `<think>` blocks), pydantic validation, and retry-with-error-feedback. `FailureGuard` aborts a batch
  step after N consecutive LLM failures.
- **DB (`db/`)**: SQLAlchemy 2.0 on SQLite (WAL, FKs on). Papers are global; per-topic state lives in
  `topic_papers` (status, relevance, depth, core, schemes) so one DB serves multiple topics. `db/repo.py`:
  dedup order DOI → arXiv → S2 → OpenAlex → normalized title(+year ±1); `merge_record` only fills empty fields and
  writes `field_provenance`. Unknown cited papers are stored as stubs (`is_stub`). Migrations: only
  `_add_missing_columns` (auto `ALTER TABLE ADD COLUMN` for new model columns) — no Alembic; avoid renames/type
  changes.
- **Enrichment rule**: LLM may fill fields only from fetched text (PDF/abstract), never invent from a title;
  LLM-filled fields are flagged with provenance.
- **Obsidian export (`obsidian/export.py`)**: rewrites only changed notes; content below `NOTES_MARKER` under
  `## Notes` is user-owned and must be preserved (legacy `liblib:` marker also recognized).

## Tests

`tests/conftest.py` registers a `FakeLLM` provider (`"fake"`) that answers based on the requested JSON schema's
properties, builds a temp config from `src/autoref/templates/mde.yaml`, and provides `cfg_path` / `ctx` fixtures.
HTTP is mocked with `respx`. New LLM-backed steps need a matching branch in `FakeLLM._chat`.
