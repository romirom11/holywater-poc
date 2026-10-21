# HOLYWATER PoC — idea → video with a verbatim line

Minimal autonomous pipeline: a 1–3 sentence idea in, an 8–30 s vertical video out, with the quoted
line spoken word for word.

Status: scaffolding. See `docs/` for the specification, architecture and implementation plan.

## Setup

```bash
uv sync
cp .env.example .env   # fill in FAL_KEY and OPENROUTER_API_KEY
```

## Development checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```
