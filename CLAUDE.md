# Telegram LLM Kit

## Quick Reference

- `make test` — run all tests (`uv run pytest tests/ -v`)
- `make lint` — lint with ruff
- `make format` — format with ruff
- `make dev-up` — run bot in long-polling mode (requires `.env`)
- `uv sync --extra dev` — install with dev dependencies

## Architecture

Modular Python package: `bot/`, `llm/`, `rag/`, `storage/`, `prompts/`
- Single entry point: `telegram_llm_kit/main.py` wires all dependencies
- Dependency injection via `HandlerDependencies` dataclass in `bot_data`
- LLM providers use raw `httpx` (not SDKs) for full request/response logging
- SQLite FTS5 triggers auto-sync search index — no app-level sync needed
- ChromaDB receives pre-computed embeddings (not auto-generated)

## Testing Patterns

- SQLite tests: use `tmp_db` fixture from `conftest.py` (temp file, auto-cleanup)
- ChromaDB tests: use `chromadb.Client()` (ephemeral) with unique collection names per test to avoid dimension conflicts
- LLM HTTP tests: use `respx` library to mock httpx calls
- Config tests: use `monkeypatch.setenv()` + `Settings(_env_file=None)` to isolate from real `.env`
- Handler tests: mock all dependencies via `HandlerDependencies` with `MagicMock`/`AsyncMock`
- Integration tests: real SQLite + real ChromaDB, mock LLM + embeddings

## Gotchas

- ChromaDB collections lock embedding dimension on first insert — mixing 2D and 3D vectors in the same collection errors
- `pytest-asyncio` configured with `asyncio_mode = "auto"` in `pyproject.toml`
- Anthropic Messages API: system prompt must be top-level `system` param, not in `messages` array
