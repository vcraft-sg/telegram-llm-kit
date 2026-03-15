# Telegram LLM Kit

A reference implementation for LLM + RAG development, using Telegram as a convenient UI layer.

## What It Does

- Connects a Telegram bot to an LLM (DeepSeek or Claude)
- Automatically indexes all conversation messages into ChromaDB for semantic search
- Provides full-text search (SQLite FTS5) and semantic search (vector similarity)
- Logs every LLM call with full request/response payloads and token counts
- Builds context from recent messages + semantically relevant older messages

## Architecture

```
Telegram → Bot Handlers → Retriever (SQLite + ChromaDB) → Prompt Assembly → LLM Provider → Response
```

### Modules

| Module | Purpose |
|--------|---------|
| `bot/` | Telegram handlers and app setup (long polling) |
| `llm/` | LLM provider abstraction (DeepSeek, Claude) via raw httpx |
| `rag/` | Embeddings (SentenceTransformer), ChromaDB store, retriever |
| `storage/` | SQLite database, message and LLM call repositories |
| `prompts/` | System prompt template and context assembly |

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Install dependencies: `uv sync`
3. Run the bot: `make dev-up`

## Development

```bash
make test     # Run all tests
make lint     # Lint with ruff
make format   # Format with ruff
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram Bot API token |
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_PROVIDER` | Yes | — | `deepseek` or `claude` |
| `LLM_MODEL` | No | Provider default | Model name override |
| `LLM_TEMPERATURE` | No | `0.7` | Sampling temperature |
| `LLM_MAX_TOKENS` | No | `1024` | Max tokens in LLM response |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `RECENCY_COUNT` | No | `20` | Number of recent messages for context |
| `SEMANTIC_COUNT` | No | `10` | Number of semantically similar messages |
| `SQLITE_DB_PATH` | No | `data/bot.db` | SQLite database path |
| `CHROMA_PERSIST_DIR` | No | `data/chroma` | ChromaDB persistence directory |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Message Flow

1. User sends message via Telegram
2. Message saved to SQLite (auto-indexed by FTS5 trigger)
3. Message embedded via SentenceTransformer → stored in ChromaDB
4. Context retrieved: last N messages + top K semantically similar (deduplicated)
5. Prompt assembled: system prompt + context + current message
6. LLM called, response saved to SQLite + ChromaDB
7. Full LLM call logged (request, response, tokens, latency)
8. Response sent back via Telegram

## Search

Send `/search <query>` to search conversation history using both full-text and semantic search.
