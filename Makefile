# Telegram LLM Kit — dev targets

.PHONY: dev-up test lint format

# Run the bot in long-polling mode (foreground)
dev-up:
	uv run python -m telegram_llm_kit.main

# Run all unit tests
test:
	uv run pytest tests/ -v

# Lint with ruff
lint:
	uv run ruff check .

# Format with ruff
format:
	uv run ruff format .
