import logging

import chromadb

from telegram_llm_kit.bot.app import create_app
from telegram_llm_kit.bot.handlers import HandlerDependencies
from telegram_llm_kit.config import Settings
from telegram_llm_kit.llm.factory import create_llm_provider
from telegram_llm_kit.rag.embeddings import EmbeddingService
from telegram_llm_kit.rag.retriever import Retriever
from telegram_llm_kit.rag.store import VectorStore
from telegram_llm_kit.storage.database import init_database
from telegram_llm_kit.storage.llm_call_repo import LLMCallRepository
from telegram_llm_kit.storage.message_repo import MessageRepository


def main() -> None:
    settings = Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Silence noisy third-party loggers
    for noisy in ("httpcore", "httpx", "chromadb", "telegram", "hpack", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Starting Telegram LLM Kit")

    # Storage
    conn = init_database(settings.sqlite_db_path)
    message_repo = MessageRepository(conn)
    llm_call_repo = LLMCallRepository(conn)

    # RAG
    embedding_service = EmbeddingService(settings.embedding_model)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    vector_store = VectorStore(chroma_client)
    retriever = Retriever(
        message_repo=message_repo,
        vector_store=vector_store,
        embedding_service=embedding_service,
        recency_count=settings.recency_count,
        semantic_count=settings.semantic_count,
    )

    # LLM
    llm_provider = create_llm_provider(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    # Wire dependencies
    deps = HandlerDependencies(
        message_repo=message_repo,
        llm_call_repo=llm_call_repo,
        llm_provider=llm_provider,
        retriever=retriever,
        vector_store=vector_store,
        embedding_service=embedding_service,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    # Build and run bot
    app = create_app(token=settings.telegram_bot_token, deps=deps)
    logger.info(
        "Bot configured: provider=%s, model=%s",
        llm_provider.provider_name,
        llm_provider.model_name,
    )
    app.run_polling()


if __name__ == "__main__":
    main()
