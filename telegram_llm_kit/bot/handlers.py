import json
import logging
from dataclasses import dataclass

from telegram import Update
from telegram.ext import ContextTypes

from telegram_llm_kit.llm.base import LLMProvider
from telegram_llm_kit.prompts.context import build_context
from telegram_llm_kit.rag.embeddings import EmbeddingService
from telegram_llm_kit.rag.retriever import Retriever
from telegram_llm_kit.rag.store import VectorStore
from telegram_llm_kit.storage.llm_call_repo import LLMCallRepository
from telegram_llm_kit.storage.message_repo import MessageRepository
from telegram_llm_kit.storage.models import LLMCall, Message

logger = logging.getLogger(__name__)


@dataclass
class HandlerDependencies:
    message_repo: MessageRepository
    llm_call_repo: LLMCallRepository
    llm_provider: LLMProvider
    retriever: Retriever
    vector_store: VectorStore
    embedding_service: EmbeddingService
    temperature: float = 0.7
    max_tokens: int = 1024


def _get_deps(context: ContextTypes.DEFAULT_TYPE) -> HandlerDependencies:
    return context.bot_data["deps"]


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Hello! I'm your AI companion. Send me a message and I'll do my best to help. "
        "Use /search <query> to search our conversation history."
    )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages — the core message flow."""
    deps = _get_deps(context)
    user_text = update.message.text
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    logger.info("Incoming message from chat_id=%s: %s", chat_id, user_text[:100])

    # 1. Save user message to SQLite
    user_msg = deps.message_repo.save(
        Message(
            role="user",
            content=user_text,
            telegram_message_id=message_id,
            telegram_chat_id=chat_id,
        )
    )
    logger.info("Saved user message id=%s, FTS5 indexed via trigger", user_msg.id)

    # 2. Embed and store in ChromaDB
    embedding = deps.embedding_service.embed(user_text)
    chroma_id = f"msg-{user_msg.id}"
    deps.vector_store.add(doc_id=chroma_id, text=user_text, embedding=embedding)
    deps.message_repo.update_chroma_id(user_msg.id, chroma_id)
    logger.info("Embedded user message in ChromaDB as %s", chroma_id)

    # 3. Retrieve context
    recent, semantic = deps.retriever.retrieve(user_text)
    logger.info("Retrieved context: %d recent, %d semantic messages", len(recent), len(semantic))

    # 4. Build prompt
    messages = build_context(recent, semantic, user_text)

    # 5. Call LLM
    try:
        llm_response = await deps.llm_provider.complete(
            messages=messages,
            temperature=deps.temperature,
            max_tokens=deps.max_tokens,
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        # Log the failed call
        deps.llm_call_repo.save(
            LLMCall(
                provider=deps.llm_provider.provider_name,
                model=deps.llm_provider.model_name,
                request_payload=json.dumps({"messages": messages}),
                response_payload="",
                error=str(e),
                message_id=user_msg.id,
            )
        )
        await update.message.reply_text("Sorry, I encountered an error. Please try again.")
        return

    logger.info(
        "LLM response: model=%s, in=%d tok, out=%d tok, latency=%dms",
        llm_response.model,
        llm_response.input_tokens,
        llm_response.output_tokens,
        llm_response.latency_ms,
    )

    # 6. Save assistant response
    assistant_msg = deps.message_repo.save(
        Message(
            role="assistant",
            content=llm_response.content,
            telegram_chat_id=chat_id,
            token_count=llm_response.output_tokens,
        )
    )

    # 7. Embed assistant response in ChromaDB
    assistant_embedding = deps.embedding_service.embed(llm_response.content)
    assistant_chroma_id = f"msg-{assistant_msg.id}"
    deps.vector_store.add(
        doc_id=assistant_chroma_id, text=llm_response.content, embedding=assistant_embedding
    )
    deps.message_repo.update_chroma_id(assistant_msg.id, assistant_chroma_id)
    logger.info(
        "Saved assistant message id=%s, embedded as %s", assistant_msg.id, assistant_chroma_id
    )

    # 8. Log the LLM call
    deps.llm_call_repo.save(
        LLMCall(
            provider=llm_response.model,
            model=llm_response.model,
            request_payload=json.dumps(llm_response.raw_request),
            response_payload=json.dumps(llm_response.raw_response),
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            latency_ms=llm_response.latency_ms,
            message_id=assistant_msg.id,
        )
    )

    # 9. Reply
    await update.message.reply_text(llm_response.content)


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command — combined FTS5 + semantic search."""
    deps = _get_deps(context)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /search <query>")
        return

    logger.info("Search query: %s", query)

    # Full-text search
    fts_results = deps.message_repo.search_fts(query, limit=5)

    # Semantic search
    query_embedding = deps.embedding_service.embed(query)
    semantic_results = deps.vector_store.query(embedding=query_embedding, n_results=5)

    logger.info("Search results: %d FTS, %d semantic", len(fts_results), len(semantic_results))

    # Format results
    lines = []
    if fts_results:
        lines.append("📝 Text search results:")
        for msg in fts_results:
            preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
            lines.append(f"  [{msg.role}] {preview}")

    if semantic_results:
        lines.append("\n🔍 Semantic search results:")
        for result in semantic_results:
            preview = result["text"][:100] + ("..." if len(result["text"]) > 100 else "")
            lines.append(f"  {preview} (similarity: {1 - result['distance']:.2f})")

    if not lines:
        lines.append("No results found.")

    await update.message.reply_text("\n".join(lines))
