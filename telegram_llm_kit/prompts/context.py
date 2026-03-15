from telegram_llm_kit.prompts.system import SYSTEM_PROMPT
from telegram_llm_kit.storage.models import Message


def build_context(
    recent_messages: list[Message],
    semantic_messages: list[Message],
    current_message: str,
) -> list[dict]:
    """Assemble the full prompt: system + deduplicated context + current message.

    Merges recent and semantic messages, deduplicates by id, and orders chronologically.
    Returns a list of message dicts ready for the LLM API.
    """
    # Deduplicate: semantic messages may overlap with recent
    seen_ids = set()
    all_messages = []
    for msg in recent_messages + semantic_messages:
        if msg.id not in seen_ids:
            seen_ids.add(msg.id)
            all_messages.append(msg)

    # Sort by id (chronological order)
    all_messages.sort(key=lambda m: m.id)

    # Build the message list for the LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in all_messages:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": current_message})

    return messages
