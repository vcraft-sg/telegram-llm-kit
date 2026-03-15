from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    id: int | None = None
    role: str = ""  # "user" or "assistant"
    content: str = ""
    telegram_message_id: int | None = None
    telegram_chat_id: int | None = None
    token_count: int | None = None
    created_at: datetime | None = None
    chroma_id: str | None = None


@dataclass
class LLMCall:
    id: int | None = None
    provider: str = ""
    model: str = ""
    request_payload: str = ""  # JSON string
    response_payload: str = ""  # JSON string
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None
    created_at: datetime | None = None
    message_id: int | None = None  # FK to messages
