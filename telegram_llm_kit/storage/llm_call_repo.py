import sqlite3

from telegram_llm_kit.storage.models import LLMCall


class LLMCallRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, call: LLMCall) -> LLMCall:
        """Insert an LLM call record and return it with the generated id."""
        cursor = self.conn.execute(
            """INSERT INTO llm_calls (provider, model, request_payload, response_payload,
                                      input_tokens, output_tokens, latency_ms, error, message_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                call.provider,
                call.model,
                call.request_payload,
                call.response_payload,
                call.input_tokens,
                call.output_tokens,
                call.latency_ms,
                call.error,
                call.message_id,
            ),
        )
        self.conn.commit()
        call.id = cursor.lastrowid
        row = self.conn.execute(
            "SELECT created_at FROM llm_calls WHERE id = ?", (call.id,)
        ).fetchone()
        call.created_at = row["created_at"]
        return call

    def get_by_message_id(self, message_id: int) -> LLMCall | None:
        """Return the LLM call associated with a message, if any."""
        row = self.conn.execute(
            "SELECT * FROM llm_calls WHERE message_id = ?", (message_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_call(row)

    def _row_to_call(self, row: sqlite3.Row) -> LLMCall:
        return LLMCall(
            id=row["id"],
            provider=row["provider"],
            model=row["model"],
            request_payload=row["request_payload"],
            response_payload=row["response_payload"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            latency_ms=row["latency_ms"],
            error=row["error"],
            created_at=row["created_at"],
            message_id=row["message_id"],
        )
