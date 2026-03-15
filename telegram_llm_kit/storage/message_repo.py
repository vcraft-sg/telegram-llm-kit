import sqlite3

from telegram_llm_kit.storage.models import Message


class MessageRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, message: Message) -> Message:
        """Insert a message and return it with the generated id and created_at."""
        cursor = self.conn.execute(
            """INSERT INTO messages (role, content, telegram_message_id, telegram_chat_id,
                                     token_count, chroma_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                message.role,
                message.content,
                message.telegram_message_id,
                message.telegram_chat_id,
                message.token_count,
                message.chroma_id,
            ),
        )
        self.conn.commit()
        message.id = cursor.lastrowid
        row = self.conn.execute(
            "SELECT created_at FROM messages WHERE id = ?", (message.id,)
        ).fetchone()
        message.created_at = row["created_at"]
        return message

    def update_chroma_id(self, message_id: int, chroma_id: str) -> None:
        """Set the chroma_id after embedding."""
        self.conn.execute(
            "UPDATE messages SET chroma_id = ? WHERE id = ?", (chroma_id, message_id)
        )
        self.conn.commit()

    def get_recent(self, limit: int = 20) -> list[Message]:
        """Return the most recent messages, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    def get_by_ids(self, ids: list[int]) -> list[Message]:
        """Return messages by their IDs, ordered by id."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders}) ORDER BY id", ids
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def search_fts(self, query: str, limit: int = 10) -> list[Message]:
        """Full-text search using FTS5. Returns matching messages ordered by rank."""
        rows = self.conn.execute(
            """SELECT m.* FROM messages m
               JOIN messages_fts fts ON m.id = fts.rowid
               WHERE messages_fts MATCH ?
               ORDER BY fts.rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            telegram_message_id=row["telegram_message_id"],
            telegram_chat_id=row["telegram_chat_id"],
            token_count=row["token_count"],
            created_at=row["created_at"],
            chroma_id=row["chroma_id"],
        )
