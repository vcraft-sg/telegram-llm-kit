import os
import tempfile

from telegram_llm_kit.storage.database import init_database


class TestDatabase:
    def test_creates_tables(self, tmp_db):
        """Schema creates messages and llm_calls tables."""
        tables = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        assert "messages" in table_names
        assert "llm_calls" in table_names
        assert "messages_fts" in table_names

    def test_creates_fts_triggers(self, tmp_db):
        """FTS5 triggers are created for insert, update, delete."""
        triggers = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
        trigger_names = [r["name"] for r in triggers]
        assert "messages_ai" in trigger_names
        assert "messages_ad" in trigger_names
        assert "messages_au" in trigger_names

    def test_creates_parent_directory(self):
        """init_database creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "test.db")
            conn = init_database(db_path)
            conn.close()
            assert os.path.exists(db_path)

    def test_idempotent_init(self, tmp_db):
        """Running init_database twice on the same DB doesn't error."""
        # The fixture already initialized; inserting data then re-initializing
        tmp_db.execute(
            "INSERT INTO messages (role, content) VALUES ('user', 'hello')"
        )
        tmp_db.commit()
        # Re-execute schema (CREATE IF NOT EXISTS)
        from telegram_llm_kit.storage.database import SCHEMA_SQL
        tmp_db.executescript(SCHEMA_SQL)
        count = tmp_db.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]
        assert count == 1
