import pytest

from telegram_llm_kit.storage.message_repo import MessageRepository
from telegram_llm_kit.storage.models import Message


class TestMessageRepository:
    @pytest.fixture
    def repo(self, tmp_db):
        return MessageRepository(tmp_db)

    def test_save_and_retrieve(self, repo):
        """Saving a message assigns id and created_at."""
        msg = Message(role="user", content="hello world")
        saved = repo.save(msg)
        assert saved.id is not None
        assert saved.created_at is not None

    def test_get_recent(self, repo):
        """get_recent returns messages in chronological order."""
        for i in range(5):
            repo.save(Message(role="user", content=f"msg {i}"))
        recent = repo.get_recent(limit=3)
        assert len(recent) == 3
        # Should be oldest first (msg 2, msg 3, msg 4)
        assert recent[0].content == "msg 2"
        assert recent[2].content == "msg 4"

    def test_get_by_ids(self, repo):
        """get_by_ids returns only requested messages, ordered by id."""
        ids = []
        for i in range(5):
            msg = repo.save(Message(role="user", content=f"msg {i}"))
            ids.append(msg.id)
        result = repo.get_by_ids([ids[0], ids[3]])
        assert len(result) == 2
        assert result[0].content == "msg 0"
        assert result[1].content == "msg 3"

    def test_get_by_ids_empty(self, repo):
        """get_by_ids with empty list returns empty list."""
        assert repo.get_by_ids([]) == []

    def test_search_fts(self, repo):
        """FTS5 search finds messages by content."""
        repo.save(Message(role="user", content="the quick brown fox"))
        repo.save(Message(role="user", content="lazy dog sleeps"))
        repo.save(Message(role="assistant", content="the fox jumps over"))
        results = repo.search_fts("fox")
        assert len(results) == 2
        contents = [r.content for r in results]
        assert "the quick brown fox" in contents
        assert "the fox jumps over" in contents

    def test_search_fts_no_results(self, repo):
        """FTS5 search returns empty list when nothing matches."""
        repo.save(Message(role="user", content="hello world"))
        results = repo.search_fts("nonexistent")
        assert results == []

    def test_update_chroma_id(self, repo):
        """update_chroma_id sets the chroma_id field."""
        msg = repo.save(Message(role="user", content="test"))
        assert msg.chroma_id is None
        repo.update_chroma_id(msg.id, "chroma-123")
        updated = repo.get_by_ids([msg.id])
        assert updated[0].chroma_id == "chroma-123"

    def test_fts_trigger_on_update(self, repo):
        """FTS5 index updates when message content is updated."""
        msg = repo.save(Message(role="user", content="original text"))
        # Manually update content to test the trigger
        repo.conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("updated text", msg.id),
        )
        repo.conn.commit()
        assert repo.search_fts("original") == []
        results = repo.search_fts("updated")
        assert len(results) == 1

    def test_fts_trigger_on_delete(self, repo):
        """FTS5 index removes entry when message is deleted."""
        msg = repo.save(Message(role="user", content="deleteme"))
        assert len(repo.search_fts("deleteme")) == 1
        repo.conn.execute("DELETE FROM messages WHERE id = ?", (msg.id,))
        repo.conn.commit()
        assert repo.search_fts("deleteme") == []
