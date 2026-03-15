from telegram_llm_kit.prompts.context import build_context
from telegram_llm_kit.prompts.system import SYSTEM_PROMPT
from telegram_llm_kit.storage.models import Message


class TestBuildContext:
    def test_basic_assembly(self):
        """Assembles system prompt + context + current message."""
        recent = [Message(id=1, role="user", content="hi")]
        result = build_context(recent, [], "hello")
        assert result[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert result[1] == {"role": "user", "content": "hi"}
        assert result[2] == {"role": "user", "content": "hello"}

    def test_deduplication(self):
        """Messages appearing in both recent and semantic are included once."""
        msg = Message(id=1, role="user", content="shared message")
        result = build_context([msg], [msg], "current")
        # system + 1 deduplicated message + current = 3
        assert len(result) == 3

    def test_chronological_order(self):
        """Messages are ordered by id regardless of source."""
        recent = [Message(id=3, role="user", content="recent")]
        semantic = [Message(id=1, role="user", content="old but relevant")]
        result = build_context(recent, semantic, "now")
        # After system prompt, semantic (id=1) comes before recent (id=3)
        assert result[1]["content"] == "old but relevant"
        assert result[2]["content"] == "recent"
        assert result[3]["content"] == "now"

    def test_empty_context(self):
        """Works with no context messages at all."""
        result = build_context([], [], "first message")
        assert len(result) == 2  # system + current
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "first message"

    def test_preserves_roles(self):
        """Both user and assistant messages preserve their roles."""
        messages = [
            Message(id=1, role="user", content="question"),
            Message(id=2, role="assistant", content="answer"),
        ]
        result = build_context(messages, [], "follow-up")
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
