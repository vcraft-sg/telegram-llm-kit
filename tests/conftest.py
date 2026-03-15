import os
import tempfile

import pytest

from telegram_llm_kit.storage.database import init_database


@pytest.fixture
def tmp_db():
    """Provide a temporary SQLite database connection, cleaned up after test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = init_database(path)
    yield conn
    conn.close()
    os.unlink(path)
