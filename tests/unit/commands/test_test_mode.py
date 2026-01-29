"""
Tests for test mode command functionality.

These tests verify:
1. Test mode enable/disable
2. Path switching
3. Clone and clear operations
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test and prod."""
    temp_base = tempfile.mkdtemp()
    prod_dir = Path(temp_base) / "prod"
    test_dir = Path(temp_base) / "test"

    prod_dir.mkdir()
    test_dir.mkdir()

    # Create mock prod database
    prod_db = prod_dir / "episodic.db"
    prod_db.write_text("production database content")

    # Create mock prod chroma
    prod_chroma = prod_dir / "chroma"
    prod_chroma.mkdir()
    (prod_chroma / "test_file.bin").write_bytes(b"chroma data")

    yield {
        'base': Path(temp_base),
        'prod_db': prod_db,
        'prod_chroma': prod_chroma,
        'test_dir': test_dir,
        'test_db': test_dir / "episodic.db",
        'test_chroma': test_dir / "chroma",
    }

    shutil.rmtree(temp_base)


class TestIsTestMode:
    """Tests for is_test_mode function."""

    def test_default_is_false(self):
        """Test that test mode is false by default."""
        mock_config = MagicMock()
        mock_config.get.return_value = False

        with patch('episodic.commands.test_mode.config', mock_config):
            from episodic.commands.test_mode import is_test_mode
            assert is_test_mode() is False

    def test_returns_config_value(self):
        """Test that is_test_mode returns config value."""
        mock_config = MagicMock()
        mock_config.get.return_value = True

        with patch('episodic.commands.test_mode.config', mock_config):
            from episodic.commands.test_mode import is_test_mode
            assert is_test_mode() is True


class TestCloneProduction:
    """Tests for clone production functionality."""

    def test_clone_copies_database(self, temp_dirs):
        """Test that clone copies the database file."""
        with patch('episodic.commands.test_mode.PROD_DB_PATH', temp_dirs['prod_db']), \
             patch('episodic.commands.test_mode.TEST_DB_PATH', temp_dirs['test_db']), \
             patch('episodic.commands.test_mode.PROD_CHROMA_PATH', temp_dirs['prod_chroma']), \
             patch('episodic.commands.test_mode.TEST_CHROMA_PATH', temp_dirs['test_chroma']), \
             patch('episodic.commands.test_mode.TEST_BASE_DIR', temp_dirs['test_dir']):

            from episodic.commands.test_mode import _clone_production_to_test

            # Clone
            _clone_production_to_test()

            # Verify DB was copied
            assert temp_dirs['test_db'].exists()
            assert temp_dirs['test_db'].read_text() == "production database content"

            # Verify ChromaDB was copied
            assert temp_dirs['test_chroma'].exists()
            assert (temp_dirs['test_chroma'] / "test_file.bin").exists()

    def test_clone_fails_without_prod_db(self, temp_dirs):
        """Test that clone fails gracefully if prod DB doesn't exist."""
        nonexistent = temp_dirs['base'] / "nonexistent.db"

        with patch('episodic.commands.test_mode.PROD_DB_PATH', nonexistent):
            from episodic.commands.test_mode import _clone_production_to_test

            # Should not raise
            _clone_production_to_test()


class TestClearTestEnvironment:
    """Tests for clear test environment functionality."""

    def test_clear_removes_test_dir(self, temp_dirs):
        """Test that clear removes the test directory."""
        # Create test DB
        temp_dirs['test_db'].write_text("test database")
        temp_dirs['test_chroma'].mkdir(exist_ok=True)

        mock_config = MagicMock()
        mock_config.get.return_value = False

        with patch('episodic.commands.test_mode.TEST_BASE_DIR', temp_dirs['test_dir']), \
             patch('episodic.commands.test_mode.config', mock_config), \
             patch('typer.confirm', return_value=True):

            from episodic.commands.test_mode import _clear_test_environment

            _clear_test_environment()

            # Verify test directory is gone
            assert not temp_dirs['test_dir'].exists()

    def test_clear_cancellation(self, temp_dirs):
        """Test that clear can be cancelled."""
        temp_dirs['test_db'].write_text("test database")

        with patch('episodic.commands.test_mode.TEST_BASE_DIR', temp_dirs['test_dir']), \
             patch('typer.confirm', return_value=False):

            from episodic.commands.test_mode import _clear_test_environment

            _clear_test_environment()

            # Directory should still exist
            assert temp_dirs['test_dir'].exists()


class TestEnableDisableTestMode:
    """Tests for enable/disable test mode."""

    def test_enable_sets_config(self, temp_dirs):
        """Test that enable sets config values."""
        # Create test DB
        temp_dirs['test_db'].write_text("test database")

        mock_config = MagicMock()
        mock_config.get.return_value = False

        with patch('episodic.commands.test_mode.TEST_DB_PATH', temp_dirs['test_db']), \
             patch('episodic.commands.test_mode.TEST_CHROMA_PATH', temp_dirs['test_chroma']), \
             patch('episodic.commands.test_mode.TEST_BASE_DIR', temp_dirs['test_dir']), \
             patch('episodic.commands.test_mode.config', mock_config), \
             patch('episodic.commands.test_mode._reset_connections'):

            from episodic.commands.test_mode import _enable_test_mode

            _enable_test_mode()

            # Verify config was set
            calls = mock_config.set.call_args_list
            set_keys = [call[0][0] for call in calls]
            assert 'test_mode' in set_keys
            assert 'database_path' in set_keys

    def test_enable_fails_without_test_db(self, temp_dirs):
        """Test that enable fails if test DB doesn't exist."""
        nonexistent = temp_dirs['base'] / "nonexistent.db"

        mock_config = MagicMock()

        with patch('episodic.commands.test_mode.TEST_DB_PATH', nonexistent), \
             patch('episodic.commands.test_mode.config', mock_config):

            from episodic.commands.test_mode import _enable_test_mode

            _enable_test_mode()

            # Config should NOT be set
            assert mock_config.set.call_count == 0

    def test_disable_clears_config(self, temp_dirs):
        """Test that disable clears config values."""
        mock_config = MagicMock()

        with patch('episodic.commands.test_mode.PROD_DB_PATH', temp_dirs['prod_db']), \
             patch('episodic.commands.test_mode.PROD_CHROMA_PATH', temp_dirs['prod_chroma']), \
             patch('episodic.commands.test_mode.config', mock_config), \
             patch('episodic.commands.test_mode._reset_connections'):

            from episodic.commands.test_mode import _disable_test_mode

            _disable_test_mode()

            # Verify config was updated
            calls = mock_config.set.call_args_list
            assert len(calls) >= 2

            # Check test_mode was set to False
            test_mode_call = next(c for c in calls if c[0][0] == 'test_mode')
            assert test_mode_call[0][1] is False


class TestPromptIndicator:
    """Tests for [TEST] prompt indicator."""

    def test_prompt_shows_test_indicator(self):
        """Test that prompt shows [TEST] when in test mode."""
        mock_config = MagicMock()

        def mock_get(key, default=None):
            if key == 'test_mode':
                return True
            if key == 'muse_mode':
                return False
            return default

        mock_config.get = mock_get

        with patch('episodic.cli_display.config', mock_config):
            from episodic.cli_display import get_prompt

            prompt = get_prompt()
            # The prompt is HTML - check it contains TEST
            assert '[TEST]' in str(prompt)

    def test_prompt_no_indicator_when_not_test(self):
        """Test that prompt doesn't show [TEST] in normal mode."""
        mock_config = MagicMock()

        def mock_get(key, default=None):
            if key == 'test_mode':
                return False
            if key == 'muse_mode':
                return False
            return default

        mock_config.get = mock_get

        with patch('episodic.cli_display.config', mock_config):
            from episodic.cli_display import get_prompt

            prompt = get_prompt()
            assert '[TEST]' not in str(prompt)
