"""
Tests for dream_haiku.py - Imaginary Friend

Tests:
- Trigger gates (cooldown, state, probability)
- Dialog generation (3-4 exchanges)
- Fragment decay (0.95x)
- Database integration
"""

import pytest
from pathlib import Path
from dream_haiku import (
    HaikuDreamContext,
    DreamConfig,
    init_dream,
    should_run_dream,
    run_dream_dialog,
    update_dream_fragments
)
from haiku import HaikuGenerator, SEED_WORDS


class TestHaikuDreamContext:
    """Test HaikuDreamContext dataclass."""

    def test_context_creation(self):
        """Test creating dream context."""
        ctx = HaikuDreamContext(
            last_haiku="test\nhaiku\nhere",
            dissonance=0.5,
            pulse_entropy=0.6,
            pulse_novelty=0.7,
            pulse_arousal=0.4,
            quality=0.5,
            cloud_size=500,
            turn_count=10
        )

        assert ctx.dissonance == 0.5
        assert ctx.turn_count == 10


class TestDreamConfig:
    """Test DreamConfig dataclass."""

    def test_default_config(self):
        """Test default dream configuration."""
        config = DreamConfig()

        assert config.min_interval_turns == 10
        assert config.trigger_probability == 0.25
        assert config.max_exchanges == 4
        assert config.fragment_buffer_size == 15


class TestDreamInit:
    """Test dream initialization."""

    def test_init_dream_creates_tables(self, tmp_path):
        """Test init_dream creates database tables."""
        db_path = tmp_path / "test_dream.db"

        init_dream(str(db_path))

        # Check tables exist
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'dream%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'dream_meta' in tables
        assert 'dream_fragments' in tables
        assert 'dream_dialogs' in tables
        assert 'dream_exchanges' in tables

        conn.close()

    def test_init_dream_with_bootstrap_haikus(self, tmp_path):
        """Test init_dream with bootstrap haikus."""
        db_path = tmp_path / "test_dream.db"

        bootstrap = [
            "words dance\nin cloud space\nresonance",
            "silence holds\nthe form within\nmeaning flows"
        ]

        init_dream(str(db_path), bootstrap_haikus=bootstrap)

        # Check fragments were added
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM dream_fragments")
        count = cursor.fetchone()[0]

        assert count > 0  # Should have fragments

        conn.close()


class TestDreamTriggers:
    """Test dream triggering logic."""

    def test_should_run_dream_cooldown(self, tmp_path):
        """Test cooldown prevents dreams."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        ctx = HaikuDreamContext(
            last_haiku="test\nhaiku\nhere",
            dissonance=0.5,
            pulse_entropy=0.6,
            pulse_novelty=0.8,
            pulse_arousal=0.5,
            quality=0.3,  # Low quality
            cloud_size=500,
            turn_count=5  # Too soon (< 10 turns)
        )

        config = DreamConfig(min_interval_turns=10)

        # Should not trigger due to cooldown
        result = should_run_dream(ctx, str(db_path), config)
        assert result == False

    def test_should_run_dream_low_quality(self, tmp_path):
        """Test low quality can trigger dream."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        ctx = HaikuDreamContext(
            last_haiku="test\nhaiku\nhere",
            dissonance=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.5,
            pulse_arousal=0.5,
            quality=0.3,  # Low quality (< 0.45)
            cloud_size=500,
            turn_count=15  # After cooldown
        )

        config = DreamConfig(
            min_interval_turns=10,
            trigger_probability=1.0  # Always trigger for testing
        )

        # Might trigger (depends on probability)
        result = should_run_dream(ctx, str(db_path), config)
        # With probability=1.0 and low quality, should trigger
        assert result == True

    def test_should_run_dream_high_novelty(self, tmp_path):
        """Test high novelty can trigger dream."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        ctx = HaikuDreamContext(
            last_haiku="test\nhaiku\nhere",
            dissonance=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.8,  # High novelty (> 0.7)
            pulse_arousal=0.5,
            quality=0.6,
            cloud_size=500,
            turn_count=15
        )

        config = DreamConfig(
            min_interval_turns=10,
            trigger_probability=1.0
        )

        result = should_run_dream(ctx, str(db_path), config)
        assert result == True


class TestDreamDialog:
    """Test dream dialog generation."""

    def test_run_dream_dialog(self, tmp_path):
        """Test running a dream dialog."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        generator = HaikuGenerator(SEED_WORDS[:50])
        last_haiku = "words\ndance\ncloud"

        config = DreamConfig(max_exchanges=4)

        dream_haikus = run_dream_dialog(
            generator,
            last_haiku,
            str(db_path),
            config
        )

        # Should generate some haikus
        assert len(dream_haikus) > 0
        assert len(dream_haikus) <= config.max_exchanges

    def test_dream_dialog_creates_record(self, tmp_path):
        """Test dream dialog creates database record."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        generator = HaikuGenerator(SEED_WORDS[:50])

        run_dream_dialog(generator, "test\nhaiku\nhere", str(db_path))

        # Check dialog was recorded
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM dream_dialogs")
        count = cursor.fetchone()[0]

        assert count >= 1

        conn.close()


class TestFragmentDecay:
    """Test fragment decay mechanism."""

    def test_update_dream_fragments_adds(self, tmp_path):
        """Test update_dream_fragments adds new fragments."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        haiku = "new words\ndance here\nin cloud"

        update_dream_fragments(haiku, str(db_path))

        # Check fragments were added
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM dream_fragments")
        count = cursor.fetchone()[0]

        assert count > 0

        conn.close()

    def test_update_dream_fragments_decays(self, tmp_path):
        """Test update_dream_fragments decays existing."""
        db_path = tmp_path / "test_dream.db"
        init_dream(str(db_path))

        # Add initial fragment
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dream_fragments (text, weight)
            VALUES ('test fragment', 1.0)
        """)
        conn.commit()
        conn.close()

        # Update with new fragment
        update_dream_fragments("new\nhaiku\nhere", str(db_path))

        # Check old fragment decayed
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT weight FROM dream_fragments
            WHERE text = 'test fragment'
        """)
        row = cursor.fetchone()

        if row:  # Might be pruned
            assert row[0] < 1.0  # Should be decayed (×0.95)

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
