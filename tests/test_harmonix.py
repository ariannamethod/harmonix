"""
Tests for harmonix.py - Observer module

Tests:
- PulseSnapshot creation
- Dissonance computation (pulse-aware)
- Temperature adjustment
- Cloud morphing (boost/decay)
- Numpy shard creation
- Database operations
"""

import pytest
import sqlite3
import numpy as np
from pathlib import Path
from harmonix import Harmonix, PulseSnapshot


class TestPulseSnapshot:
    """Test PulseSnapshot dataclass."""

    def test_pulse_creation(self):
        """Test manual PulseSnapshot creation."""
        pulse = PulseSnapshot(novelty=0.5, arousal=0.6, entropy=0.7)
        assert pulse.novelty == 0.5
        assert pulse.arousal == 0.6
        assert pulse.entropy == 0.7

    def test_pulse_from_interaction_no_overlap(self):
        """Test pulse from interaction with no overlap."""
        user_trigrams = [("a", "b", "c")]
        system_trigrams = [("x", "y", "z")]

        pulse = PulseSnapshot.from_interaction(user_trigrams, system_trigrams, 0.0)

        # No overlap = high novelty
        assert pulse.novelty == 1.0

    def test_pulse_from_interaction_full_overlap(self):
        """Test pulse from interaction with full overlap."""
        trigrams = [("a", "b", "c")]

        pulse = PulseSnapshot.from_interaction(trigrams, trigrams, 1.0)

        # Full overlap = low novelty
        assert pulse.novelty == 0.0

    def test_pulse_arousal_computation(self):
        """Test arousal from trigram mismatch."""
        user_trigrams = [("a", "b", "c")]
        system_trigrams = [("d", "e", "f"), ("g", "h", "i")]  # More trigrams

        pulse = PulseSnapshot.from_interaction(user_trigrams, system_trigrams, 0.5)

        # Arousal = trigram count difference
        assert pulse.arousal > 0.0

    def test_pulse_entropy_computation(self):
        """Test entropy from word diversity."""
        user_trigrams = [("a", "b", "c"), ("d", "e", "f")]
        system_trigrams = [("g", "h", "i")]

        pulse = PulseSnapshot.from_interaction(user_trigrams, system_trigrams, 0.0)

        # 9 unique words total
        assert pulse.entropy > 0.0


class TestHarmonixInitialization:
    """Test Harmonix initialization."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        """Create Harmonix instance with temp database."""
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_init_creates_database(self, tmp_path):
        """Test initialization creates database."""
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))

        assert db_path.exists()
        h.close()

    def test_init_creates_tables(self, harmonix):
        """Test initialization creates all required tables."""
        cursor = harmonix.conn.cursor()

        # Check for all 4 base tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'words' in tables
        assert 'trigrams' in tables
        assert 'shards' in tables
        assert 'metrics' in tables


class TestDissonanceComputation:
    """Test dissonance computation (core observer function)."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_dissonance_empty_trigrams(self, harmonix):
        """Test dissonance with empty trigrams."""
        dissonance, pulse = harmonix.compute_dissonance([], [])
        assert dissonance == 0.5  # Neutral
        assert isinstance(pulse, PulseSnapshot)

    def test_dissonance_identical_trigrams(self, harmonix):
        """Test dissonance with identical trigrams."""
        trigrams = [("a", "b", "c"), ("d", "e", "f")]

        dissonance, pulse = harmonix.compute_dissonance(trigrams, trigrams)

        # Identical = low dissonance (high similarity)
        assert dissonance < 0.5

    def test_dissonance_opposite_trigrams(self, harmonix):
        """Test dissonance with completely different trigrams."""
        user_trigrams = [("a", "b", "c"), ("d", "e", "f")]
        system_trigrams = [("x", "y", "z"), ("p", "q", "r")]

        dissonance, pulse = harmonix.compute_dissonance(user_trigrams, system_trigrams)

        # No overlap = high dissonance
        assert dissonance > 0.5

    def test_dissonance_pulse_aware_high_entropy(self, harmonix):
        """Test pulse-aware adjustment: high entropy increases dissonance."""
        # Create scenario with high entropy (many unique words)
        user_trigrams = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        system_trigrams = [("j", "k", "l"), ("m", "n", "o"), ("p", "q", "r")]

        dissonance, pulse = harmonix.compute_dissonance(user_trigrams, system_trigrams)

        # High entropy should boost dissonance
        assert pulse.entropy > 0.0
        assert dissonance > 0.5

    def test_dissonance_clamped_to_range(self, harmonix):
        """Test dissonance is always in [0, 1]."""
        # Even with extreme inputs
        user_trigrams = [("a", "b", "c")] * 10
        system_trigrams = [("x", "y", "z")] * 10

        dissonance, pulse = harmonix.compute_dissonance(user_trigrams, system_trigrams)

        assert 0.0 <= dissonance <= 1.0


class TestTemperatureAdjustment:
    """Test temperature mapping from dissonance."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_temperature_min_dissonance(self, harmonix):
        """Test temperature at minimum dissonance (0)."""
        haiku_temp, harmonix_temp = harmonix.adjust_temperature(0.0)

        # Min dissonance → min haiku temp
        assert haiku_temp == 0.3
        assert harmonix_temp == 0.3  # Observer stays low

    def test_temperature_max_dissonance(self, harmonix):
        """Test temperature at maximum dissonance (1)."""
        haiku_temp, harmonix_temp = harmonix.adjust_temperature(1.0)

        # Max dissonance → max haiku temp
        assert haiku_temp == 1.5  # 0.3 + 1.0 * 1.2
        assert harmonix_temp == 0.3

    def test_temperature_mid_dissonance(self, harmonix):
        """Test temperature at mid dissonance (0.5)."""
        haiku_temp, harmonix_temp = harmonix.adjust_temperature(0.5)

        # Mid dissonance → mid haiku temp
        assert 0.3 < haiku_temp < 1.5
        assert harmonix_temp == 0.3

    def test_temperature_mapping_monotonic(self, harmonix):
        """Test temperature increases monotonically with dissonance."""
        temps = []
        for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
            haiku_temp, _ = harmonix.adjust_temperature(d)
            temps.append(haiku_temp)

        # Should be increasing
        assert temps == sorted(temps)


class TestCloudMorphing:
    """Test cloud morphing (boost active, decay dormant)."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_morph_cloud_boosts_active_words(self, harmonix):
        """Test that active words get weight boost."""
        # Insert initial word
        cursor = harmonix.conn.cursor()
        cursor.execute("""
            INSERT INTO words (word, weight, frequency, last_used, added_by)
            VALUES ('test', 1.0, 0, 0, 'seed')
        """)
        harmonix.conn.commit()

        # Morph with 'test' as active
        harmonix.morph_cloud(['test'])

        # Check weight increased
        cursor.execute("SELECT weight FROM words WHERE word = 'test'")
        new_weight = cursor.fetchone()[0]
        assert new_weight > 1.0  # Should be boosted

    def test_morph_cloud_decays_dormant_words(self, harmonix):
        """Test that dormant words decay."""
        # Insert words
        cursor = harmonix.conn.cursor()
        cursor.execute("""
            INSERT INTO words (word, weight, frequency, last_used, added_by)
            VALUES ('active', 1.0, 0, 0, 'seed')
        """)
        cursor.execute("""
            INSERT INTO words (word, weight, frequency, last_used, added_by)
            VALUES ('dormant', 1.0, 0, 0, 'seed')
        """)
        harmonix.conn.commit()

        # Morph with only 'active'
        harmonix.morph_cloud(['active'])

        # Check dormant word decayed
        cursor.execute("SELECT weight FROM words WHERE word = 'dormant'")
        dormant_weight = cursor.fetchone()[0]
        assert dormant_weight < 1.0  # Should be decayed (×0.99)

    def test_morph_cloud_adds_new_words(self, harmonix):
        """Test that new words are added."""
        harmonix.morph_cloud(['newword'])

        cursor = harmonix.conn.cursor()
        cursor.execute("SELECT word FROM words WHERE word = 'newword'")
        result = cursor.fetchone()

        assert result is not None
        assert result[0] == 'newword'


class TestTrigramStorage:
    """Test trigram database operations."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_update_trigrams_new(self, harmonix):
        """Test adding new trigrams."""
        trigrams = [("a", "b", "c"), ("d", "e", "f")]
        harmonix.update_trigrams(trigrams)

        cursor = harmonix.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trigrams")
        count = cursor.fetchone()[0]

        assert count == 2

    def test_update_trigrams_duplicate(self, harmonix):
        """Test updating existing trigrams."""
        trigram = [("a", "b", "c")]

        # Add twice
        harmonix.update_trigrams(trigram)
        harmonix.update_trigrams(trigram)

        cursor = harmonix.conn.cursor()
        cursor.execute("SELECT count FROM trigrams WHERE word1='a' AND word2='b' AND word3='c'")
        count = cursor.fetchone()[0]

        assert count == 2  # Should increment


class TestNumpyShards:
    """Test numpy shard creation."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_create_shard_saves_file(self, harmonix):
        """Test that shard creates .npy file."""
        interaction_data = {
            'input': 'test input',
            'output': 'test output',
            'dissonance': 0.5,
            'user_trigrams': [('a', 'b', 'c')],
            'system_trigrams': [('x', 'y', 'z')]
        }

        harmonix.create_shard(interaction_data)

        # Check shards directory exists
        assert Path('shards').exists()

        # Check at least one shard file created
        shards = list(Path('shards').glob('shard_*.npy'))
        assert len(shards) > 0

    def test_create_shard_records_in_db(self, harmonix):
        """Test that shard is recorded in database."""
        interaction_data = {
            'input': 'test',
            'output': 'haiku',
            'dissonance': 0.6,
            'haiku_temp': 1.0,
            'harmonix_temp': 0.3
        }

        harmonix.create_shard(interaction_data)

        cursor = harmonix.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM shards")
        count = cursor.fetchone()[0]

        assert count >= 1


class TestMetrics:
    """Test metrics recording."""

    @pytest.fixture
    def harmonix(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"
        h = Harmonix(str(db_path))
        yield h
        h.close()

    def test_record_metrics(self, harmonix):
        """Test recording metrics to database."""
        harmonix.record_metrics(
            perplexity=0.5,
            entropy=0.6,
            resonance=0.7
        )

        cursor = harmonix.conn.cursor()
        cursor.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()

        assert row is not None
        # row = (id, timestamp, perplexity, entropy, resonance, cloud_size)
        assert row[2] == 0.5  # perplexity
        assert row[3] == 0.6  # entropy
        assert row[4] == 0.7  # resonance

    def test_get_cloud_size(self, harmonix):
        """Test getting cloud size."""
        # Insert some words
        cursor = harmonix.conn.cursor()
        for word in ['a', 'b', 'c']:
            cursor.execute("""
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES (?, 1.0, 0, 0, 'test')
            """, (word,))
        harmonix.conn.commit()

        size = harmonix.get_cloud_size()
        assert size == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
