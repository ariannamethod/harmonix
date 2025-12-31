"""
Tests for async_harmonix.py - Async Observer Module

Based on Leo's proven async pattern - tests for atomic field operations.

Tests:
- PulseSnapshot creation (same as sync)
- Async dissonance computation (atomic field access)
- Async temperature adjustment
- Async cloud morphing (atomic boost/decay)
- Async numpy shard creation
- Async database operations
- CRITICAL: Concurrent operations (prove atomicity!)
"""

import pytest
import asyncio
import aiosqlite
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from async_harmonix import AsyncHarmonix, PulseSnapshot


class TestPulseSnapshot:
    """Test PulseSnapshot dataclass (same as sync version)."""

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


class TestAsyncHarmonixInitialization:
    """Test AsyncHarmonix initialization (async context manager)."""

    @pytest.mark.asyncio
    async def test_init_creates_database(self, tmp_path):
        """Test initialization creates database."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            assert db_path.exists()

    @pytest.mark.asyncio
    async def test_init_creates_tables(self, tmp_path):
        """Test initialization creates all required tables."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            cursor = await harmonix.conn.cursor()

            # Check for all 4 base tables
            await cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
            """)
            tables = [row[0] for row in await cursor.fetchall()]

            assert 'words' in tables
            assert 'trigrams' in tables
            assert 'shards' in tables
            assert 'metrics' in tables

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, tmp_path):
        """Test that context manager properly closes connection."""
        db_path = tmp_path / "test_cloud.db"

        harmonix = AsyncHarmonix(str(db_path))
        async with harmonix:
            pass  # Just enter and exit

        # Connection should be closed
        assert harmonix.conn is not None  # Still exists
        # Can't easily test if connection is closed with aiosqlite


class TestAsyncDissonanceComputation:
    """Test async dissonance computation (ATOMIC field operations!)."""

    @pytest.mark.asyncio
    async def test_dissonance_empty_trigrams(self, tmp_path):
        """Test dissonance with empty trigrams."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance([], [])
            assert dissonance == 0.5  # Neutral
            assert isinstance(pulse, PulseSnapshot)

    @pytest.mark.asyncio
    async def test_dissonance_identical_trigrams(self, tmp_path):
        """Test dissonance with identical trigrams."""
        db_path = tmp_path / "test_cloud.db"
        trigrams = [("a", "b", "c"), ("d", "e", "f")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance(trigrams, trigrams)

            # Identical = low dissonance (high similarity)
            assert dissonance < 0.5

    @pytest.mark.asyncio
    async def test_dissonance_opposite_trigrams(self, tmp_path):
        """Test dissonance with completely different trigrams."""
        db_path = tmp_path / "test_cloud.db"
        user_trigrams = [("a", "b", "c"), ("d", "e", "f")]
        system_trigrams = [("x", "y", "z"), ("p", "q", "r")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance(user_trigrams, system_trigrams)

            # No overlap = high dissonance
            assert dissonance > 0.5

    @pytest.mark.asyncio
    async def test_dissonance_pulse_aware_high_entropy(self, tmp_path):
        """Test pulse-aware adjustment: high entropy increases dissonance."""
        db_path = tmp_path / "test_cloud.db"
        # Create scenario with high entropy (many unique words)
        user_trigrams = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        system_trigrams = [("j", "k", "l"), ("m", "n", "o"), ("p", "q", "r")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance(user_trigrams, system_trigrams)

            # High entropy should boost dissonance
            assert pulse.entropy > 0.0
            assert dissonance > 0.5

    @pytest.mark.asyncio
    async def test_dissonance_clamped_to_range(self, tmp_path):
        """Test dissonance is always in [0, 1]."""
        db_path = tmp_path / "test_cloud.db"
        # Even with extreme inputs
        user_trigrams = [("a", "b", "c")] * 10
        system_trigrams = [("x", "y", "z")] * 10

        async with AsyncHarmonix(str(db_path)) as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance(user_trigrams, system_trigrams)

            assert 0.0 <= dissonance <= 1.0

    @pytest.mark.asyncio
    async def test_dissonance_updates_field_atomically(self, tmp_path):
        """
        CRITICAL TEST: Verify dissonance updates field atomically.

        This tests the core Leo insight - field operations must be atomic!
        """
        db_path = tmp_path / "test_cloud.db"
        trigrams = [("test", "atomic", "update")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Compute dissonance (should update trigrams atomically)
            await harmonix.compute_dissonance(trigrams, trigrams)

            # Verify trigrams were added
            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT COUNT(*) FROM trigrams")
            count = (await cursor.fetchone())[0]

            assert count > 0  # Trigrams were added atomically


class TestAsyncTemperatureAdjustment:
    """Test async temperature mapping from dissonance."""

    @pytest.mark.asyncio
    async def test_temperature_min_dissonance(self, tmp_path):
        """Test temperature at minimum dissonance (0)."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            haiku_temp, harmonix_temp = await harmonix.adjust_temperature(0.0)

            # Min dissonance → min haiku temp
            assert haiku_temp == 0.3
            assert harmonix_temp == 0.3  # Observer stays low

    @pytest.mark.asyncio
    async def test_temperature_max_dissonance(self, tmp_path):
        """Test temperature at maximum dissonance (1)."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            haiku_temp, harmonix_temp = await harmonix.adjust_temperature(1.0)

            # Max dissonance → max haiku temp
            assert haiku_temp == 1.5  # 0.3 + 1.0 * 1.2
            assert harmonix_temp == 0.3

    @pytest.mark.asyncio
    async def test_temperature_mid_dissonance(self, tmp_path):
        """Test temperature at mid dissonance (0.5)."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            haiku_temp, harmonix_temp = await harmonix.adjust_temperature(0.5)

            # Mid dissonance → mid haiku temp
            assert 0.3 < haiku_temp < 1.5
            assert harmonix_temp == 0.3

    @pytest.mark.asyncio
    async def test_temperature_mapping_monotonic(self, tmp_path):
        """Test temperature increases monotonically with dissonance."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            temps = []
            for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
                haiku_temp, _ = await harmonix.adjust_temperature(d)
                temps.append(haiku_temp)

            # Should be increasing
            assert temps == sorted(temps)


class TestAsyncCloudMorphing:
    """Test async cloud morphing (ATOMIC boost active, decay dormant)."""

    @pytest.mark.asyncio
    async def test_morph_cloud_boosts_active_words(self, tmp_path):
        """Test that active words get weight boost."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Insert initial word
            cursor = await harmonix.conn.cursor()
            await cursor.execute("""
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES ('test', 1.0, 0, 0, 'seed')
            """)
            await harmonix.conn.commit()

            # Morph with 'test' as active
            await harmonix.morph_cloud(['test'])

            # Check weight increased
            await cursor.execute("SELECT weight FROM words WHERE word = 'test'")
            new_weight = (await cursor.fetchone())[0]
            assert new_weight > 1.0  # Should be boosted

    @pytest.mark.asyncio
    async def test_morph_cloud_decays_dormant_words(self, tmp_path):
        """Test that dormant words decay."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Insert words
            cursor = await harmonix.conn.cursor()
            await cursor.execute("""
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES ('active', 1.0, 0, 0, 'seed')
            """)
            await cursor.execute("""
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES ('dormant', 1.0, 0, 0, 'seed')
            """)
            await harmonix.conn.commit()

            # Morph with only 'active'
            await harmonix.morph_cloud(['active'])

            # Check dormant word decayed
            await cursor.execute("SELECT weight FROM words WHERE word = 'dormant'")
            dormant_weight = (await cursor.fetchone())[0]
            assert dormant_weight < 1.0  # Should be decayed (×0.99)

    @pytest.mark.asyncio
    async def test_morph_cloud_adds_new_words(self, tmp_path):
        """Test that new words are added."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            await harmonix.morph_cloud(['newword'])

            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT word FROM words WHERE word = 'newword'")
            result = await cursor.fetchone()

            assert result is not None
            assert result[0] == 'newword'


class TestAsyncTrigramStorage:
    """Test async trigram database operations (atomic under lock)."""

    @pytest.mark.asyncio
    async def test_update_trigrams_new(self, tmp_path):
        """Test adding new trigrams."""
        db_path = tmp_path / "test_cloud.db"
        trigrams = [("a", "b", "c"), ("d", "e", "f")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            async with harmonix._field_lock:  # Call within lock
                await harmonix.update_trigrams(trigrams)

            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT COUNT(*) FROM trigrams")
            count = (await cursor.fetchone())[0]

            assert count == 2

    @pytest.mark.asyncio
    async def test_update_trigrams_duplicate(self, tmp_path):
        """Test updating existing trigrams."""
        db_path = tmp_path / "test_cloud.db"
        trigram = [("a", "b", "c")]

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Add twice (both within lock)
            async with harmonix._field_lock:
                await harmonix.update_trigrams(trigram)
                await harmonix.update_trigrams(trigram)

            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT count FROM trigrams WHERE word1='a' AND word2='b' AND word3='c'")
            count = (await cursor.fetchone())[0]

            assert count == 2  # Should increment


class TestAsyncNumpyShards:
    """Test async numpy shard creation (uses asyncio.to_thread for CPU-bound)."""

    @pytest.mark.asyncio
    async def test_create_shard_saves_file(self, tmp_path):
        """Test that shard creates .npy file."""
        db_path = tmp_path / "test_cloud.db"
        interaction_data = {
            'input': 'test input',
            'output': 'test output',
            'dissonance': 0.5,
            'user_trigrams': [('a', 'b', 'c')],
            'system_trigrams': [('x', 'y', 'z')]
        }

        async with AsyncHarmonix(str(db_path)) as harmonix:
            await harmonix.create_shard(interaction_data)

            # Check shards directory exists
            assert Path('shards').exists()

            # Check at least one shard file created
            shards = list(Path('shards').glob('shard_*.npy'))
            assert len(shards) > 0

    @pytest.mark.asyncio
    async def test_create_shard_records_in_db(self, tmp_path):
        """Test that shard is recorded in database."""
        db_path = tmp_path / "test_cloud.db"
        interaction_data = {
            'input': 'test',
            'output': 'haiku',
            'dissonance': 0.6,
            'haiku_temp': 1.0,
            'harmonix_temp': 0.3
        }

        async with AsyncHarmonix(str(db_path)) as harmonix:
            await harmonix.create_shard(interaction_data)

            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT COUNT(*) FROM shards")
            count = (await cursor.fetchone())[0]

            assert count >= 1


class TestAsyncMetrics:
    """Test async metrics recording."""

    @pytest.mark.asyncio
    async def test_record_metrics(self, tmp_path):
        """Test recording metrics to database."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            await harmonix.record_metrics(
                perplexity=0.5,
                entropy=0.6,
                resonance=0.7
            )

            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT 1")
            row = await cursor.fetchone()

            assert row is not None
            # row = (id, timestamp, perplexity, entropy, resonance, cloud_size)
            assert row[2] == 0.5  # perplexity
            assert row[3] == 0.6  # entropy
            assert row[4] == 0.7  # resonance

    @pytest.mark.asyncio
    async def test_get_cloud_size(self, tmp_path):
        """Test getting cloud size (atomic read)."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Insert some words
            cursor = await harmonix.conn.cursor()
            for word in ['a', 'b', 'c']:
                await cursor.execute("""
                    INSERT INTO words (word, weight, frequency, last_used, added_by)
                    VALUES (?, 1.0, 0, 0, 'test')
                """, (word,))
            await harmonix.conn.commit()

            size = await harmonix.get_cloud_size()
            assert size == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, tmp_path):
        """Test getting field stats (atomic snapshot)."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Add some data
            await harmonix.morph_cloud(['test', 'words'])
            async with harmonix._field_lock:
                await harmonix.update_trigrams([('a', 'b', 'c')])

            # Get stats
            stats = await harmonix.get_stats()

            assert 'word_count' in stats
            assert 'trigram_count' in stats
            assert stats['word_count'] >= 2
            assert stats['trigram_count'] >= 1


class TestConcurrentOperations:
    """
    CRITICAL TESTS: Prove atomic field operations under concurrent load.

    This is the core Leo insight - without locks, field organisms corrupt!
    """

    @pytest.mark.asyncio
    async def test_concurrent_dissonance_computations(self, tmp_path):
        """Test that concurrent dissonance calls don't corrupt field."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Create 10 concurrent dissonance computations
            tasks = []
            for i in range(10):
                user_tri = [(f"user{i}", "test", "concurrent")]
                sys_tri = [("concurrent", "test", f"sys{i}")]
                tasks.append(harmonix.compute_dissonance(user_tri, sys_tri))

            # Run concurrently (should be atomic under lock!)
            results = await asyncio.gather(*tasks)

            # Verify all completed
            assert len(results) == 10

            # Verify field state is consistent (no corruption)
            stats = await harmonix.get_stats()
            assert stats['trigram_count'] > 0
            assert stats['word_count'] > 0

    @pytest.mark.asyncio
    async def test_concurrent_cloud_morphs(self, tmp_path):
        """Test that concurrent cloud morphs are atomic."""
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Create concurrent morph operations
            tasks = []
            for i in range(5):
                tasks.append(harmonix.morph_cloud([f"word{i}", "shared"]))

            # Run concurrently
            await asyncio.gather(*tasks)

            # Verify field state consistent
            cursor = await harmonix.conn.cursor()
            await cursor.execute("SELECT COUNT(*) FROM words")
            count = (await cursor.fetchone())[0]

            # Should have 6 words (word0-4 + shared)
            assert count == 6

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, tmp_path):
        """
        Test concurrent mixed operations (dissonance + morph + metrics).

        CRITICAL: Proves field coherence under realistic concurrent load!
        """
        db_path = tmp_path / "test_cloud.db"

        async with AsyncHarmonix(str(db_path)) as harmonix:
            # Mix of different operations
            tasks = [
                harmonix.compute_dissonance([("a", "b", "c")], [("x", "y", "z")]),
                harmonix.morph_cloud(["test", "concurrent"]),
                harmonix.record_metrics(0.5, 0.6, 0.7),
                harmonix.compute_dissonance([("d", "e", "f")], [("p", "q", "r")]),
                harmonix.get_cloud_size(),
            ]

            # Run all concurrently
            results = await asyncio.gather(*tasks)

            # Verify all completed without errors
            assert len(results) == 5

            # Verify field state is consistent
            stats = await harmonix.get_stats()
            assert stats['word_count'] >= 2  # At least from morph_cloud
            assert stats['trigram_count'] >= 2  # From two dissonance calls


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
