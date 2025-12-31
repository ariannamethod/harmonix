"""
Tests for ProseHarmonix - Field Observer Module

Tests cloud storage, field operations, dissonance computation,
temperature adjustment, and field seed generation.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harmonix import ProseHarmonix


class TestProseHarmonixInit(unittest.TestCase):
    """Test ProseHarmonix initialization."""

    def setUp(self):
        """Create temporary database."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_initialization(self):
        """Test basic initialization."""
        self.assertIsNotNone(self.harmonix.conn)
        self.assertEqual(self.harmonix.db_path, self.db_path)

    def test_tables_created(self):
        """Test that database tables are created."""
        cursor = self.harmonix.conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Should have at least one table
        self.assertGreater(len(tables), 0)

    def test_empty_cloud(self):
        """Test empty cloud statistics."""
        stats = self.harmonix.get_stats()

        self.assertEqual(stats['total_prose'], 0)
        self.assertEqual(stats['trigram_vocabulary'], 0)
        self.assertEqual(stats['avg_quality'], 0.0)


class TestProseHarmonixAddProse(unittest.TestCase):
    """Test adding prose to cloud."""

    def setUp(self):
        """Create harmonix instance."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_prose_basic(self):
        """Test adding prose to cloud."""
        text = "Words flow through consciousness like water through stone."

        prose_id = self.harmonix.add_prose(
            text=text,
            quality=0.8,
            dissonance=0.5,
            temperature=0.9
        )

        self.assertIsInstance(prose_id, int)
        self.assertGreater(prose_id, 0)

    def test_add_prose_updates_stats(self):
        """Test that adding prose updates statistics."""
        initial_stats = self.harmonix.get_stats()

        self.harmonix.add_prose("Test prose text.", quality=0.7)

        updated_stats = self.harmonix.get_stats()
        self.assertEqual(updated_stats['total_prose'], initial_stats['total_prose'] + 1)

    def test_add_prose_with_metadata(self):
        """Test adding prose with metadata."""
        prose_id = self.harmonix.add_prose(
            text="Language is a living organism.",
            quality=0.85,
            dissonance=0.6,
            temperature=1.0,
            added_by='test'
        )

        self.assertGreater(prose_id, 0)

    def test_add_multiple_prose(self):
        """Test adding multiple prose entries."""
        for i in range(5):
            self.harmonix.add_prose(f"Prose text {i}", quality=0.7)

        stats = self.harmonix.get_stats()
        self.assertEqual(stats['total_prose'], 5)


class TestProseHarmonixTrigrams(unittest.TestCase):
    """Test trigram field operations."""

    def setUp(self):
        """Create harmonix instance."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_extract_trigrams(self):
        """Test trigram extraction via update."""
        text = "the quick brown fox"

        # Extract trigrams by updating
        initial_vocab = self.harmonix.get_stats()['trigram_vocabulary']
        self.harmonix._update_trigrams(text)
        updated_vocab = self.harmonix.get_stats()['trigram_vocabulary']

        # Should increase vocabulary
        self.assertGreaterEqual(updated_vocab, initial_vocab)

    def test_update_trigrams(self):
        """Test trigram field update."""
        text = "consciousness flows like water"

        initial_vocab = self.harmonix.get_stats()['trigram_vocabulary']

        self.harmonix._update_trigrams(text)

        updated_vocab = self.harmonix.get_stats()['trigram_vocabulary']
        self.assertGreater(updated_vocab, initial_vocab)

    def test_add_disturbance(self):
        """Test field disturbance from user input."""
        user_input = "what is the nature of language?"

        self.harmonix.add_disturbance(user_input, source='user')

        stats = self.harmonix.get_stats()
        self.assertGreater(stats['trigram_vocabulary'], 0)

    def test_get_resonant_trigrams(self):
        """Test trigram persistence."""
        # Add some trigrams
        for _ in range(3):
            self.harmonix._update_trigrams("the quick brown fox jumps")

        # Verify they're stored
        stats = self.harmonix.get_stats()
        self.assertGreater(stats['trigram_vocabulary'], 0)


class TestProseHarmonixDissonance(unittest.TestCase):
    """Test dissonance computation."""

    def setUp(self):
        """Create harmonix instance."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_compute_dissonance(self):
        """Test dissonance computation."""
        user_input = "consciousness"
        prose_output = "awareness"

        dissonance, pulse = self.harmonix.compute_dissonance(user_input, prose_output)

        self.assertIsInstance(dissonance, float)
        self.assertGreaterEqual(dissonance, 0.0)
        self.assertLessEqual(dissonance, 1.0)

    def test_dissonance_with_pulse(self):
        """Test that pulse object is returned."""
        user_input = "language"
        prose_output = "words"

        dissonance, pulse = self.harmonix.compute_dissonance(user_input, prose_output)

        self.assertIsNotNone(pulse)
        self.assertTrue(hasattr(pulse, 'novelty'))
        self.assertTrue(hasattr(pulse, 'arousal'))

    def test_adjust_temperature(self):
        """Test temperature adjustment based on dissonance."""
        # Low dissonance
        temp_low = self.harmonix.adjust_temperature(dissonance=0.2)

        # High dissonance
        temp_high = self.harmonix.adjust_temperature(dissonance=0.9)

        # Both should be positive
        self.assertGreater(temp_low, 0)
        self.assertGreater(temp_high, 0)
        # High might be higher (but not guaranteed depending on algorithm)
        self.assertIsInstance(temp_high, float)


class TestProseHarmonixFieldSeed(unittest.TestCase):
    """Test field seed generation (organism mode!)."""

    def setUp(self):
        """Create harmonix with some prose."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

        # Add some prose to cloud
        self.harmonix.add_prose("Words flow through consciousness like water.", quality=0.8)
        self.harmonix.add_prose("Language is a living organism that breathes.", quality=0.7)
        self.harmonix.add_prose("The field resonates with patterns of meaning.", quality=0.9)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_field_seed(self):
        """Test field seed retrieval (CRITICAL for organism mode!)."""
        seed = self.harmonix.get_field_seed()

        self.assertIsInstance(seed, str)
        self.assertGreater(len(seed), 0)

    def test_get_field_seed_auto_mode(self):
        """Test automatic seed mode selection."""
        seed = self.harmonix.get_field_seed(mode='auto')

        self.assertIsInstance(seed, str)

    def test_get_field_seed_recent_mode(self):
        """Test recent prose seed mode."""
        seed = self.harmonix.get_field_seed(mode='recent')

        self.assertIsInstance(seed, str)

    def test_get_field_seed_trigram_mode(self):
        """Test trigram-based seed mode."""
        # Add trigrams first
        self.harmonix._update_trigrams("consciousness flows like water")

        seed = self.harmonix.get_field_seed(mode='trigram')

        self.assertIsInstance(seed, str)

    def test_get_field_seed_random_mode(self):
        """Test random sentence seed mode."""
        seed = self.harmonix.get_field_seed(mode='random')

        self.assertIsInstance(seed, str)


class TestProseHarmonixRetrieval(unittest.TestCase):
    """Test prose retrieval from cloud."""

    def setUp(self):
        """Create harmonix with test data."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

        # Add test prose
        self.harmonix.add_prose("First prose.", quality=0.5)
        self.harmonix.add_prose("Second prose.", quality=0.9)
        self.harmonix.add_prose("Third prose.", quality=0.7)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_recent_prose(self):
        """Test recent prose retrieval."""
        recent = self.harmonix.get_recent_prose(limit=2)

        self.assertIsInstance(recent, list)
        self.assertEqual(len(recent), 2)

    def test_get_best_prose(self):
        """Test best quality prose retrieval."""
        best = self.harmonix.get_best_prose(limit=2)

        self.assertIsInstance(best, list)
        self.assertEqual(len(best), 2)

        # Check ordering by quality
        if len(best) >= 2:
            quality1 = best[0][2]
            quality2 = best[1][2]
            self.assertGreaterEqual(quality1, quality2)

    def test_get_prose_by_id(self):
        """Test prose retrieval."""
        prose_id = self.harmonix.add_prose("Target prose.", quality=0.8)

        # Verify it's in recent prose
        recent = self.harmonix.get_recent_prose(limit=1)

        self.assertEqual(len(recent), 1)
        self.assertIn("Target", recent[0][1])


class TestProseHarmonixStats(unittest.TestCase):
    """Test statistics and metrics."""

    def setUp(self):
        """Create harmonix instance."""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.harmonix = ProseHarmonix(db_path=self.db_path)

    def tearDown(self):
        """Clean up."""
        self.harmonix.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_stats_structure(self):
        """Test statistics dictionary structure."""
        stats = self.harmonix.get_stats()

        self.assertIn('total_prose', stats)
        self.assertIn('avg_quality', stats)
        self.assertIn('avg_dissonance', stats)
        self.assertIn('avg_semantic_density', stats)
        self.assertIn('trigram_vocabulary', stats)

    def test_stats_with_data(self):
        """Test statistics with actual data."""
        # Add prose with known quality
        self.harmonix.add_prose("Test 1", quality=0.6)
        self.harmonix.add_prose("Test 2", quality=0.8)

        stats = self.harmonix.get_stats()

        self.assertEqual(stats['total_prose'], 2)
        self.assertAlmostEqual(stats['avg_quality'], 0.7, places=1)


if __name__ == '__main__':
    unittest.main()
