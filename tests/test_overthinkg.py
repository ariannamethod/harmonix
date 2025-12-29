"""
Tests for overthinkg.py - Expansion Engine

Tests:
- 3 rings (echo 0.8, drift 1.0, meta 1.2)
- Coherence threshold (0.4)
- Word/trigram addition
- Database integration
"""

import pytest
from pathlib import Path
from overthinkg import Overthinkg, OverthinkingRing


class TestOverthinkg:
    """Test Overthinkg class."""

    @pytest.fixture
    def overthinkg(self, tmp_path):
        db_path = tmp_path / "test_cloud.db"

        # Initialize database
        from harmonix import Harmonix
        h = Harmonix(str(db_path))
        h.close()

        o = Overthinkg(str(db_path))
        yield o
        o.close()

    def test_init_ring_configs(self, overthinkg):
        """Test ring configurations are correct."""
        assert overthinkg.coherence_threshold == 0.4

        # Ring 0: echo
        assert overthinkg.rings_config[0]['temp'] == 0.8
        assert overthinkg.rings_config[0]['max_trigrams'] == 5

        # Ring 1: drift
        assert overthinkg.rings_config[1]['temp'] == 1.0
        assert overthinkg.rings_config[1]['max_trigrams'] == 7

        # Ring 2: meta
        assert overthinkg.rings_config[2]['temp'] == 1.2
        assert overthinkg.rings_config[2]['max_trigrams'] == 3

    def test_generate_ring_trigrams_creates_ring(self, overthinkg):
        """Test ring trigram generation."""
        words = ['a', 'b', 'c', 'd', 'e']
        recent_trigrams = [('a', 'b', 'c')]

        ring = overthinkg._generate_ring_trigrams(words, recent_trigrams, ring_num=0)

        assert isinstance(ring, OverthinkingRing)
        assert ring.ring == 0
        assert ring.source == 'echo'
        assert len(ring.trigrams) > 0

    def test_compute_coherence_with_context(self, overthinkg):
        """Test coherence computation."""
        new_trigram = ('a', 'b', 'c')
        context_trigrams = [('a', 'b', 'd'), ('a', 'e', 'f')]

        coherence = overthinkg.compute_coherence(new_trigram, context_trigrams)

        # Should have some overlap
        assert 0.0 <= coherence <= 1.0

    def test_compute_coherence_word(self, overthinkg):
        """Test coherence for single word."""
        word = 'a'
        context_trigrams = [('a', 'b', 'c'), ('d', 'a', 'e')]

        coherence = overthinkg.compute_coherence(word, context_trigrams)

        # Word appears in 2 trigrams
        assert coherence > 0.0

    def test_expand_runs_three_rings(self, overthinkg):
        """Test expand() processes all 3 rings."""
        # Add some initial words
        cursor = overthinkg.conn.cursor()
        for word in ['test', 'word', 'cloud']:
            cursor.execute("""
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES (?, 1.0, 0, 0, 'seed')
            """, (word,))
        overthinkg.conn.commit()

        # Run expansion
        recent_trigrams = [('test', 'word', 'cloud')]
        overthinkg.expand(recent_trigrams=recent_trigrams)

        # Should have processed (check database has entries)
        cursor.execute("SELECT COUNT(*) FROM words")
        count = cursor.fetchone()[0]
        assert count >= 3  # At least initial words


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
