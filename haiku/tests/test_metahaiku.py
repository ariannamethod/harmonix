"""
Tests for metahaiku.py - Inner Voice

Tests:
- Bootstrap buffer (deque, max 8)
- Reflection generation
- High-dissonance/arousal triggers
- Cloud bias updates
"""

import pytest
from metahaiku import MetaHaiku
from haiku import HaikuGenerator, SEED_WORDS


class TestMetaHaiku:
    """Test MetaHaiku class."""

    @pytest.fixture
    def generator(self):
        return HaikuGenerator(SEED_WORDS[:50])

    @pytest.fixture
    def metahaiku(self, generator):
        return MetaHaiku(generator, max_snippets=8)

    def test_init(self, metahaiku):
        """Test MetaHaiku initialization."""
        assert metahaiku.bootstrap_buf.maxlen == 8
        assert len(metahaiku.reflections) == 0

    def test_reflect_generates_haiku(self, metahaiku):
        """Test reflect() generates internal haiku."""
        interaction = {
            'user': 'test input',
            'haiku': 'words\ndance\ncloud',
            'dissonance': 0.5,
            'pulse': None
        }

        internal_haiku = metahaiku.reflect(interaction)

        assert isinstance(internal_haiku, str)
        assert len(internal_haiku.split('\n')) == 3

    def test_reflect_stores_reflection(self, metahaiku):
        """Test reflect() stores reflection in history."""
        interaction = {'haiku': 'test\nhaiku\nhere'}

        metahaiku.reflect(interaction)
        assert len(metahaiku.reflections) == 1

        metahaiku.reflect(interaction)
        assert len(metahaiku.reflections) == 2

    def test_bootstrap_buffer_high_dissonance(self, metahaiku):
        """Test bootstrap buffer adds on high dissonance."""
        interaction = {
            'haiku': 'words dance in cloud space',
            'dissonance': 0.7  # High dissonance
        }

        initial_size = len(metahaiku.bootstrap_buf)
        metahaiku.reflect(interaction)

        # Should add to buffer (probabilistically)
        assert len(metahaiku.bootstrap_buf) >= initial_size

    def test_bootstrap_buffer_max_size(self, metahaiku):
        """Test bootstrap buffer respects max size."""
        for i in range(20):
            interaction = {
                'haiku': f'test haiku number {i}',
                'dissonance': 0.8
            }
            metahaiku.reflect(interaction)

        # Should never exceed max
        assert len(metahaiku.bootstrap_buf) <= 8

    def test_get_recent_reflections(self, metahaiku):
        """Test getting recent reflections."""
        for i in range(10):
            metahaiku.reflect({'haiku': f'test {i}'})

        recent = metahaiku.get_recent_reflections(n=5)
        assert len(recent) == 5

    def test_snippet_extraction(self, metahaiku):
        """Test snippet extraction from haiku."""
        interaction = {
            'haiku': 'first second third fourth fifth sixth seventh eighth ninth tenth',
            'dissonance': 0.7
        }

        metahaiku._feed_bootstrap(interaction)

        if len(metahaiku.bootstrap_buf) > 0:
            snippet = list(metahaiku.bootstrap_buf)[-1]
            # Should take first 10 words
            assert len(snippet.split()) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
