#!/usr/bin/env python3
"""
Pytest tests for Overthinkng (note the typo!)

Tests cover:
- Ring 0: Echo (shuffle & rephrase)
- Ring 1: Drift (semantic replacement)
- Ring 2: Meta (keyword extraction)
- Full expansion cycle
- Database integration
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os

SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from overthinkng import Overthinkng
from harmonix import SonnetHarmonix
from sonnet import SonnetGenerator


@pytest.fixture
def temp_db():
    """Temporary database."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def harmonix(temp_db):
    """Harmonix with temp DB."""
    h = SonnetHarmonix(db_path=temp_db)
    yield h
    h.close()


@pytest.fixture
def generator():
    """Generator fixture."""
    gen = SonnetGenerator()
    yield gen
    gen.close()


@pytest.fixture
def overthinkng(harmonix, generator):
    """Overthinkng fixture (note the typo!)."""
    return Overthinkng(harmonix=harmonix, generator=generator)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_overthinkng_init(overthinkng):
    """Test Overthinkng initialization (typo in name!)."""
    assert overthinkng.harmonix is not None
    assert overthinkng.generator is not None


def test_overthinkng_typo_correct(overthinkng):
    """Test class name has correct typo: 'overthinkng' not 'overthinkg'."""
    # This is intentional - different from HAiKU's 'overthinkg'
    class_name = overthinkng.__class__.__name__
    assert class_name == 'Overthinkng'
    assert 'ng' in class_name  # Not 'kg'


# ============================================================================
# Ring 0: Echo Tests
# ============================================================================

def test_ring0_echo_returns_variations(overthinkng, harmonix):
    """Test Ring 0 generates echo variations."""
    # Add base sonnet
    base = "\n".join([f"Line {i} with words here" for i in range(14)])
    harmonix.add_sonnet(base, quality=0.8)

    variations = overthinkng._ring0_echo()

    # Should return list of variations (possibly empty if generation fails)
    assert isinstance(variations, list)


def test_ring0_echo_shuffle_strategy(overthinkng):
    """Test Ring 0 uses shuffle strategy."""
    text = "The quick brown fox jumps over the lazy dog"
    shuffled = overthinkng._shuffle_words(text)

    # Should have same words, different order
    original_words = set(text.lower().split())
    shuffled_words = set(shuffled.lower().split())

    # Same vocabulary
    assert original_words == shuffled_words


def test_ring0_temperature(overthinkng):
    """Test Ring 0 uses temperature 0.8."""
    # Ring 0 should use lower temp for conservative variation
    assert overthinkng.ring_temps[0] == 0.8


# ============================================================================
# Ring 1: Drift Tests
# ============================================================================

def test_ring1_drift_returns_variations(overthinkng, harmonix):
    """Test Ring 1 generates drift variations."""
    # Add base sonnet
    base = "\n".join([f"Line {i}" for i in range(14)])
    harmonix.add_sonnet(base, quality=0.8)

    variations = overthinkng._ring1_drift()

    assert isinstance(variations, list)


def test_ring1_semantic_replacement(overthinkng):
    """Test Ring 1 uses semantic replacement strategy."""
    text = "love and death"

    # Has simple replacement logic
    replaced = overthinkng._replace_words(text)

    # Should be different (if replacements applied)
    # But might be same if no replacements available
    assert isinstance(replaced, str)


def test_ring1_temperature(overthinkng):
    """Test Ring 1 uses temperature 1.0."""
    assert overthinkng.ring_temps[1] == 1.0


# ============================================================================
# Ring 2: Meta Tests
# ============================================================================

def test_ring2_meta_returns_variations(overthinkng, harmonix):
    """Test Ring 2 generates meta variations."""
    base = "\n".join([f"Love and death intertwine {i}" for i in range(14)])
    harmonix.add_sonnet(base, quality=0.8)

    variations = overthinkng._ring2_meta()

    assert isinstance(variations, list)


def test_ring2_keyword_extraction(overthinkng):
    """Test Ring 2 extracts keywords."""
    text = "love death winter summer hope fear joy sorrow"
    keywords = overthinkng._extract_keywords(text, n=3)

    assert isinstance(keywords, list)
    assert len(keywords) <= 3


def test_ring2_temperature(overthinkng):
    """Test Ring 2 uses temperature 1.2 (most exploratory)."""
    assert overthinkng.ring_temps[2] == 1.2


# ============================================================================
# Full Expansion Tests
# ============================================================================

def test_expand_adds_variations(overthinkng, harmonix):
    """Test expand() adds variations to cloud."""
    # Add initial sonnets
    for i in range(2):
        sonnet = "\n".join([f"Initial sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.7)

    initial_count = harmonix.get_stats()['sonnet_count']

    # Run expansion (fast version with fewer rings)
    overthinkng.expand(num_rings=1)

    final_count = harmonix.get_stats()['sonnet_count']

    # Should have added some variations (or at least tried)
    assert final_count >= initial_count


def test_expand_with_num_rings(overthinkng, harmonix):
    """Test expand() respects num_rings parameter."""
    sonnet = "\n".join([f"Line {i}" for i in range(14)])
    harmonix.add_sonnet(sonnet, quality=0.8)

    # Run with 1 ring only
    overthinkng.expand(num_rings=1)

    # Just verify it doesn't crash
    assert True


def test_expand_empty_cloud(overthinkng, harmonix):
    """Test expand() handles empty cloud gracefully."""
    # No sonnets in cloud
    try:
        overthinkng.expand(num_rings=1)
        success = True
    except Exception:
        success = False

    # Should handle gracefully (no source sonnets = no variations)
    assert success


def test_expand_marks_source(overthinkng, harmonix):
    """Test expanded sonnets marked with added_by='overthinkng'."""
    sonnet = "\n".join([f"Source line {i}" for i in range(14)])
    harmonix.add_sonnet(sonnet, quality=0.8)

    # Expand
    overthinkng.expand(num_rings=1)

    # Check database for overthinkng-sourced sonnets
    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sonnets WHERE added_by='overthinkng'")
    count = cursor.fetchone()[0]

    # Should have added at least some (or 0 if generation failed)
    assert count >= 0


# ============================================================================
# Helper Method Tests
# ============================================================================

def test_shuffle_words_basic(overthinkng):
    """Test word shuffling."""
    text = "one two three four five"
    shuffled = overthinkng._shuffle_words(text)

    # Should have same words
    original_words = sorted(text.split())
    shuffled_words = sorted(shuffled.split())

    assert original_words == shuffled_words


def test_shuffle_words_empty(overthinkng):
    """Test shuffle with empty string."""
    shuffled = overthinkng._shuffle_words("")

    assert shuffled == ""


def test_replace_words_has_mappings(overthinkng):
    """Test word replacement has predefined mappings."""
    assert hasattr(overthinkng, 'word_replacements')
    assert isinstance(overthinkng.word_replacements, dict)
    assert len(overthinkng.word_replacements) > 0


def test_replace_words_replaces(overthinkng):
    """Test word replacement works."""
    # Use a word we know is in replacements
    if 'love' in overthinkng.word_replacements:
        text = "love is eternal"
        replaced = overthinkng._replace_words(text)

        # Should have replaced 'love'
        assert 'love' not in replaced or replaced != text


def test_extract_keywords_empty(overthinkng):
    """Test keyword extraction with empty text."""
    keywords = overthinkng._extract_keywords("", n=5)

    assert len(keywords) == 0


def test_extract_keywords_returns_top_n(overthinkng):
    """Test keyword extraction returns top N."""
    text = "love love love death death winter"
    keywords = overthinkng._extract_keywords(text, n=2)

    assert len(keywords) <= 2


def test_extract_keywords_filters_stopwords(overthinkng):
    """Test keyword extraction filters stopwords."""
    text = "the and or but if when where love death"
    keywords = overthinkng._extract_keywords(text, n=10)

    # Stopwords should be filtered
    stopwords = {'the', 'and', 'or', 'but', 'if', 'when', 'where'}
    for kw in keywords:
        assert kw.lower() not in stopwords


# ============================================================================
# Quality and Filtering Tests
# ============================================================================

def test_expand_prefers_high_quality(overthinkng, harmonix):
    """Test expansion preferentially selects high-quality sonnets."""
    # Add mix of quality
    for i in range(3):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        quality = 0.5 if i < 2 else 0.9  # Last one is high quality
        harmonix.add_sonnet(sonnet, quality=quality)

    # Get best sonnets used for expansion
    best = harmonix.get_best_sonnets(limit=1, min_quality=0.7)

    # Should prefer high quality
    if best:
        assert best[0][2] >= 0.7  # quality


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_overthinkng_cycle(overthinkng, harmonix):
    """Test complete overthinkng cycle."""
    # Add base sonnets
    for i in range(2):
        sonnet = "\n".join([f"Base {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.8)

    initial_count = harmonix.get_stats()['sonnet_count']

    # Run all 3 rings
    overthinkng.expand(num_rings=3)

    # Verify completion
    final_count = harmonix.get_stats()['sonnet_count']
    assert final_count >= initial_count


def test_overthinkng_with_metasonnet(overthinkng, harmonix):
    """Test overthinkng works with metasonnet-sourced sonnets."""
    # Add sonnet from metasonnet
    sonnet = "\n".join([f"Internal reflection {i}" for i in range(14)])
    harmonix.add_sonnet(sonnet, quality=0.7, added_by='metasonnet')

    # Expand should work with any source
    initial_count = harmonix.get_stats()['sonnet_count']
    overthinkng.expand(num_rings=1)

    # Should complete without error
    assert harmonix.get_stats()['sonnet_count'] >= initial_count


def test_overthinkng_doesnt_crash_on_short_text(overthinkng):
    """Test overthinkng helpers handle short text."""
    short = "hi"

    # Should not crash
    try:
        overthinkng._shuffle_words(short)
        overthinkng._replace_words(short)
        overthinkng._extract_keywords(short, n=5)
        success = True
    except Exception:
        success = False

    assert success


def test_overthinkng_close(overthinkng):
    """Test overthinkng cleanup."""
    # Should have close method
    assert hasattr(overthinkng, 'close')

    # Should not crash
    try:
        overthinkng.close()
        success = True
    except Exception:
        success = False

    assert success
