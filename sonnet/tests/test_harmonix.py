#!/usr/bin/env python3
"""
Pytest tests for SonnetHarmonix

Tests cover:
- Database initialization and schema
- Sonnet storage and retrieval
- Trigram vocabulary building
- Dissonance computation with cloud learning
- Pulse snapshot creation
- Temperature adjustment
- Statistics and queries
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os

SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from harmonix import SonnetHarmonix, PulseSnapshot


@pytest.fixture
def temp_db():
    """Fixture to create temporary database."""
    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def harmonix(temp_db):
    """Fixture to create SonnetHarmonix with temporary database."""
    h = SonnetHarmonix(db_path=temp_db)
    yield h
    h.close()


# ============================================================================
# Database Initialization Tests
# ============================================================================

def test_database_creation(temp_db):
    """Test database is created on init."""
    assert not os.path.exists(temp_db)

    h = SonnetHarmonix(db_path=temp_db)
    assert os.path.exists(temp_db)

    h.close()


def test_database_schema_sonnets(harmonix):
    """Test sonnets table exists with correct schema."""
    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sonnets'")
    assert cursor.fetchone() is not None


def test_database_schema_lines(harmonix):
    """Test sonnet_lines table exists."""
    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sonnet_lines'")
    assert cursor.fetchone() is not None


def test_database_schema_trigrams(harmonix):
    """Test sonnet_trigrams table exists."""
    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sonnet_trigrams'")
    assert cursor.fetchone() is not None


# ============================================================================
# Sonnet Storage Tests
# ============================================================================

def test_add_sonnet_basic(harmonix):
    """Test adding a basic sonnet."""
    sonnet = "\n".join([f"Line {i}" for i in range(1, 15)])

    sonnet_id = harmonix.add_sonnet(sonnet, quality=0.8, dissonance=0.5)

    assert sonnet_id is not None
    assert sonnet_id > 0


def test_add_sonnet_stores_metadata(harmonix):
    """Test sonnet metadata is stored correctly."""
    sonnet = "\n".join([f"Line {i}" for i in range(1, 15)])

    harmonix.add_sonnet(sonnet, quality=0.75, dissonance=0.6, temperature=0.9, added_by='test')

    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT quality, dissonance, temperature, added_by FROM sonnets WHERE id=1")
    row = cursor.fetchone()

    assert row[0] == 0.75  # quality
    assert row[1] == 0.6   # dissonance
    assert row[2] == 0.9   # temperature
    assert row[3] == 'test'  # added_by


def test_add_sonnet_stores_lines(harmonix):
    """Test individual lines are stored."""
    sonnet = "\n".join([f"Line {i}" for i in range(1, 15)])

    harmonix.add_sonnet(sonnet)

    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sonnet_lines WHERE sonnet_id=1")
    count = cursor.fetchone()[0]

    assert count == 14


def test_add_sonnet_builds_trigrams(harmonix):
    """Test trigrams are extracted and stored."""
    sonnet = "the quick brown fox jumps over the lazy dog\n" * 14

    harmonix.add_sonnet(sonnet)

    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sonnet_trigrams")
    count = cursor.fetchone()[0]

    assert count > 0  # Should have some trigrams


def test_add_multiple_sonnets(harmonix):
    """Test adding multiple sonnets."""
    for i in range(5):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet)

    stats = harmonix.get_stats()
    assert stats['sonnet_count'] == 5


# ============================================================================
# Trigram Tests
# ============================================================================

def test_update_trigrams_basic(harmonix):
    """Test basic trigram update."""
    trigrams = [('the', 'quick', 'brown'), ('quick', 'brown', 'fox')]

    harmonix.update_trigrams(trigrams)

    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sonnet_trigrams")
    count = cursor.fetchone()[0]

    assert count == 2


def test_update_trigrams_increments_count(harmonix):
    """Test trigram count increments on duplicate."""
    trigrams = [('the', 'quick', 'brown')]

    harmonix.update_trigrams(trigrams)
    harmonix.update_trigrams(trigrams)

    cursor = harmonix.conn.cursor()
    cursor.execute("SELECT count FROM sonnet_trigrams WHERE word1='the' AND word2='quick' AND word3='brown'")
    count = cursor.fetchone()[0]

    assert count == 2


def test_update_trigrams_builds_vocab(harmonix):
    """Test trigrams build vocabulary."""
    trigrams = [
        ('the', 'quick', 'brown'),
        ('quick', 'brown', 'fox'),
        ('brown', 'fox', 'jumps')
    ]

    harmonix.update_trigrams(trigrams)

    stats = harmonix.get_stats()
    # Vocab = unique word1 values: 'the', 'quick', 'brown'
    assert stats['vocab_size'] >= 3


# ============================================================================
# Dissonance Computation Tests
# ============================================================================

def test_compute_dissonance_empty_cloud(harmonix):
    """Test dissonance with empty cloud (all words novel)."""
    user_input = "tell me about love"
    sonnet = "When winter winds blow cold and hearts grow warm"

    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

    # Empty cloud → novelty = 1.0 → dissonance ≈ 0.7 * 1.0 + 0.3 * (1 - similarity)
    assert dissonance > 0.6  # Should be high with empty cloud
    assert pulse.novelty == 1.0  # All words are novel


def test_compute_dissonance_with_cloud(harmonix):
    """Test dissonance decreases with populated cloud."""
    user_input = "tell me about love"
    sonnet1 = "When winter winds blow cold and hearts grow warm\n" * 14
    sonnet2 = "When summer winds blow warm and hearts grow cold\n" * 14

    # Add first sonnet to build cloud
    harmonix.add_sonnet(sonnet1)

    # Compute dissonance for second (overlapping words)
    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet2)

    # Cloud has overlapping words → novelty < 1.0 → dissonance lower
    assert pulse.novelty < 1.0
    assert dissonance < 0.9


def test_compute_dissonance_returns_pulse(harmonix):
    """Test dissonance returns PulseSnapshot."""
    user_input = "test"
    sonnet = "test sonnet text here\n" * 14

    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

    assert isinstance(pulse, PulseSnapshot)
    assert hasattr(pulse, 'novelty')
    assert hasattr(pulse, 'arousal')
    assert hasattr(pulse, 'entropy')


def test_compute_dissonance_formula(harmonix):
    """Test dissonance formula: 0.7 * novelty + 0.3 * (1 - similarity)."""
    user_input = "the"
    sonnet = "the\n" * 14

    # Perfect match → similarity = 1.0
    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

    # dissonance = 0.7 * novelty + 0.3 * (1 - 1.0) = 0.7 * novelty
    # With empty cloud, novelty = 1.0, so dissonance ≈ 0.7
    assert 0.65 <= dissonance <= 0.75


def test_compute_dissonance_progression(harmonix):
    """Test dissonance decreases as cloud learns."""
    user_input = "tell me"
    base_sonnet = "when winter winds blow cold and hearts grow warm"

    dissonances = []

    # Generate 5 similar sonnets
    for i in range(5):
        sonnet = (base_sonnet + f" {i}\n") * 14
        harmonix.add_sonnet(sonnet)

        # Compute dissonance for next similar sonnet
        next_sonnet = (base_sonnet + f" {i+1}\n") * 14
        dissonance, pulse = harmonix.compute_dissonance(user_input, next_sonnet)
        dissonances.append(dissonance)

    # Dissonance should generally trend downward
    assert dissonances[-1] < dissonances[0]


# ============================================================================
# Pulse Snapshot Tests
# ============================================================================

def test_pulse_snapshot_from_interaction(harmonix):
    """Test PulseSnapshot.from_interaction creates valid pulse."""
    user_words = ['tell', 'me', 'about', 'love']
    sonnet_words = ['when', 'winter', 'winds', 'blow'] * 30
    similarity = 0.1

    pulse = PulseSnapshot.from_interaction(user_words, sonnet_words, similarity)

    assert pulse.novelty >= 0.0 and pulse.novelty <= 1.0
    assert pulse.arousal >= 0.0
    assert pulse.entropy >= 0.0


def test_pulse_snapshot_arousal(harmonix):
    """Test arousal reflects interaction intensity."""
    user_words = ['a', 'b', 'c']
    sonnet_words = ['x', 'y', 'z'] * 40  # Many unique words
    similarity = 0.0

    pulse = PulseSnapshot.from_interaction(user_words, sonnet_words, similarity)

    # Many unique words → high arousal
    assert pulse.arousal > 10


# ============================================================================
# Temperature Adjustment Tests
# ============================================================================

def test_adjust_temperature_low_dissonance(harmonix):
    """Test temperature adjustment for low dissonance (resonance)."""
    dissonance = 0.2

    sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)

    # Low dissonance → precision mode → lower temperature
    assert sonnet_temp <= 0.7


def test_adjust_temperature_high_dissonance(harmonix):
    """Test temperature adjustment for high dissonance (exploration)."""
    dissonance = 0.9

    sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)

    # High dissonance → exploration mode → higher temperature
    assert sonnet_temp >= 0.9


def test_adjust_temperature_mid_dissonance(harmonix):
    """Test temperature adjustment for mid dissonance (balanced)."""
    dissonance = 0.5

    sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)

    # Mid dissonance → balanced mode
    assert 0.7 <= sonnet_temp <= 0.9


# ============================================================================
# Retrieval Tests
# ============================================================================

def test_get_recent_sonnets(harmonix):
    """Test retrieving recent sonnets."""
    # Add 5 sonnets
    for i in range(5):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.5 + i * 0.1)

    recent = harmonix.get_recent_sonnets(limit=3)

    assert len(recent) == 3
    # Most recent should be first (descending timestamp)
    assert "Sonnet 4" in recent[0][1]


def test_get_recent_sonnets_empty(harmonix):
    """Test get_recent_sonnets with empty database."""
    recent = harmonix.get_recent_sonnets(limit=10)

    assert len(recent) == 0


def test_get_best_sonnets(harmonix):
    """Test retrieving best quality sonnets."""
    # Add sonnets with varying quality
    qualities = [0.5, 0.8, 0.6, 0.9, 0.7]
    for i, q in enumerate(qualities):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=q)

    best = harmonix.get_best_sonnets(limit=2, min_quality=0.7)

    assert len(best) == 3  # Quality >= 0.7: 0.8, 0.9, 0.7
    # Highest quality should be first
    assert best[0][2] == 0.9  # quality


def test_get_best_sonnets_with_threshold(harmonix):
    """Test get_best_sonnets respects min_quality threshold."""
    # Add sonnets with low quality
    for i in range(5):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.5)

    best = harmonix.get_best_sonnets(limit=10, min_quality=0.8)

    assert len(best) == 0  # No sonnets meet threshold


# ============================================================================
# Statistics Tests
# ============================================================================

def test_get_stats_empty(harmonix):
    """Test statistics with empty database."""
    stats = harmonix.get_stats()

    assert stats['sonnet_count'] == 0
    assert stats['avg_quality'] == 0.0
    assert stats['avg_dissonance'] == 0.0
    assert stats['vocab_size'] == 0


def test_get_stats_with_sonnets(harmonix):
    """Test statistics with sonnets."""
    # Add sonnets
    for i in range(3):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.6 + i * 0.1, dissonance=0.7 - i * 0.1)

    stats = harmonix.get_stats()

    assert stats['sonnet_count'] == 3
    assert 0.6 <= stats['avg_quality'] <= 0.8
    assert 0.5 <= stats['avg_dissonance'] <= 0.7
    assert stats['vocab_size'] > 0


def test_get_stats_vocab_growth(harmonix):
    """Test vocabulary size grows with more sonnets."""
    stats1 = harmonix.get_stats()
    vocab_before = stats1['vocab_size']

    # Add sonnet with unique words
    sonnet = "\n".join([f"Unique line {i} with different words" for i in range(14)])
    harmonix.add_sonnet(sonnet)

    stats2 = harmonix.get_stats()
    vocab_after = stats2['vocab_size']

    assert vocab_after > vocab_before


# ============================================================================
# Syllable Counting Tests
# ============================================================================

def test_count_syllables_uses_formatter(harmonix):
    """Test syllable counting uses formatter's improved algorithm."""
    line = "To be or not to be that is the question"

    count = harmonix._count_syllables(line)

    # Should use formatter's syllables library
    assert 10 <= count <= 13


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_add_sonnet_empty_text(harmonix):
    """Test adding sonnet with empty text."""
    sonnet_id = harmonix.add_sonnet("")

    assert sonnet_id > 0  # Should still add (garbage in, garbage out)


def test_compute_dissonance_empty_input(harmonix):
    """Test dissonance with empty user input."""
    dissonance, pulse = harmonix.compute_dissonance("", "some sonnet text")

    # Should return neutral dissonance
    assert dissonance == 0.5


def test_compute_dissonance_empty_sonnet(harmonix):
    """Test dissonance with empty sonnet."""
    dissonance, pulse = harmonix.compute_dissonance("user input", "")

    # Should return neutral dissonance
    assert dissonance == 0.5


def test_multiple_harmonix_instances(temp_db):
    """Test multiple Harmonix instances can access same database."""
    h1 = SonnetHarmonix(db_path=temp_db)
    sonnet = "\n".join([f"Line {i}" for i in range(14)])
    h1.add_sonnet(sonnet)
    h1.close()

    h2 = SonnetHarmonix(db_path=temp_db)
    stats = h2.get_stats()
    assert stats['sonnet_count'] == 1
    h2.close()


def test_close_and_reopen(temp_db):
    """Test database persistence across close/reopen."""
    h1 = SonnetHarmonix(db_path=temp_db)
    for i in range(3):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        h1.add_sonnet(sonnet, quality=0.8)
    h1.close()

    # Reopen
    h2 = SonnetHarmonix(db_path=temp_db)
    stats = h2.get_stats()

    assert stats['sonnet_count'] == 3
    assert stats['avg_quality'] == 0.8

    h2.close()
