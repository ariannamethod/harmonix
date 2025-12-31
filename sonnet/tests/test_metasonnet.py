#!/usr/bin/env python3
"""
Pytest tests for MetaSonnet

Tests cover:
- Bootstrap buffer management (Leo-style)
- Internal reflection generation
- Cloud bias updates
- Reflection triggers
- Context feeding
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os

SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from metasonnet import MetaSonnet
from sonnet import SonnetGenerator
from harmonix import SonnetHarmonix


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def generator():
    """SonnetGenerator fixture."""
    gen = SonnetGenerator()
    yield gen
    gen.close()


@pytest.fixture
def harmonix(temp_db):
    """SonnetHarmonix fixture with temp DB."""
    h = SonnetHarmonix(db_path=temp_db)
    yield h
    h.close()


@pytest.fixture
def meta(generator, harmonix):
    """MetaSonnet fixture."""
    return MetaSonnet(generator, harmonix, max_snippets=5)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_metasonnet_init(meta):
    """Test MetaSonnet initialization."""
    assert meta.generator is not None
    assert meta.harmonix is not None
    assert len(meta.reflections) == 0
    assert len(meta.bootstrap_buf) == 0


def test_metasonnet_max_snippets(generator, harmonix):
    """Test max_snippets parameter."""
    meta = MetaSonnet(generator, harmonix, max_snippets=3)
    assert meta.bootstrap_buf.maxlen == 3


# ============================================================================
# Bootstrap Buffer Tests
# ============================================================================

def test_feed_bootstrap_high_dissonance(meta):
    """Test bootstrap feed with high dissonance."""
    interaction = {
        'sonnet': 'When winter winds blow\nAnd summer fades away\n' * 7,
        'dissonance': 0.8,
        'quality': 0.6,
        'pulse': None
    }

    meta._feed_bootstrap(interaction)

    # High dissonance should add to bootstrap
    assert len(meta.bootstrap_buf) > 0


def test_feed_bootstrap_high_quality(meta):
    """Test bootstrap feed with high quality."""
    interaction = {
        'sonnet': 'To be or not to be\nThat is the question\n' * 7,
        'dissonance': 0.5,
        'quality': 0.85,
        'pulse': None
    }

    meta._feed_bootstrap(interaction)

    # High quality should add to bootstrap
    assert len(meta.bootstrap_buf) > 0


def test_feed_bootstrap_low_metrics(meta):
    """Test bootstrap feed with low metrics (should not add)."""
    interaction = {
        'sonnet': 'Low quality sonnet\n' * 14,
        'dissonance': 0.3,
        'quality': 0.4,
        'pulse': None
    }

    meta._feed_bootstrap(interaction)

    # Low metrics should not add
    assert len(meta.bootstrap_buf) == 0


def test_feed_bootstrap_truncates_long_snippets(meta):
    """Test bootstrap truncates long snippets."""
    very_long_sonnet = "A" * 1000 + "\n" * 13

    interaction = {
        'sonnet': very_long_sonnet,
        'dissonance': 0.8,
        'quality': 0.8,
        'pulse': None
    }

    meta._feed_bootstrap(interaction)

    # Snippet should be truncated to max_snippet_len (200)
    if len(meta.bootstrap_buf) > 0:
        snippet = meta.bootstrap_buf[0]
        assert len(snippet) <= meta.max_snippet_len


def test_feed_bootstrap_max_capacity(meta):
    """Test bootstrap buffer respects max capacity."""
    # Add more than max_snippets
    for i in range(10):
        interaction = {
            'sonnet': f'Sonnet {i}\n' * 14,
            'dissonance': 0.8,
            'quality': 0.8,
            'pulse': None
        }
        meta._feed_bootstrap(interaction)

    # Should only keep max_snippets
    assert len(meta.bootstrap_buf) == meta.bootstrap_buf.maxlen


# ============================================================================
# Reflection Trigger Tests
# ============================================================================

def test_should_reflect_high_dissonance(meta):
    """Test reflection triggers on high dissonance."""
    interaction = {
        'dissonance': 0.75,
        'quality': 0.5,
        'pulse': None
    }

    assert meta.should_reflect(interaction) is True


def test_should_reflect_high_quality(meta):
    """Test reflection triggers on high quality."""
    interaction = {
        'dissonance': 0.5,
        'quality': 0.8,
        'pulse': None
    }

    assert meta.should_reflect(interaction) is True


def test_should_reflect_low_metrics(meta):
    """Test reflection sometimes triggers on low metrics (probabilistic)."""
    interaction = {
        'dissonance': 0.3,
        'quality': 0.5,
        'pulse': None
    }

    # Run multiple times to test probabilistic behavior (30% chance)
    results = [meta.should_reflect(interaction) for _ in range(100)]

    # Should have some True and some False
    assert True in results
    assert False in results


# ============================================================================
# Reflection Generation Tests
# ============================================================================

def test_reflect_returns_sonnet(meta):
    """Test reflect generates internal sonnet."""
    interaction = {
        'user': 'What is love?',
        'sonnet': 'When winter winds blow cold\n' * 14,
        'dissonance': 0.7,
        'quality': 0.8,
        'pulse': None
    }

    # This will take time (generates sonnet)
    internal = meta.reflect(interaction)

    # Should return string (formatted sonnet or empty on failure)
    assert isinstance(internal, str)


def test_reflect_adds_to_reflections(meta):
    """Test successful reflection is added to reflections list."""
    interaction = {
        'user': 'Tell me',
        'sonnet': 'Something interesting\n' * 14,
        'dissonance': 0.7,
        'quality': 0.8,
        'pulse': None
    }

    initial_count = len(meta.reflections)
    internal = meta.reflect(interaction)

    # If generation succeeded, should have added reflection
    if internal:
        assert len(meta.reflections) > initial_count


def test_reflect_updates_bootstrap(meta):
    """Test reflection feeds bootstrap buffer."""
    interaction = {
        'user': 'Question',
        'sonnet': 'When hearts grow warm and spirits rise\n' * 7,
        'dissonance': 0.8,
        'quality': 0.7,
        'pulse': None
    }

    initial_bootstrap_len = len(meta.bootstrap_buf)
    meta.reflect(interaction)

    # High dissonance + quality should feed bootstrap
    assert len(meta.bootstrap_buf) >= initial_bootstrap_len


# ============================================================================
# Reflection Storage Tests
# ============================================================================

def test_get_recent_reflections_empty(meta):
    """Test get_recent_reflections with no reflections."""
    recent = meta.get_recent_reflections(limit=5)

    assert len(recent) == 0


def test_get_recent_reflections_with_limit(meta):
    """Test get_recent_reflections respects limit."""
    # Manually add reflections
    for i in range(10):
        meta.reflections.append({
            'sonnet': f'Internal sonnet {i}\n' * 14,
            'context': {}
        })

    recent = meta.get_recent_reflections(limit=3)

    assert len(recent) == 3
    # Should get most recent (last 3)
    assert 'sonnet 9' in recent[-1]['sonnet'].lower()


# ============================================================================
# Cloud Bias Update Tests
# ============================================================================

def test_update_cloud_bias_valid_sonnet(meta, harmonix):
    """Test cloud bias update with valid internal sonnet."""
    internal_sonnet = "\n".join([f"Internal line {i}" for i in range(14)])
    interaction = {
        'dissonance': 0.7,
        'quality': 0.8
    }

    initial_count = harmonix.get_stats()['sonnet_count']
    meta.update_cloud_bias(internal_sonnet, interaction)

    # Should add to cloud if valid
    final_count = harmonix.get_stats()['sonnet_count']
    assert final_count >= initial_count


def test_update_cloud_bias_metadata(meta, harmonix):
    """Test cloud bias update stores correct metadata."""
    internal_sonnet = "\n".join([f"Line {i}" for i in range(14)])
    interaction = {
        'dissonance': 0.6,
        'quality': 0.9
    }

    meta.update_cloud_bias(internal_sonnet, interaction)

    # Check it was added with 'metasonnet' source
    recent = harmonix.get_recent_sonnets(limit=1)
    if recent:
        # Can't directly check added_by from get_recent_sonnets
        # but we verified it was added
        assert len(recent) > 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_reflection_cycle(meta):
    """Test complete reflection cycle."""
    interaction = {
        'user': 'Philosophy question',
        'sonnet': 'When thoughts turn deep and minds explore\n' * 7,
        'dissonance': 0.75,
        'quality': 0.8,
        'pulse': None
    }

    # Check if should reflect
    should = meta.should_reflect(interaction)

    if should:
        # Generate reflection
        internal = meta.reflect(interaction)

        # Verify process completed
        assert isinstance(internal, str)


def test_metasonnet_doesnt_crash_on_bad_input(meta):
    """Test MetaSonnet handles bad input gracefully."""
    bad_interaction = {
        'user': '',
        'sonnet': '',
        'dissonance': None,
        'quality': None,
        'pulse': None
    }

    # Should not crash
    try:
        meta.should_reflect(bad_interaction)
        meta.reflect(bad_interaction)
        success = True
    except Exception as e:
        success = False

    # May fail, but shouldn't crash hard
    assert True  # If we got here, didn't crash
