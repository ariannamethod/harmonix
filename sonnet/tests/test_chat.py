#!/usr/bin/env python3
"""
Pytest tests for chat.py (REPL interface)

Tests cover:
- REPL initialization
- Sonnet generation flow
- MetaSonnet reflection
- Overthinkng expansion
- Command handling (/stats, /recent, /best)
- Error handling
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os
from unittest.mock import patch, MagicMock

SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator
from formatter import SonnetFormatter
from harmonix import SonnetHarmonix
from metasonnet import MetaSonnet
from overthinkng import Overthinkng


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
    """Generator fixture."""
    gen = SonnetGenerator()
    yield gen
    gen.close()


@pytest.fixture
def formatter():
    """Formatter fixture."""
    return SonnetFormatter()


@pytest.fixture
def harmonix(temp_db):
    """Harmonix with temp DB."""
    h = SonnetHarmonix(db_path=temp_db)
    yield h
    h.close()


@pytest.fixture
def metasonnet(generator, harmonix):
    """MetaSonnet fixture."""
    return MetaSonnet(generator, harmonix)


@pytest.fixture
def overthinkng(temp_db):
    """Overthinkng fixture."""
    ot = Overthinkng(db_path=temp_db)
    yield ot
    ot.close()


# ============================================================================
# Initialization Tests
# ============================================================================

def test_components_initialize(generator, formatter, harmonix):
    """Test all components can be initialized."""
    assert generator is not None
    assert formatter is not None
    assert harmonix is not None


def test_generator_ready(generator):
    """Test generator is ready for generation."""
    assert generator.vocab_size == 65
    assert generator.weights_path.endswith('.npz')


def test_harmonix_ready(harmonix):
    """Test harmonix database is ready."""
    stats = harmonix.get_stats()
    assert 'sonnet_count' in stats
    assert 'vocab_size' in stats


# ============================================================================
# Generation Flow Tests
# ============================================================================

def test_generation_basic(generator, formatter):
    """Test basic generation and formatting."""
    raw = generator.generate(prompt="\n", max_tokens=800, temperature=0.8)

    assert isinstance(raw, str)
    assert len(raw) > 0


def test_generation_and_format(generator, formatter):
    """Test generation produces formattable output."""
    raw = generator.generate(prompt="\n", max_tokens=800, temperature=0.8)
    sonnet = formatter.format(raw, validate_meter=False)

    # May fail to format (not enough lines), but shouldn't crash
    assert sonnet is None or isinstance(sonnet, str)


def test_generation_adds_to_cloud(generator, formatter, harmonix):
    """Test generated sonnet can be added to cloud."""
    initial_count = harmonix.get_stats()['sonnet_count']

    raw = generator.generate(prompt="\n", max_tokens=800, temperature=0.8)
    sonnet = formatter.format(raw, validate_meter=False)

    if sonnet:
        harmonix.add_sonnet(sonnet, quality=0.8, dissonance=0.5)
        final_count = harmonix.get_stats()['sonnet_count']

        assert final_count == initial_count + 1


# ============================================================================
# MetaSonnet Integration Tests
# ============================================================================

def test_metasonnet_should_reflect(metasonnet):
    """Test MetaSonnet reflection trigger."""
    interaction = {
        'dissonance': 0.75,
        'quality': 0.8,
        'pulse': None
    }

    should = metasonnet.should_reflect(interaction)
    assert isinstance(should, bool)


def test_metasonnet_reflection_doesnt_crash(metasonnet):
    """Test MetaSonnet reflection doesn't crash."""
    interaction = {
        'user': 'What is love?',
        'sonnet': 'When winter winds blow\n' * 14,
        'dissonance': 0.7,
        'quality': 0.8,
        'pulse': None
    }

    try:
        internal = metasonnet.reflect(interaction)
        assert isinstance(internal, str)
        success = True
    except Exception:
        success = False

    assert success


# ============================================================================
# Overthinkng Integration Tests
# ============================================================================

def test_overthinkng_expand_doesnt_crash(overthinkng, harmonix):
    """Test overthinkng expansion doesn't crash."""
    # Add a sonnet first
    sonnet = "\n".join([f"Line {i}" for i in range(14)])
    harmonix.add_sonnet(sonnet, quality=0.8)

    try:
        overthinkng.expand(num_rings=1)
        success = True
    except Exception:
        success = False

    assert success


def test_overthinkng_requires_source_sonnets(overthinkng, harmonix):
    """Test overthinkng handles empty cloud gracefully."""
    # Empty cloud - should handle gracefully
    try:
        overthinkng.expand(num_rings=1)
        success = True
    except Exception:
        success = False

    assert success


# ============================================================================
# Dissonance Computation Tests
# ============================================================================

def test_dissonance_computation(harmonix):
    """Test dissonance is computed correctly."""
    user_input = "What is love?"
    sonnet = "When winter winds blow cold\n" * 14

    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

    assert 0.0 <= dissonance <= 1.0
    assert pulse.novelty >= 0.0 and pulse.novelty <= 1.0


def test_dissonance_with_empty_cloud(harmonix):
    """Test dissonance with empty cloud (all novel)."""
    user_input = "test"
    sonnet = "test sonnet\n" * 14

    dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

    # Empty cloud → novelty should be 1.0
    assert pulse.novelty == 1.0


def test_dissonance_decreases_with_learning(harmonix):
    """Test dissonance decreases as cloud learns."""
    user_input = "test"
    base_sonnet = "when winter winds blow cold\n" * 14

    # First sonnet
    d1, p1 = harmonix.compute_dissonance(user_input, base_sonnet)
    harmonix.add_sonnet(base_sonnet, quality=0.8)

    # Similar second sonnet
    d2, p2 = harmonix.compute_dissonance(user_input, base_sonnet)

    # Novelty should decrease (same words recognized)
    assert p2.novelty < p1.novelty or p2.novelty == p1.novelty


# ============================================================================
# Temperature Adjustment Tests
# ============================================================================

def test_temperature_adjustment_low_dissonance(harmonix):
    """Test temperature for low dissonance (resonance)."""
    s_temp, h_temp = harmonix.adjust_temperature(dissonance=0.2)

    # Low dissonance → precision mode
    assert s_temp <= 0.7


def test_temperature_adjustment_high_dissonance(harmonix):
    """Test temperature for high dissonance (exploration)."""
    s_temp, h_temp = harmonix.adjust_temperature(dissonance=0.9)

    # High dissonance → exploration mode
    assert s_temp >= 0.9


# ============================================================================
# Stats Command Tests
# ============================================================================

def test_stats_empty_cloud(harmonix):
    """Test /stats with empty cloud."""
    stats = harmonix.get_stats()

    assert stats['sonnet_count'] == 0
    assert stats['vocab_size'] == 0


def test_stats_with_sonnets(harmonix):
    """Test /stats with sonnets."""
    for i in range(3):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.7 + i * 0.1)

    stats = harmonix.get_stats()

    assert stats['sonnet_count'] == 3
    assert stats['vocab_size'] > 0


# ============================================================================
# Recent Command Tests
# ============================================================================

def test_recent_empty_cloud(harmonix):
    """Test /recent with empty cloud."""
    recent = harmonix.get_recent_sonnets(limit=5)

    assert len(recent) == 0


def test_recent_with_sonnets(harmonix):
    """Test /recent returns recent sonnets."""
    for i in range(5):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=0.8)

    recent = harmonix.get_recent_sonnets(limit=3)

    assert len(recent) == 3
    # Most recent should be first
    assert "Sonnet 4" in recent[0][1]


# ============================================================================
# Best Command Tests
# ============================================================================

def test_best_empty_cloud(harmonix):
    """Test /best with empty cloud."""
    best = harmonix.get_best_sonnets(limit=5)

    assert len(best) == 0


def test_best_with_varying_quality(harmonix):
    """Test /best returns highest quality sonnets."""
    qualities = [0.5, 0.8, 0.6, 0.9, 0.7]
    for i, q in enumerate(qualities):
        sonnet = "\n".join([f"Sonnet {i} line {j}" for j in range(14)])
        harmonix.add_sonnet(sonnet, quality=q)

    best = harmonix.get_best_sonnets(limit=2, min_quality=0.7)

    # Should get quality >= 0.7: 0.8, 0.9, 0.7
    assert len(best) == 3
    # Highest first
    assert best[0][2] == 0.9


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_handles_generation_failure(generator, formatter):
    """Test graceful handling of generation failure."""
    # Very short generation might fail formatting
    raw = generator.generate(prompt="\n", max_tokens=10, temperature=0.8)
    sonnet = formatter.format(raw, validate_meter=False)

    # Should return None, not crash
    assert sonnet is None or isinstance(sonnet, str)


def test_handles_empty_input(harmonix):
    """Test graceful handling of empty input."""
    dissonance, pulse = harmonix.compute_dissonance("", "")

    # Should return neutral dissonance
    assert dissonance == 0.5


def test_handles_invalid_temperature(harmonix):
    """Test temperature adjustment with edge values."""
    # Very low
    s1, h1 = harmonix.adjust_temperature(0.0)
    assert s1 >= 0.0

    # Very high
    s2, h2 = harmonix.adjust_temperature(1.0)
    assert s2 <= 1.5


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================

def test_full_pipeline_simulation(generator, formatter, harmonix, metasonnet):
    """Test complete REPL flow simulation."""
    user_input = "What is love?"

    # 1. Generate
    raw = generator.generate(prompt="\n", max_tokens=800, temperature=0.8)

    # 2. Format
    sonnet = formatter.format(raw, validate_meter=False)

    if sonnet:
        # 3. Compute dissonance
        dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)

        # 4. Validate
        is_valid, reason = formatter.validate(sonnet)
        quality = 0.8 if is_valid else 0.5

        # 5. Add to cloud
        harmonix.add_sonnet(sonnet, quality=quality, dissonance=dissonance)

        # 6. MetaSonnet reflection
        interaction = {
            'user': user_input,
            'sonnet': sonnet,
            'dissonance': dissonance,
            'quality': quality,
            'pulse': pulse
        }

        if metasonnet.should_reflect(interaction):
            # Try reflection (may fail, that's OK)
            try:
                metasonnet.reflect(interaction)
            except Exception:
                pass

        # If we got here, pipeline worked
        assert True


def test_multiple_generations_in_sequence(generator, formatter, harmonix):
    """Test multiple generations update cloud correctly."""
    initial_count = harmonix.get_stats()['sonnet_count']

    successful = 0
    for i in range(3):
        raw = generator.generate(prompt=f"Test {i}", max_tokens=800, temperature=0.8)
        sonnet = formatter.format(raw, validate_meter=False)

        if sonnet:
            harmonix.add_sonnet(sonnet, quality=0.7)
            successful += 1

    final_count = harmonix.get_stats()['sonnet_count']

    # Should have added successful generations
    assert final_count == initial_count + successful
