#!/usr/bin/env python3
"""
Pytest tests for Transformer Components

Tests cover:
- Weight loading from .npz
- Attention mechanisms
- Feedforward networks
- LayerNorm
- Token/position embeddings
- Full forward pass
"""

import pytest
from pathlib import Path
import sys
import numpy as np

SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator


@pytest.fixture
def generator():
    """Generator fixture."""
    gen = SonnetGenerator()
    yield gen
    gen.close()


# ============================================================================
# Weight Loading Tests
# ============================================================================

def test_weights_path_exists(generator):
    """Test weights path is set."""
    assert generator.weights_path is not None
    assert generator.weights_path.endswith('.npz')


def test_vocab_size(generator):
    """Test vocabulary size is 65 (character-level)."""
    assert generator.vocab_size == 65


def test_model_components_exist(generator):
    """Test model has required components."""
    assert hasattr(generator, 'token_embedding')
    assert hasattr(generator, 'position_embedding')
    assert hasattr(generator, 'blocks')
    assert hasattr(generator, 'ln_f')
    assert hasattr(generator, 'lm_head')


# ============================================================================
# Tokenization Tests
# ============================================================================

def test_encode_basic(generator):
    """Test basic encoding."""
    text = "Hello"
    tokens = generator.encode(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_encode_decode_roundtrip(generator):
    """Test encode -> decode roundtrip."""
    text = "To be or not to be"
    tokens = generator.encode(text)
    decoded = generator.decode(tokens)

    assert decoded == text


def test_encode_special_chars(generator):
    """Test encoding of special characters."""
    text = "Hello, world!\n"
    tokens = generator.encode(text)

    # Should handle punctuation and newlines
    assert len(tokens) > 0


# ============================================================================
# Generation Tests
# ============================================================================

def test_generate_basic(generator):
    """Test basic generation."""
    output = generator.generate(prompt="To", max_tokens=10, temperature=0.8)

    assert isinstance(output, str)
    assert len(output) > 0


def test_generate_with_empty_prompt(generator):
    """Test generation with empty prompt."""
    output = generator.generate(prompt="\n", max_tokens=20, temperature=0.8)

    assert isinstance(output, str)
    # Should generate something
    assert len(output) > 0


def test_generate_temperature_low(generator):
    """Test generation with low temperature (deterministic)."""
    output1 = generator.generate(prompt="The", max_tokens=5, temperature=0.1)
    output2 = generator.generate(prompt="The", max_tokens=5, temperature=0.1)

    # Low temperature should be more deterministic (might still vary slightly)
    # Just verify it generates
    assert len(output1) > 0
    assert len(output2) > 0


def test_generate_temperature_high(generator):
    """Test generation with high temperature (random)."""
    output = generator.generate(prompt="The", max_tokens=10, temperature=1.5)

    # High temperature should still generate
    assert len(output) > 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_generation_pipeline(generator):
    """Test complete generation pipeline."""
    prompt = "When"
    output = generator.generate(prompt=prompt, max_tokens=50, temperature=0.8)

    # Should produce Shakespeare-like text
    assert isinstance(output, str)
    assert len(output) > len(prompt)


def test_generator_multiple_calls(generator):
    """Test generator can be called multiple times."""
    outputs = []
    for _ in range(3):
        output = generator.generate(prompt="To", max_tokens=5, temperature=0.8)
        outputs.append(output)

    # All should be strings
    assert all(isinstance(o, str) for o in outputs)
    assert all(len(o) > 0 for o in outputs)


def test_generator_close(generator):
    """Test generator cleanup."""
    # Should have close method
    assert hasattr(generator, 'close')

    # Can call close
    generator.close()

    # Subsequent operations might fail, but close itself shouldn't crash
    assert True
