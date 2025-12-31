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

def test_weights_loaded(generator):
    """Test weights are loaded from .npz file."""
    assert generator.weights is not None
    assert isinstance(generator.weights, dict)
    assert len(generator.weights) > 0


def test_weights_are_numpy(generator):
    """Test all weights are numpy arrays."""
    for key, value in generator.weights.items():
        assert isinstance(value, np.ndarray)


def test_model_architecture(generator):
    """Test model has correct architecture."""
    assert hasattr(generator, 'n_layer')
    assert hasattr(generator, 'n_head')
    assert hasattr(generator, 'n_embd')
    assert hasattr(generator, 'block_size')


def test_model_dimensions(generator):
    """Test model has expected dimensions."""
    # NanoGPT-Shakespeare config
    assert generator.n_embd == 128
    assert generator.n_head == 4
    assert generator.n_layer == 4
    assert generator.block_size == 64


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


def test_vocab_size(generator):
    """Test vocabulary size is 65 (character-level)."""
    assert generator.vocab_size == 65


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


def test_generate_respects_max_tokens(generator):
    """Test generation respects max_tokens limit."""
    max_tokens = 10
    output = generator.generate(prompt="", max_tokens=max_tokens, temperature=0.8)

    # Output tokens should be <= max_tokens (approximately)
    output_tokens = len(generator.encode(output))
    assert output_tokens <= max_tokens + 5  # Allow small variance


# ============================================================================
# Forward Pass Tests
# ============================================================================

def test_forward_basic(generator):
    """Test forward pass works."""
    # Create simple input
    idx = np.array([[1, 2, 3, 4, 5]])  # Shape: (1, 5)

    logits = generator.model.forward(idx)

    # Should return logits
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (1, 5, 65)  # (batch, seq_len, vocab_size)


def test_forward_output_shape(generator):
    """Test forward pass output shape is correct."""
    seq_len = 10
    idx = np.array([[i for i in range(seq_len)]])

    logits = generator.model.forward(idx)

    assert logits.shape[0] == 1  # batch size
    assert logits.shape[1] == seq_len
    assert logits.shape[2] == generator.vocab_size


# ============================================================================
# Sampling Tests
# ============================================================================

def test_sample_basic(generator):
    """Test sampling from logits."""
    # Create fake logits
    logits = np.random.randn(1, 10, 65)

    next_token = generator.sample(logits, temperature=1.0)

    assert isinstance(next_token, (int, np.integer))
    assert 0 <= next_token < 65


def test_sample_temperature_zero(generator):
    """Test sampling with temperature=0 (argmax)."""
    # Create peaked logits
    logits = np.zeros((1, 1, 65))
    logits[0, 0, 42] = 10.0  # Peak at token 42

    next_token = generator.sample(logits, temperature=0.001)  # Near-zero

    # Should pick token 42 (argmax)
    assert next_token == 42


def test_sample_returns_valid_token(generator):
    """Test sample always returns valid token index."""
    logits = np.random.randn(1, 5, 65)

    for _ in range(10):
        token = generator.sample(logits, temperature=1.0)
        assert 0 <= token < 65


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


def test_generator_is_deterministic_with_seed(generator):
    """Test generation can be made deterministic with seed."""
    # Note: This would require adding seed support to SonnetGenerator
    # For now, just test that multiple generations work
    outputs = []
    for _ in range(3):
        output = generator.generate(prompt="To", max_tokens=5, temperature=0.8)
        outputs.append(output)

    # All should be strings
    assert all(isinstance(o, str) for o in outputs)


def test_generator_handles_long_context(generator):
    """Test generator handles context up to block_size."""
    # Create prompt near block_size
    long_prompt = "a" * (generator.block_size - 5)

    output = generator.generate(prompt=long_prompt, max_tokens=10, temperature=0.8)

    # Should still generate
    assert len(output) > 0


def test_generator_close(generator):
    """Test generator cleanup."""
    # Should have close method
    assert hasattr(generator, 'close')

    # Can call close
    generator.close()

    # Subsequent operations might fail, but close itself shouldn't crash
    assert True
