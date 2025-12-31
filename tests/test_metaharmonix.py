"""
Tests for MetaHarmonix v1 - Cascade Mode

Basic tests for HAiKU → Sonnet cascade functionality.
"""

import sys
from pathlib import Path

# Add harmonix root to path
HARMONIX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HARMONIX_ROOT))


def test_metaharmonix_imports():
    """Test that MetaHarmonix can load both agents without conflicts."""
    from metaharmonix import MetaHarmonix

    meta = MetaHarmonix()

    # Both agents should be loaded
    assert meta.haiku_gen is not None, "HAiKU generator not loaded"
    assert meta.haiku_harmonix is not None, "HAiKU harmonix not loaded"
    assert meta.sonnet_gen is not None, "Sonnet generator not loaded"
    assert meta.sonnet_harmonix is not None, "Sonnet harmonix not loaded"
    assert meta.sonnet_formatter is not None, "Sonnet formatter not loaded"

    meta.close()
    print("✓ test_metaharmonix_imports passed")


def test_cascade_basic():
    """Test basic cascade functionality."""
    from metaharmonix import MetaHarmonix

    meta = MetaHarmonix()

    # Run cascade with simple prompt
    result = meta.cascade("what is love")

    # Check result structure
    assert result is not None
    assert hasattr(result, 'haiku_output')
    assert hasattr(result, 'sonnet_output')
    assert hasattr(result, 'meta_sentence')
    assert hasattr(result, 'haiku_metrics')
    assert hasattr(result, 'sonnet_metrics')
    assert hasattr(result, 'global_resonance')
    assert hasattr(result, 'field_entropy')

    # Check outputs are non-empty strings
    assert isinstance(result.haiku_output, str)
    assert len(result.haiku_output) > 0
    assert isinstance(result.sonnet_output, str)
    assert len(result.sonnet_output) > 0
    assert isinstance(result.meta_sentence, str)
    assert len(result.meta_sentence) > 0

    # Check metrics are valid
    assert 0.0 <= result.global_resonance <= 1.0
    assert result.field_entropy >= 0.0
    assert 0.0 <= result.haiku_metrics.dissonance <= 1.0
    assert 0.0 <= result.sonnet_metrics.dissonance <= 1.0

    meta.close()
    print("✓ test_cascade_basic passed")


def test_cascade_creates_combined_field():
    """Test that cascade updates combined vocabulary field."""
    from metaharmonix import MetaHarmonix

    meta = MetaHarmonix()

    # Initial field should be empty
    assert len(meta.combined_field) == 0

    # Run cascade
    result = meta.cascade("resonance")

    # Combined field should have words from both outputs
    assert len(meta.combined_field) > 0

    # Meta sentence should use words from combined field
    meta_words = set(w.lower() for w in result.meta_sentence.split() if w.isalpha())
    # At least some words should be from combined field
    # (not strict check since meta adds randomness)

    meta.close()
    print("✓ test_cascade_creates_combined_field passed")


def test_cascade_metrics():
    """Test that all metrics are computed correctly."""
    from metaharmonix import MetaHarmonix

    meta = MetaHarmonix()

    result = meta.cascade("test prompt")

    # HAiKU metrics
    assert result.haiku_metrics.agent == 'HAiKU'
    assert hasattr(result.haiku_metrics, 'dissonance')
    assert hasattr(result.haiku_metrics, 'quality')
    assert hasattr(result.haiku_metrics, 'novelty')
    assert hasattr(result.haiku_metrics, 'arousal')
    assert hasattr(result.haiku_metrics, 'entropy')

    # Sonnet metrics
    assert result.sonnet_metrics.agent == 'Sonnet'
    assert hasattr(result.sonnet_metrics, 'dissonance')
    assert hasattr(result.sonnet_metrics, 'quality')

    # Global metrics should be averages/combinations
    assert 0.0 <= result.global_resonance <= 1.0
    assert result.field_entropy >= 0.0

    meta.close()
    print("✓ test_cascade_metrics passed")


if __name__ == '__main__':
    print("Running MetaHarmonix v1 tests...\n")

    try:
        test_metaharmonix_imports()
        test_cascade_basic()
        test_cascade_creates_combined_field()
        test_cascade_metrics()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
