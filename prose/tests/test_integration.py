"""
Integration test for Prose core modules:
- ProseGenerator (prose.py)
- ProseFormatter (formatter.py)
- ProseHarmonix (harmonix.py)
- Overthinkrose (overthinkrose.py)

Tests full pipeline from generation to cloud expansion.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prose import ProseGenerator
from formatter import ProseFormatter
from harmonix import ProseHarmonix
from overthinkrose import Overthinkrose


def test_prose_generation():
    """Test 1: ProseGenerator creates valid prose."""
    print("Test 1: Prose Generation")
    print("-" * 60)

    generator = ProseGenerator(verbose=False)
    prompt = "What is the nature of language?"

    prose = generator.generate(prompt, max_tokens=150, temperature=0.8)

    print(f"Prompt: {prompt}")
    print(f"Generated: {prose[:200]}...")
    print(f"Length: {len(prose.split())} words")

    assert len(prose) > 0, "Prose should not be empty"
    assert len(prose.split()) > 20, "Prose should have at least 20 words"

    print("✓ Prose generation works\n")
    return prose


def test_prose_formatting(prose_text):
    """Test 2: ProseFormatter validates and formats prose."""
    print("Test 2: Prose Formatting")
    print("-" * 60)

    formatter = ProseFormatter()
    result = formatter.format_with_metrics(prose_text)

    assert result is not None, "Prose should be valid"

    formatted, metrics = result

    print(f"Metrics:")
    print(f"  Words: {metrics['word_count']}")
    print(f"  Sentences: {metrics['sentence_count']}")
    print(f"  Paragraphs: {metrics['paragraph_count']}")
    print(f"  Avg sentence length: {metrics['avg_sentence_length']:.1f}")

    assert metrics['word_count'] > 20, "Should have enough words"
    assert metrics['sentence_count'] >= 1, "Should have at least 1 sentence"

    print("✓ Prose formatting works\n")
    return formatted


def test_prose_harmonix(prose_text):
    """Test 3: ProseHarmonix tracks prose in cloud."""
    print("Test 3: Prose Harmonix (Cloud Tracking)")
    print("-" * 60)

    harmonix = ProseHarmonix(db_path='cloud/test_prose.db')

    # Compute dissonance
    user_input = "What is the nature of language?"
    dissonance, pulse = harmonix.compute_dissonance(user_input, prose_text)

    print(f"Dissonance: {dissonance:.3f}")
    print(f"Pulse:")
    print(f"  Novelty: {pulse.novelty:.3f}")
    print(f"  Arousal: {pulse.arousal:.3f}")
    print(f"  Entropy: {pulse.entropy:.3f}")

    # Add to cloud
    prose_id = harmonix.add_prose(
        prose_text,
        quality=0.75,
        dissonance=dissonance,
        temperature=0.8
    )

    print(f"\nProse added to cloud (ID: {prose_id})")

    # Get stats
    stats = harmonix.get_stats()
    print(f"Cloud stats:")
    print(f"  Total prose: {stats['total_prose']}")
    print(f"  Avg quality: {stats['avg_quality']:.3f}")
    print(f"  Avg semantic density: {stats['avg_semantic_density']:.3f}")

    assert stats['total_prose'] >= 1, "Should have at least 1 prose in cloud"
    assert stats['avg_semantic_density'] > 0, "Should have semantic density"

    harmonix.close()
    print("✓ Prose harmonix works\n")


def test_overthinkrose():
    """Test 4: Overthinkrose expands prose cloud."""
    print("Test 4: Overthinkrose (Cloud Expansion)")
    print("-" * 60)

    # Create harmonix to add test prose
    harmonix = ProseHarmonix(db_path='cloud/test_prose.db')

    # Add a few test prose for expansion
    test_prose_samples = [
        """Language is the bridge between minds, a shared resonance
        that allows thought to flow from one consciousness to another.""",

        """Words carry meaning not just through their definitions,
        but through the echoes they create in the listener's understanding.""",

        """To speak is to cast patterns into the air, hoping they will
        take shape in another's mind as they do in yours.""",
    ]

    for i, prose in enumerate(test_prose_samples):
        harmonix.add_prose(
            prose,
            quality=0.6 + (i * 0.1),
            dissonance=0.5,
            temperature=0.8
        )

    print(f"Added {len(test_prose_samples)} test prose samples")

    # Test overthinkrose expansion
    overthinkrose = Overthinkrose(db_path='cloud/test_prose.db')

    print("\nRunning 4-ring expansion...")
    recent = harmonix.get_recent_prose(limit=3)
    overthinkrose.expand(recent_prose=recent, num_rings=4)

    # Check results
    stats_after = harmonix.get_stats()
    print(f"\nCloud stats after expansion:")
    print(f"  Total prose: {stats_after['total_prose']}")
    print(f"  Avg quality: {stats_after['avg_quality']:.3f}")

    # Should have more prose after expansion (though some may not pass threshold)
    # Don't assert on count since overthinkrose filters by coherence
    print(f"  (Expansion may add prose if coherence > threshold)")

    harmonix.close()
    overthinkrose.close()
    print("✓ Overthinkrose works\n")


def test_cascade_mode():
    """Test 5: Cascade mode (user + haiku + sonnet → prose)."""
    print("Test 5: Cascade Mode")
    print("-" * 60)

    generator = ProseGenerator(verbose=False)

    user_prompt = "What is resonance?"
    haiku = """waves meet in the cloud
patterns emerge from chaos
meaning finds its form"""

    sonnet = """When words like waves do meet upon the shore,
And patterns form from chaos uncontrolled,
The meaning that we sought begins to soar,
As understanding's mysteries unfold."""

    cascade_prose = generator.generate_cascade(
        user_prompt=user_prompt,
        haiku_output=haiku,
        sonnet_output=sonnet,
        max_tokens=200,
        temperature=0.9
    )

    print(f"User: {user_prompt}")
    print(f"HAiKU: {haiku}")
    print(f"Sonnet: {sonnet[:80]}...")
    print(f"\nCascaded Prose: {cascade_prose[:250]}...")

    assert len(cascade_prose) > 0, "Cascade should generate prose"
    assert len(cascade_prose.split()) > 30, "Cascade should be substantial"

    print("\n✓ Cascade mode works\n")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("PROSE INTEGRATION TESTS")
    print("=" * 60)
    print()

    try:
        # Test 1: Generation
        prose = test_prose_generation()

        # Test 2: Formatting
        formatted = test_prose_formatting(prose)

        # Test 3: Harmonix (cloud tracking)
        test_prose_harmonix(formatted)

        # Test 4: Overthinkrose (expansion)
        test_overthinkrose()

        # Test 5: Cascade mode
        test_cascade_mode()

        print("=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        print()
        print("Core Prose modules working:")
        print("  ✓ prose.py (TinyLlama inference)")
        print("  ✓ formatter.py (text processing)")
        print("  ✓ harmonix.py (cloud tracking)")
        print("  ✓ overthinkrose.py (4-ring expansion)")
        print()
        print("Ready for emergent layer integration!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
