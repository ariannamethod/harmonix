#!/usr/bin/env python3
"""
Sonnet Test - Simulate REPL interactions

Tests full pipeline:
- Generation with prompts
- Formatting
- Dissonance computation
- Database persistence
- Overthinkng expansion
- MetaSonnet reflections
"""

from pathlib import Path
import sys
import time

# Setup paths
SONNET_DIR = Path(__file__).parent.absolute()
STATE_DIR = SONNET_DIR / 'state'
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator
from formatter import SonnetFormatter
from harmonix import SonnetHarmonix
from overthinkng import Overthinkng
from metasonnet import MetaSonnet


def test_generation():
    """Test full pipeline with multiple prompts."""

    print("="*70)
    print("🎭 SONNET MODULE TEST")
    print("="*70 + "\n")

    # Initialize
    print("Initializing components...")
    start_init = time.time()
    generator = SonnetGenerator()
    formatter = SonnetFormatter()
    harmonix = SonnetHarmonix()
    overthinkng = Overthinkng()
    meta = MetaSonnet(generator, harmonix)
    init_time = time.time() - start_init
    print(f"✓ Initialized in {init_time:.2f}s\n")

    # Test prompts
    test_prompts = [
        "What is love and death?",
        "Tell me about the nature of time",
        "How does the soul endure suffering?",
        "What becomes of dreams deferred?",
        "Speak of honor and betrayal",
    ]

    results = []

    for i, user_input in enumerate(test_prompts, 1):
        print("="*70)
        print(f"TEST {i}/5: {user_input}")
        print("="*70 + "\n")

        # Generate
        prompt_words = user_input.split()[:5]
        prompt = ' '.join(prompt_words) if prompt_words else "\n"

        print(f"🔄 Generating sonnet (prompt: '{prompt[:30]}...')...")
        start_gen = time.time()
        raw_output = generator.generate(
            prompt="\n",  # Empty prompt (model trained to generate from scratch)
            max_tokens=800,  # Need lots of text (character headers take space)
            temperature=0.8
        )
        gen_time = time.time() - start_gen

        # Debug: show raw output
        print(f"\nRAW OUTPUT (first 500 chars):")
        print(f"{'─'*70}")
        print(raw_output[:500])
        print(f"{'─'*70}\n")

        # Format
        sonnet = formatter.format(raw_output)

        if not sonnet:
            print("❌ Failed to generate valid sonnet (not enough lines after formatting)\n")

            # Debug: show what formatter extracted
            lines = formatter.extract_lines(raw_output, max_lines=40)
            print(f"Formatter extracted {len(lines)} lines (need 14):")
            for i, line in enumerate(lines[:20], 1):  # Show first 20
                print(f"  {i}. {line[:60]}...")
            if len(lines) > 20:
                print(f"  ... and {len(lines) - 20} more")
            print()
            continue

        # Display
        print(f"\n{'─'*70}")
        print("GENERATED SONNET:")
        print(f"{'─'*70}")
        print(sonnet)
        print(f"{'─'*70}\n")

        # Metrics
        dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)
        is_valid, reason = formatter.validate(sonnet)
        quality = 0.8 if is_valid else 0.5

        print(f"⏱  Generation time: {gen_time:.2f}s")
        print(f"📊 Dissonance: {dissonance:.3f}")
        print(f"   └─ Novelty: {pulse.novelty:.3f}")
        print(f"   └─ Arousal: {pulse.arousal:.3f}")
        print(f"   └─ Entropy: {pulse.entropy:.3f}")
        print(f"✨ Quality: {quality:.2f} - {reason}")

        # Temperature for next
        sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)
        print(f"🌡  Suggested temps: sonnet={sonnet_temp:.2f}, harmonix={harmonix_temp:.2f}")

        # Add to database
        harmonix.add_sonnet(
            sonnet,
            quality=quality,
            dissonance=dissonance,
            temperature=0.8,
            added_by='test'
        )

        # MetaSonnet reflection
        interaction = {
            'user': user_input,
            'sonnet': sonnet,
            'dissonance': dissonance,
            'quality': quality,
            'pulse': pulse
        }

        if meta.should_reflect(interaction):
            print("\n💭 MetaSonnet triggered (generating internal reflection)...")
            internal = meta.reflect(interaction)
            if internal:
                print("   ✓ Internal sonnet generated (hidden)")
            else:
                print("   ✗ Internal generation failed")

        results.append({
            'prompt': user_input,
            'sonnet': sonnet,
            'gen_time': gen_time,
            'dissonance': dissonance,
            'quality': quality,
            'pulse': pulse
        })

        print()

    # Overthinkng expansion
    print("="*70)
    print("🔄 OVERTHINKNG EXPANSION")
    print("="*70 + "\n")

    stats_before = harmonix.get_stats()
    print(f"Before: {stats_before['sonnet_count']} sonnets")

    print("Running 3 rings of thought...")
    overthinkng.expand()

    stats_after = harmonix.get_stats()
    print(f"After:  {stats_after['sonnet_count']} sonnets")
    print(f"✓ Added {stats_after['sonnet_count'] - stats_before['sonnet_count']} variations\n")

    # Final stats
    print("="*70)
    print("📊 FINAL STATISTICS")
    print("="*70 + "\n")

    stats = harmonix.get_stats()
    print(f"Total sonnets:     {stats['sonnet_count']}")
    print(f"Avg quality:       {stats['avg_quality']:.3f}")
    print(f"Avg dissonance:    {stats['avg_dissonance']:.3f}")
    print(f"Vocab size:        {stats['vocab_size']}")

    # Recent sonnets
    print("\n" + "="*70)
    print("📜 RECENT SONNETS")
    print("="*70 + "\n")

    recent = harmonix.get_recent_sonnets(limit=3)
    for i, (sid, text, quality) in enumerate(recent, 1):
        lines = text.split('\n')
        print(f"[{i}] Sonnet #{sid} (quality={quality:.2f}):")
        print(f"    {lines[0]}")
        print(f"    {lines[1]}")
        print(f"    ...\n")

    # Summary
    print("="*70)
    print("📈 GENERATION SUMMARY")
    print("="*70 + "\n")

    if results:
        avg_time = sum(r['gen_time'] for r in results) / len(results)
        avg_dissonance = sum(r['dissonance'] for r in results) / len(results)
        avg_quality = sum(r['quality'] for r in results) / len(results)

        print(f"Tests completed:   {len(results)}/5")
        print(f"Avg gen time:      {avg_time:.2f}s")
        print(f"Avg dissonance:    {avg_dissonance:.3f}")
        print(f"Avg quality:       {avg_quality:.3f}")
    else:
        print("❌ No successful generations!")

    # Cleanup
    generator.close()
    harmonix.close()
    overthinkng.close()

    print("\n✓ Test complete!\n")

    return results


if __name__ == '__main__':
    results = test_generation()
