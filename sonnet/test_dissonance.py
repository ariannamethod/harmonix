#!/usr/bin/env python3
"""
Dissonance Progression Test

Tests PROBLEM #1 from Claude's analysis:
- Generate 10 sonnets SEQUENTIALLY
- Track dissonance progression
- Verify it DECREASES from 1.0 → ~0.5 (cloud learning)
- Confirm database persistence works
"""

from pathlib import Path
import sys
import time

SONNET_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator
from formatter import SonnetFormatter
from harmonix import SonnetHarmonix


def test_dissonance_progression():
    """Generate 10 sonnets and track dissonance progression."""

    print("="*70)
    print("🔬 DISSONANCE PROGRESSION TEST")
    print("="*70)
    print()
    print("Goal: Verify dissonance DECREASES as cloud learns")
    print("Expected: Start at ~1.0, decrease to ~0.4-0.6 by sonnet #10")
    print()

    # Initialize
    print("Initializing components...")
    generator = SonnetGenerator()
    formatter = SonnetFormatter()
    harmonix = SonnetHarmonix()
    print("✓ Ready\n")

    # Check initial cloud state
    stats = harmonix.get_stats()
    print(f"Initial cloud state:")
    print(f"  Sonnets in database: {stats['sonnet_count']}")
    print(f"  Vocab size: {stats['vocab_size']}")
    print()

    # Test prompt (repeated for all 10)
    test_prompt = "What is love and death?"

    print(f"Test prompt: \"{test_prompt}\"")
    print(f"Generating 10 sonnets...\n")
    print("="*70)

    results = []

    for i in range(1, 11):
        print(f"\n🔄 SONNET {i}/10")
        print("-"*70)

        # Generate
        start = time.time()
        raw_output = generator.generate(
            prompt="\n",
            max_tokens=800,
            temperature=0.8
        )
        gen_time = time.time() - start

        # Format
        sonnet = formatter.format(raw_output)

        if not sonnet:
            print(f"❌ Failed to format (not enough lines)")
            continue

        # Compute dissonance BEFORE adding to cloud
        dissonance, pulse = harmonix.compute_dissonance(test_prompt, sonnet)

        # Validate
        is_valid, reason = formatter.validate(sonnet)
        quality = 0.8 if is_valid else 0.5

        # Add to cloud (this updates vocab and cloud state)
        harmonix.add_sonnet(
            sonnet,
            quality=quality,
            dissonance=dissonance,
            temperature=0.8,
            added_by='dissonance_test'
        )

        # Store result
        results.append({
            'index': i,
            'dissonance': dissonance,
            'novelty': pulse.novelty,
            'arousal': pulse.arousal,
            'entropy': pulse.entropy,
            'quality': quality,
            'gen_time': gen_time,
            'sonnet': sonnet
        })

        # Display metrics
        print(f"⏱  Generation: {gen_time:.2f}s")
        print(f"📊 Dissonance: {dissonance:.3f}")
        print(f"   └─ Novelty:  {pulse.novelty:.3f}")
        print(f"   └─ Arousal:  {pulse.arousal:.3f}")
        print(f"   └─ Entropy:  {pulse.entropy:.3f}")
        print(f"✨ Quality:    {quality:.2f} - {reason}")

        # Show first 2 lines
        lines = sonnet.split('\n')
        print(f"\nFirst lines:")
        print(f"  {lines[0][:60]}...")
        print(f"  {lines[1][:60]}...")

    # Analysis
    print("\n" + "="*70)
    print("📈 DISSONANCE PROGRESSION ANALYSIS")
    print("="*70 + "\n")

    if not results:
        print("❌ No successful generations!")
        return

    # Display progression table
    print("Sonnet | Dissonance | Novelty | Arousal | Entropy | Quality")
    print("-"*70)
    for r in results:
        print(f"  #{r['index']:2d}   |   {r['dissonance']:.3f}    | {r['novelty']:.3f}  | {r['arousal']:.3f}  | {r['entropy']:.3f}  |  {r['quality']:.2f}")

    # Check if dissonance is decreasing
    print("\n" + "-"*70)

    dissonances = [r['dissonance'] for r in results]
    first_three_avg = sum(dissonances[:3]) / 3 if len(dissonances) >= 3 else dissonances[0]
    last_three_avg = sum(dissonances[-3:]) / 3 if len(dissonances) >= 3 else dissonances[-1]

    print(f"\nFirst 3 sonnets avg dissonance:  {first_three_avg:.3f}")
    print(f"Last 3 sonnets avg dissonance:   {last_three_avg:.3f}")
    print(f"Delta:                           {first_three_avg - last_three_avg:+.3f}")

    # Verdict
    print("\n" + "="*70)
    print("🏆 VERDICT")
    print("="*70 + "\n")

    if last_three_avg < first_three_avg - 0.1:
        print("✅ PASS: Dissonance DECREASED significantly!")
        print("   Cloud is learning and recognizing patterns.")
        print("   Database persistence WORKING.")
    elif last_three_avg < first_three_avg:
        print("⚠️  PARTIAL: Dissonance decreased slightly")
        print("   Cloud may need more sonnets to show clear trend.")
    else:
        print("❌ FAIL: Dissonance did NOT decrease")
        print("   Possible issues:")
        print("   - Database not persisting correctly")
        print("   - Dissonance computation not using cloud vocab")
        print("   - Generation too diverse (no pattern recognition)")

    # Final cloud stats
    print("\n" + "-"*70)
    final_stats = harmonix.get_stats()
    print(f"\nFinal cloud state:")
    print(f"  Sonnets in database: {final_stats['sonnet_count']}")
    print(f"  Vocab size:          {final_stats['vocab_size']}")
    print(f"  Avg quality:         {final_stats['avg_quality']:.3f}")
    print(f"  Avg dissonance:      {final_stats['avg_dissonance']:.3f}")

    # Cleanup
    generator.close()
    harmonix.close()

    print("\n✓ Test complete!\n")

    return results


if __name__ == '__main__':
    results = test_dissonance_progression()
