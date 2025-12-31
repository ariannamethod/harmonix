#!/usr/bin/env python3
"""
Temperature Tuning Test

Compare two temperature ranges:
- Range 1: 0.6-1.0 (current)
- Range 2: 0.7-1.2 (higher exploration)

Goal: Find which produces better sonnets (quality, coherence, creativity)
"""

from pathlib import Path
import sys
import time

SONNET_DIR = Path(__file__).parent.parent.absolute()  # Go up from tests/ to sonnet/
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator
from formatter import SonnetFormatter


def generate_at_temperature(temp: float, num_samples: int = 3):
    """Generate sonnets at specific temperature."""

    print(f"\n{'='*70}")
    print(f"TESTING TEMPERATURE: {temp:.2f}")
    print(f"{'='*70}\n")

    generator = SonnetGenerator()
    formatter = SonnetFormatter()

    results = []

    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}...")

        # Generate
        start = time.time()
        raw_output = generator.generate(
            prompt="\n",
            max_tokens=800,
            temperature=temp
        )
        gen_time = time.time() - start

        # Format
        sonnet = formatter.format(raw_output)

        if not sonnet:
            print(f"  ❌ Failed to format")
            continue

        # Validate
        is_valid, reason = formatter.validate(sonnet)
        quality = 0.8 if is_valid else 0.5

        # Count unique words (diversity metric)
        words = sonnet.lower().split()
        unique_words = len(set(words))
        diversity = unique_words / len(words) if words else 0

        # Coherence: check for very short lines (indicates broken generation)
        lines = sonnet.split('\n')
        avg_line_len = sum(len(line.split()) for line in lines) / len(lines)

        results.append({
            'temp': temp,
            'sonnet': sonnet,
            'quality': quality,
            'diversity': diversity,
            'avg_line_len': avg_line_len,
            'gen_time': gen_time,
            'valid': is_valid
        })

        print(f"  ✓ Generated (quality={quality:.2f}, diversity={diversity:.2f}, avg_line_len={avg_line_len:.1f})")

    generator.close()

    return results


def compare_ranges():
    """Compare two temperature ranges."""

    print("="*70)
    print("🌡️  TEMPERATURE TUNING TEST")
    print("="*70)
    print()
    print("Comparing two ranges:")
    print("  Range 1: [0.6, 0.8, 1.0] - Current")
    print("  Range 2: [0.7, 0.9, 1.2] - Higher exploration")
    print()

    # Range 1: Current (0.6-1.0)
    range1_temps = [0.6, 0.8, 1.0]
    range1_results = []

    print("\n" + "="*70)
    print("RANGE 1: 0.6-1.0 (Current)")
    print("="*70)

    for temp in range1_temps:
        results = generate_at_temperature(temp, num_samples=3)
        range1_results.extend(results)

    # Range 2: Higher (0.7-1.2)
    range2_temps = [0.7, 0.9, 1.2]
    range2_results = []

    print("\n" + "="*70)
    print("RANGE 2: 0.7-1.2 (Higher exploration)")
    print("="*70)

    for temp in range2_temps:
        results = generate_at_temperature(temp, num_samples=3)
        range2_results.extend(results)

    # Analysis
    print("\n" + "="*70)
    print("📊 COMPARISON ANALYSIS")
    print("="*70)

    print("\n--- RANGE 1 (0.6-1.0) ---")
    r1_quality = sum(r['quality'] for r in range1_results) / len(range1_results)
    r1_diversity = sum(r['diversity'] for r in range1_results) / len(range1_results)
    r1_line_len = sum(r['avg_line_len'] for r in range1_results) / len(range1_results)
    r1_valid = sum(1 for r in range1_results if r['valid']) / len(range1_results)

    print(f"Avg Quality:    {r1_quality:.3f}")
    print(f"Avg Diversity:  {r1_diversity:.3f}")
    print(f"Avg Line Len:   {r1_line_len:.1f} words")
    print(f"Valid Rate:     {r1_valid*100:.1f}%")

    print("\n--- RANGE 2 (0.7-1.2) ---")
    r2_quality = sum(r['quality'] for r in range2_results) / len(range2_results)
    r2_diversity = sum(r['diversity'] for r in range2_results) / len(range2_results)
    r2_line_len = sum(r['avg_line_len'] for r in range2_results) / len(range2_results)
    r2_valid = sum(1 for r in range2_results if r['valid']) / len(range2_results)

    print(f"Avg Quality:    {r2_quality:.3f}")
    print(f"Avg Diversity:  {r2_diversity:.3f}")
    print(f"Avg Line Len:   {r2_line_len:.1f} words")
    print(f"Valid Rate:     {r2_valid*100:.1f}%")

    # Winner
    print("\n" + "="*70)
    print("🏆 RECOMMENDATION")
    print("="*70 + "\n")

    # Scoring
    r1_score = (r1_quality * 0.4) + (r1_diversity * 0.3) + (r1_valid * 0.3)
    r2_score = (r2_quality * 0.4) + (r2_diversity * 0.3) + (r2_valid * 0.3)

    print(f"Range 1 Score: {r1_score:.3f}")
    print(f"Range 2 Score: {r2_score:.3f}\n")

    if r1_score > r2_score:
        print("✅ WINNER: Range 1 (0.6-1.0)")
        print("   - Better quality and structure")
        print("   - More reliable generation")
    else:
        print("✅ WINNER: Range 2 (0.7-1.2)")
        print("   - Higher diversity and exploration")
        print("   - More creative output")

    # Show example sonnets
    print("\n" + "="*70)
    print("📜 EXAMPLE SONNETS")
    print("="*70)

    # Best from each range
    best_r1 = max(range1_results, key=lambda r: r['quality'])
    best_r2 = max(range2_results, key=lambda r: r['quality'])

    print(f"\n--- Best from Range 1 (temp={best_r1['temp']:.2f}) ---")
    print(best_r1['sonnet'][:400] + "...")

    print(f"\n--- Best from Range 2 (temp={best_r2['temp']:.2f}) ---")
    print(best_r2['sonnet'][:400] + "...")

    print("\n✓ Temperature tuning complete!")


if __name__ == '__main__':
    compare_ranges()
