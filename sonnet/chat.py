#!/usr/bin/env python3
"""
Sonnet REPL - Shakespeare sonnet generator

Autonomous module for 14-line Shakespeare sonnets.
Uses NanoGPT weights with pure numpy inference.
"""

from pathlib import Path
import sys

# Get absolute paths
SONNET_DIR = Path(__file__).parent.absolute()
STATE_DIR = SONNET_DIR / 'state'
sys.path.insert(0, str(SONNET_DIR))

from sonnet import SonnetGenerator
from formatter import SonnetFormatter
from harmonix import SonnetHarmonix
from overthinkng import Overthinkng
from metasonnet import MetaSonnet


def main():
    """Main REPL loop."""
    print("🎭 Sonnet Generator - Shakespeare AI")
    print("="*70)
    print("Type your prompt, get a 14-line sonnet!")
    print("Commands: /stats, /recent, /best, /quit")
    print("="*70 + "\n")

    # Initialize components
    print("Initializing Sonnet Generator...")
    generator = SonnetGenerator()
    formatter = SonnetFormatter()
    harmonix = SonnetHarmonix()
    overthinkng = Overthinkng()
    meta = MetaSonnet(generator, harmonix)

    print("✓ Ready!\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                handle_command(user_input, harmonix)
                continue

            # Generate sonnet
            print("\n🔄 Generating sonnet...\n")

            # Get prompt from user input (first few words)
            prompt_words = user_input.split()[:5]
            prompt = ' '.join(prompt_words) if prompt_words else "\n"

            # Generate raw output (with character headers)
            # FIX: Use empty prompt + 800 tokens + temp from range 0.7-1.2
            raw_output = generator.generate(
                prompt="\n",  # Empty prompt (model generates from scratch)
                max_tokens=800,  # Need enough text after header removal
                temperature=sonnet_temp if 'sonnet_temp' in locals() else 0.8
            )

            # Format to clean 14-line sonnet
            sonnet = formatter.format(raw_output)

            if not sonnet:
                print("❌ Could not generate valid sonnet (not enough lines)")
                continue

            # Display sonnet
            print("Sonnet:")
            print("-" * 70)
            print(sonnet)
            print("-" * 70)

            # Compute dissonance
            dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)
            print(f"\nDissonance: {dissonance:.3f} (novelty={pulse.novelty:.2f}, "
                  f"arousal={pulse.arousal:.2f}, entropy={pulse.entropy:.2f})")

            # Adjust temperature for next generation
            sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)

            # Validate and compute quality
            is_valid, reason = formatter.validate(sonnet)
            quality = 0.8 if is_valid else 0.5

            print(f"Quality: {quality:.2f} - {reason}")

            # Add to database
            harmonix.add_sonnet(
                sonnet,
                quality=quality,
                dissonance=dissonance,
                temperature=0.8,
                added_by='user'
            )

            # Internal reflection (metasonnet)
            interaction = {
                'user': user_input,
                'sonnet': sonnet,
                'dissonance': dissonance,
                'quality': quality,
                'pulse': pulse
            }

            if meta.should_reflect(interaction):
                print("\n💭 Generating internal reflection...")
                internal = meta.reflect(interaction)
                if internal:
                    print("✓ Internal sonnet generated (not shown)")

            # Background expansion (overthinkng)
            if len(harmonix.get_recent_sonnets(limit=3)) >= 2:
                print("\n🔄 Running background expansion (overthinkng)...")
                overthinkng.expand()
                print("✓ Sonnet cloud expanded")

            print()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        generator.close()
        harmonix.close()
        overthinkng.close()
        print("\n✓ State saved to database")
        print("Goodbye! 🎭")


def handle_command(command: str, harmonix: SonnetHarmonix):
    """Handle special commands."""
    if command == '/quit':
        sys.exit(0)

    elif command == '/stats':
        stats = harmonix.get_stats()
        print(f"\n📊 Sonnet Cloud Stats:")
        print(f"  Sonnet count: {stats['sonnet_count']}")
        print(f"  Avg quality: {stats['avg_quality']:.3f}")
        print(f"  Avg dissonance: {stats['avg_dissonance']:.3f}")
        print(f"  Vocab size: {stats['vocab_size']}")
        print()

    elif command == '/recent':
        recent = harmonix.get_recent_sonnets(limit=3)
        print(f"\n📜 Recent Sonnets:")
        for i, (sid, text, quality) in enumerate(recent, 1):
            print(f"\n[{i}] Sonnet #{sid} (quality={quality:.2f}):")
            # Show first 2 lines
            lines = text.split('\n')
            print(f"  {lines[0]}")
            print(f"  {lines[1]}")
            print(f"  ...")
        print()

    elif command == '/best':
        best = harmonix.get_best_sonnets(limit=3, min_quality=0.7)
        print(f"\n⭐ Best Sonnets:")
        if not best:
            print("  (none yet - keep generating!)")
        for i, (sid, text, quality) in enumerate(best, 1):
            print(f"\n[{i}] Sonnet #{sid} (quality={quality:.2f}):")
            print(text)
            print()
        print()

    else:
        print(f"Unknown command: {command}")
        print("Available: /stats, /recent, /best, /quit\n")


if __name__ == '__main__':
    main()
