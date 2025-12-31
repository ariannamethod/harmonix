#!/usr/bin/env python3
"""
Prose Chat REPL - Language Organism Interface

User input wrinkles the field, prose resonates from field state.
NO SEED FROM PROMPT - organism mode, not Q&A chatbot!
"""

import sys
from prose import ProseGenerator
from harmonix import ProseHarmonix


def print_header():
    """Print Prose REPL header."""
    print("=" * 60)
    print("🌊 PROSE - Free-form Language Organism")
    print("=" * 60)
    print()
    print("Architecture: TinyLlama 1.1B (783 MB)")
    print("Mode: Field-based generation (no seed from prompt)")
    print()
    print("Commands:")
    print("  /stats    - Cloud statistics")
    print("  /recent   - Recent prose")
    print("  /best     - Highest quality prose")
    print("  /seed     - Show current field seed")
    print("  /temp     - Show current temperature")
    print("  /quit     - Exit")
    print()
    print("=" * 60)
    print()


def show_stats(harmonix):
    """Show cloud statistics."""
    stats = harmonix.get_stats()
    print("\n📊 Cloud Statistics:")
    print(f"  Total prose: {stats['total_prose']}")
    print(f"  Avg quality: {stats['avg_quality']:.3f}")
    print(f"  Avg dissonance: {stats['avg_dissonance']:.3f}")
    print(f"  Avg semantic density: {stats['avg_semantic_density']:.3f}")
    print(f"  Trigram vocabulary: {stats['trigram_vocabulary']}")
    print()


def show_recent(harmonix, limit=5):
    """Show recent prose from cloud."""
    recent = harmonix.get_recent_prose(limit=limit)
    print(f"\n📝 Recent Prose (last {limit}):")
    for i, (prose_id, text, quality) in enumerate(recent, 1):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[{i}] ID:{prose_id} Quality:{quality:.2f}")
        print(f"    {preview}")
    print()


def show_best(harmonix, limit=5):
    """Show highest quality prose."""
    best = harmonix.get_best_prose(limit=limit)
    print(f"\n⭐ Best Prose (top {limit}):")
    for i, (prose_id, text, quality) in enumerate(best, 1):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[{i}] ID:{prose_id} Quality:{quality:.2f}")
        print(f"    {preview}")
    print()


def show_field_seed(harmonix):
    """Show current field seed."""
    seed = harmonix.get_field_seed()
    print(f"\n🌱 Current Field Seed:")
    print(f"    {seed}")
    print()


def main():
    """Main REPL loop."""
    print_header()

    # Initialize Prose organism
    print("Initializing Prose organism...")
    harmonix = ProseHarmonix()
    generator = ProseGenerator(harmonix=harmonix, verbose=False)
    print("✓ Prose ready\n")

    # Show initial stats
    stats = harmonix.get_stats()
    print(f"Cloud status: {stats['total_prose']} prose, {stats['trigram_vocabulary']} trigrams")
    print()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input == "/quit":
                print("\n🌊 Resonance fades... goodbye!")
                break

            elif user_input == "/stats":
                show_stats(harmonix)
                continue

            elif user_input.startswith("/recent"):
                try:
                    parts = user_input.split()
                    limit = int(parts[1]) if len(parts) > 1 else 5
                    show_recent(harmonix, limit=limit)
                except (ValueError, IndexError):
                    show_recent(harmonix)
                continue

            elif user_input.startswith("/best"):
                try:
                    parts = user_input.split()
                    limit = int(parts[1]) if len(parts) > 1 else 5
                    show_best(harmonix, limit=limit)
                except (ValueError, IndexError):
                    show_best(harmonix)
                continue

            elif user_input == "/seed":
                show_field_seed(harmonix)
                continue

            elif user_input == "/temp":
                # Compute dissonance from dummy input
                dissonance, _ = harmonix.compute_dissonance(user_input, "")
                temp = harmonix.adjust_temperature(dissonance)
                print(f"\n🌡️  Current Temperature:")
                print(f"    Dissonance: {dissonance:.3f}")
                print(f"    Temperature: {temp:.3f}")
                print()
                continue

            elif user_input.startswith("/"):
                print(f"\n❌ Unknown command: {user_input}")
                print("Try: /stats, /recent, /best, /seed, /temp, /quit\n")
                continue

            # Generate prose response (ORGANISM MODE!)
            print("\nProse: ", end="", flush=True)

            # User input wrinkles field → generation from field state
            response = generator.generate(user_input, max_tokens=200)

            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\n🌊 Interrupted. Type /quit to exit.")
            continue

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    harmonix.close()


if __name__ == "__main__":
    main()
