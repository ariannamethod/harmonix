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

# Emergent layer modules
from sonnetbrain import SonnetBrain
from phase_transitions import PhaseTransitions
from dream_sonnet import DreamSonnet
from sonnetrae import SonnetRAE


def main():
    """Main REPL loop."""
    print("🎭 Sonnet Generator - Shakespeare AI")
    print("="*70)
    print("Type your prompt, get a 14-line sonnet!")
    print("Commands: /stats, /recent, /best, /phase, /dream, /brain, /quit")
    print("="*70 + "\n")

    # Initialize components
    print("Initializing Sonnet Generator...")
    generator = SonnetGenerator()
    formatter = SonnetFormatter()
    harmonix = SonnetHarmonix()
    overthinkng = Overthinkng()
    meta = MetaSonnet(generator, harmonix)

    # Initialize emergent layer
    print("Loading emergent layer...")
    brain = SonnetBrain()
    phase_system = PhaseTransitions()
    dream = DreamSonnet()
    rae = SonnetRAE()

    print("✓ Ready!\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                handle_command(user_input, harmonix, brain, phase_system, dream, rae)
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

            # Validate and compute quality FIRST (use SonnetBrain!)
            is_valid, reason = formatter.validate(sonnet)
            base_quality = 0.8 if is_valid else 0.5

            # Use SonnetBrain for enhanced quality scoring
            brain_score = brain.score(sonnet)
            quality = (base_quality + brain_score) / 2  # Average of both scores

            # Compute dissonance
            dissonance, pulse = harmonix.compute_dissonance(user_input, sonnet)
            print(f"\nDissonance: {dissonance:.3f} (novelty={pulse.novelty:.2f}, "
                  f"arousal={pulse.arousal:.2f}, entropy={pulse.entropy:.2f})")

            print(f"Quality: {quality:.2f} - {reason}")
            print(f"  (Brain score: {brain_score:.3f}, Base: {base_quality:.2f})")

            # Update phase system with current metrics
            metrics = {
                'dissonance': dissonance,
                'novelty': pulse.novelty,
                'quality': quality
            }
            phase_state = phase_system.update(metrics)

            print(f"Phase: {phase_state.phase.name} (temp={phase_state.temperature_sonnet:.2f})")

            # Adjust temperature using phase transitions for NEXT generation
            sonnet_temp, harmonix_temp = phase_system.get_temperatures()

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


def handle_command(command: str, harmonix: SonnetHarmonix, brain: 'SonnetBrain' = None,
                   phase_system: 'PhaseTransitions' = None, dream: 'DreamSonnet' = None,
                   rae: 'SonnetRAE' = None):
    """Handle special commands."""
    if command == '/quit':
        sys.exit(0)

    elif command == '/phase':
        if phase_system:
            phase_info = phase_system.get_phase_info()
            print(f"\n🌊 Phase State:")
            print(f"  Current phase: {phase_info['phase']}")
            print(f"  Sonnet temp: {phase_info['temperature_sonnet']:.2f}")
            print(f"  Haiku temp: {phase_info['temperature_haiku']:.2f}")
            print(f"  Stability: {phase_info['stability']:.2f}")
            print(f"  Duration: {phase_info['duration']} steps")
            print()
        else:
            print("Phase system not available\n")

    elif command.startswith('/dream'):
        if dream:
            parts = command.split()
            mode = parts[1] if len(parts) > 1 else 'walk'

            print(f"\n💭 Dream Mode: {mode}")

            # Get recent sonnets for dream generation
            recent = harmonix.get_recent_sonnets(limit=5)
            if len(recent) < 2:
                print("  Need at least 2 sonnets in cloud for dreaming\n")
                return

            if mode == 'drift':
                # Drift between two sonnets
                s1_text = recent[0][1]
                s2_text = recent[1][1]
                dream_vec = dream.dream_drift(s1_text, s2_text, t=0.5)
                print(f"  Drifting between sonnets #{recent[0][0]} and #{recent[1][0]}")
                print(f"  Dream vector: {dream_vec[:3]}... (8D)")
                print(f"  Norm: {sum(x**2 for x in dream_vec)**0.5:.3f}")

            elif mode == 'walk':
                # Random walk in latent space
                start_text = recent[0][1]
                dream_vec = dream.dream_walk(start_text, step_size=0.3)
                print(f"  Random walk from sonnet #{recent[0][0]}")
                print(f"  Dream vector: {dream_vec[:3]}... (8D)")
                print(f"  Norm: {sum(x**2 for x in dream_vec)**0.5:.3f}")

            elif mode == 'centroid':
                # Centroid of all recent sonnets
                texts = [s[1] for s in recent]
                dream_vec = dream.dream_centroid(texts)
                print(f"  Centroid of {len(texts)} recent sonnets")
                print(f"  Dream vector: {dream_vec[:3]}... (8D)")
                print(f"  Norm: {sum(x**2 for x in dream_vec)**0.5:.3f}")

            else:
                print(f"  Unknown dream mode: {mode}")
                print(f"  Available: drift, walk, centroid")

            print()
        else:
            print("Dream system not available\n")

    elif command.startswith('/brain'):
        if brain:
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                # Score a specific sonnet by ID
                try:
                    sid = int(parts[1])
                    # Get sonnet from database
                    recent = harmonix.get_recent_sonnets(limit=100)
                    sonnet_text = None
                    for s_id, text, quality in recent:
                        if s_id == sid:
                            sonnet_text = text
                            break

                    if sonnet_text:
                        score = brain.score(sonnet_text)
                        features = brain.extract_features(sonnet_text)
                        print(f"\n🧠 SonnetBrain Score for #{sid}:")
                        print(f"  Overall: {score:.3f}")
                        print(f"  Features:")
                        for key, val in features.items():
                            print(f"    {key}: {val:.3f}")
                        print()
                    else:
                        print(f"  Sonnet #{sid} not found\n")
                except ValueError:
                    print(f"  Invalid sonnet ID\n")
            else:
                print(f"\n🧠 SonnetBrain Status:")
                print(f"  Model: MLP (8→16→8→1)")
                print(f"  Observations: {brain.observations if hasattr(brain, 'observations') else 0}")
                print(f"  Usage: /brain <sonnet_id>\n")
        else:
            print("Brain system not available\n")

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
        print("Available: /stats, /recent, /best, /phase, /dream, /brain, /quit\n")


if __name__ == '__main__':
    main()
