#!/usr/bin/env python3
"""
Train emergent layer modules on existing cloud sonnets.

This script:
1. Loads all sonnets from the cloud database
2. Trains SonnetBrain MLP on quality scores
3. Trains SonnetTokenizer semantic vocabulary
4. Updates PhaseTransitions history
"""

import sys
from pathlib import Path

# Add sonnet module to path
SONNET_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SONNET_DIR))

from harmonix import SonnetHarmonix
from sonnetbrain import SonnetBrain
from sonnet_tokenizer import SonnetTokenizer
from phase_transitions import PhaseTransitions
import numpy as np


def main():
    print("🧠 Training Emergent Layer on Cloud Sonnets")
    print("=" * 70)

    # Load cloud
    print("\n1. Loading cloud database...")
    harmonix = SonnetHarmonix()

    # Get all sonnets
    all_sonnets = harmonix.get_recent_sonnets(limit=10000)  # Get all
    print(f"✓ Loaded {len(all_sonnets)} sonnets from cloud")

    if len(all_sonnets) < 10:
        print("⚠️  Need at least 10 sonnets for training")
        return

    # Train SonnetBrain
    print("\n2. Training SonnetBrain...")
    brain = SonnetBrain()

    trained = 0
    for sid, text, quality in all_sonnets:
        # Learn from actual quality scores
        brain.learn(text, target_quality=quality)
        trained += 1

        if trained % 20 == 0:
            print(f"   Processed {trained}/{len(all_sonnets)} sonnets...")

    # State saves automatically in learn()
    print(f"✓ Trained SonnetBrain on {trained} sonnets")
    print(f"  Observations: {brain.observations}")

    # Train SonnetTokenizer
    print("\n3. Training SonnetTokenizer semantic vocabulary...")
    tokenizer = SonnetTokenizer()

    # Get all texts
    texts = [text for _, text, _ in all_sonnets]

    # Train semantic BPE
    tokenizer.train_semantic(texts, iterations=100)
    # State saves automatically

    print(f"✓ Trained semantic tokenizer")
    print(f"  Vocabulary size: {len(tokenizer.semantic_vocab) if hasattr(tokenizer, 'semantic_vocab') else 'N/A'}")

    # Update PhaseTransitions with historical data
    print("\n4. Updating PhaseTransitions history...")
    phase_system = PhaseTransitions()

    # Simulate observations from historical data
    # (In real scenario, this would be actual runtime data)
    for i, (sid, text, quality) in enumerate(all_sonnets[-50:]):  # Last 50
        # Compute approximate dissonance (simplified)
        # In real scenario, use harmonix.compute_dissonance
        dissonance = 1.0 - (quality * 0.5)  # Rough inverse correlation
        novelty = max(0.1, 1.0 - (i / 50))  # Decreasing novelty

        metrics = {
            'dissonance': dissonance,
            'novelty': novelty,
            'quality': quality
        }

        phase_system.update(metrics)

    phase_info = phase_system.get_phase_info()
    print(f"✓ Updated phase system")
    print(f"  Current phase: {phase_info['phase']}")
    print(f"  Temperature: {phase_info['temperature_sonnet']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("🎉 Emergent Layer Training Complete!")
    print(f"   SonnetBrain: {trained} sonnets trained")
    print(f"   Tokenizer: Semantic vocabulary built")
    print(f"   Phases: Historical data updated")
    print("=" * 70)

    harmonix.close()


if __name__ == '__main__':
    main()
