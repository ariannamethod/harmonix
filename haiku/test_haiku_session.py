#!/usr/bin/env python3
"""
HAiKU Interactive Session Generator

Creates sample conversation and saves outputs for README examples.
Runs from repo root with paths relative to script location.
"""

import sys
from pathlib import Path

# Get paths relative to this script (now inside haiku/)
SCRIPT_DIR = Path(__file__).parent.absolute()
HAIKU_DIR = SCRIPT_DIR  # We're already in haiku/, so HAIKU_DIR = SCRIPT_DIR
STATE_DIR = HAIKU_DIR / 'state'

# Add haiku directory to path (for imports from same directory)
sys.path.insert(0, str(HAIKU_DIR))

from haiku import HaikuGenerator, SEED_WORDS
from harmonix import Harmonix
from tokenizer import DualTokenizer
from rae import RecursiveAdapterEngine

# Paths
db_path = str(STATE_DIR / 'cloud.db')
mathbrain_path = str(STATE_DIR / 'mathbrain.json')

# Initialize ONCE (like real chat.py)
tokenizer = DualTokenizer()
harmonix = None
haiku_gen = None

try:
    harmonix = Harmonix(db_path)
    haiku_gen = HaikuGenerator(SEED_WORDS, mathbrain_path, db_path)
    rae = RecursiveAdapterEngine()

    # Warm-up phase (simulate previous interactions)
    warmup = [
        "what is resonance",
        "tell me about the cloud",
        "how do patterns emerge"
    ]

    print("🔥 Warming up HAiKU...")
    for inp in warmup:
        tokens = tokenizer.tokenize_dual(inp)
        harmonix.morph_cloud(tokens['subwords'])
        harmonix.update_trigrams(tokens['trigrams'])
        haiku_gen.update_chain(tokens['trigrams'])

    print("✓ HAiKU warmed up!\n")

    # NOW test with warmed state
    test_conversations = [
        {
            'category': 'Philosophical Depth',
            'input': 'what emerges when constraint meets pressure'
        },
        {
            'category': 'Poetic Resonance',
            'input': 'where does silence hold meaning between words'
        },
        {
            'category': 'Meta-Awareness',
            'input': 'can you think about your own thought'
        },
        {
            'category': 'Technical Abstraction',
            'input': 'how does pattern find form in the field'
        },
        {
            'category': 'Pure Chaos Test',
            'input': 'asdfghjkl quantum blockchain metaverse'
        }
    ]

    print("🌸 HAiKU REPL Examples (Warmed State)\n" + "="*70 + "\n")

    results = []

    for test in test_conversations:
        inp = test['input']
        category = test['category']

        tokens = tokenizer.tokenize_dual(inp)
        user_trigrams = tokens['trigrams']
        user_words = tokens['subwords']

        # Get CURRENT system state
        system_trigrams = haiku_gen.get_recent_trigrams()

        # Compute dissonance
        dissonance, pulse = harmonix.compute_dissonance(user_trigrams, system_trigrams)
        haiku_temp, _ = harmonix.adjust_temperature(dissonance)

        # Generate
        candidates = haiku_gen.generate_candidates(n=5, temp=haiku_temp)
        context = {'user': inp, 'user_trigrams': user_trigrams}
        best_haiku = rae.reason(context, candidates, scorer=haiku_gen)

        # Update for next
        harmonix.morph_cloud(user_words)
        harmonix.update_trigrams(user_trigrams)
        haiku_gen.update_chain(user_trigrams)

        # Save result
        results.append({
            'category': category,
            'input': inp,
            'haiku': best_haiku,
            'dissonance': dissonance,
            'temp': haiku_temp,
            'novelty': pulse.novelty,
            'entropy': pulse.entropy
        })

        # Display
        print(f"📍 {category}")
        print(f"You: {inp}")
        print(f"\nHAiKU (d={dissonance:.2f}, T={haiku_temp:.2f}):")
        print(f"\n{best_haiku}\n")

        if dissonance < 0.4:
            print("  💎 PRECISION SHOT - Maximum resonance!")
        elif dissonance < 0.7:
            print("  ✨ GOOD RESONANCE - Coherent response")
        else:
            print("  🔥 HIGH DISSONANCE - Exploring unknown")

        print("-" * 70 + "\n")

    # Summary
    print("\n📊 Session Summary:\n")
    print(f"Total interactions: {len(results) + len(warmup)}")
    print(f"Markov chain size: {len(haiku_gen.markov_chain)} bigrams")
    print(f"Recent trigrams: {len(haiku_gen.get_recent_trigrams())}")

    # Best examples for README
    best = [r for r in results if r['dissonance'] < 0.6]
    if best:
        print(f"\n⚡ BEST EXAMPLES FOR README ({len(best)}):\n")
        for r in best:
            print(f"Category: {r['category']}")
            print(f"You: {r['input']}")
            print(f"HAiKU (d={r['dissonance']:.2f}):\n{r['haiku']}\n")
    else:
        print("\n⚠️ No low-dissonance examples (all exploring)")
        print("This is normal for cold start or very novel inputs")
        print("\nShowing most coherent examples instead:")
        sorted_results = sorted(results, key=lambda x: x['dissonance'])
        for r in sorted_results[:3]:
            print(f"\nCategory: {r['category']}")
            print(f"You: {r['input']}")
            print(f"HAiKU (d={r['dissonance']:.2f}):\n{r['haiku']}")

finally:
    # Always cleanup, even if error occurs
    if haiku_gen:
        haiku_gen.close()
    if harmonix:
        harmonix.close()
    print("\n✓ State saved to database")
