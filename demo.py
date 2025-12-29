"""
HAiKU Demo: Minimal Working System

Usage:
  python demo.py

Watch it:
1. Take your input
2. Compute dissonance with pulse awareness
3. Generate haiku candidates
4. Select best via RAE
5. Reflect internally (metahaiku)
6. Expand cloud (overthinking rings)
7. Repeat
"""

import sqlite3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haiku import HaikuGenerator, SEED_WORDS
from harmonix import Harmonix, PulseSnapshot
from tokenizer import DualTokenizer
from rae import RecursiveAdapterEngine
from metahaiku import MetaHaiku
from overthinkg import Overthinkg
from phase4_bridges import HaikuBridges, state_id_from_metrics
from dream_haiku import (HaikuDreamContext, DreamConfig, init_dream, 
                         should_run_dream, run_dream_dialog, update_dream_fragments)


def init_database():
    """Initialize cloud database with seed words."""
    # First, ensure harmonix initializes tables
    harmonix_init = Harmonix('state/cloud.db')
    harmonix_init.close()

    conn = sqlite3.connect('state/cloud.db')
    cursor = conn.cursor()
    
    # Check if words table has entries
    cursor.execute('SELECT COUNT(*) FROM words')
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Seed the database
        print("Seeding database with 500 words...")
        import time
        for word in SEED_WORDS:
            cursor.execute('''
                INSERT OR IGNORE INTO words (word, weight, frequency, last_used, added_by)
                VALUES (?, 1.0, 0, ?, 'seed')
            ''', (word, time.time()))
        conn.commit()
        print(f"Seeded {len(SEED_WORDS)} words.")
    
    conn.close()


def main():
    """Run HAiKU interactive demo."""
    print("=" * 60)
    print("HAiKU v1 - Resonance-Driven Haiku Generator")
    print("=" * 60)
    print()
    print("Core philosophy: Constraint → Emergence → Coherence")
    print("500 words → 5-7-5 haiku → cloud expansion")
    print()
    print("Type your message, I'll respond in haiku.")
    print("Type 'quit' to exit.\n")
    
    # Initialize database
    init_database()

    # Initialize dream space
    init_dream('state/cloud.db')

    # Initialize components
    db = sqlite3.connect('state/cloud.db')
    tokenizer = DualTokenizer()
    haiku_gen = HaikuGenerator(SEED_WORDS)
    harmonix = Harmonix('state/cloud.db')
    rae = RecursiveAdapterEngine()
    meta = MetaHaiku(haiku_gen)
    over = Overthinkg('state/cloud.db')
    bridges = HaikuBridges('state/cloud.db')  # Phase 4: Island Bridges
    
    # Interaction counter
    turn = 0
    prev_state_id = None
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nFarewell. The cloud remembers.")
                break
            
            if not user_input:
                continue
            
            turn += 1
            
            # Tokenize user input
            tokens = tokenizer.tokenize_dual(user_input)
            user_trigrams = tokens['trigrams']
            user_words = tokens['subwords']
            
            # Update harmonix with user input
            harmonix.morph_cloud(user_words)
            harmonix.update_trigrams(user_trigrams)
            
            # Update haiku generator's Markov chain
            haiku_gen.update_chain(user_trigrams)
            
            # Get system's recent trigrams
            system_trigrams = haiku_gen.get_recent_trigrams()
            
            # Compute dissonance with pulse awareness
            dissonance, pulse = harmonix.compute_dissonance(
                user_trigrams,
                system_trigrams
            )
            
            # Adjust temperatures based on dissonance
            haiku_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)
            
            # Generate haiku candidates
            candidates = haiku_gen.generate_candidates(n=5, temp=haiku_temp)
            
            # Select best via RAE
            context = {
                'user': user_input,
                'user_trigrams': user_trigrams,
                'pulse': pulse,
                'dissonance': dissonance
            }
            best_haiku = rae.reason(context, candidates, scorer=haiku_gen)
            
            # Display haiku response
            print(f"\nHAiKU (d={dissonance:.2f}, T={haiku_temp:.2f}):\n")
            print(best_haiku)
            print()
            
            # Compute quality score for this haiku
            # In v1: simple heuristic based on dissonance and pulse
            # In v2: could be user feedback
            quality = 0.5  # Base quality
            
            # Good quality indicators:
            # - Moderate dissonance (not too high, not too low)
            if 0.3 < dissonance < 0.7:
                quality += 0.2
            
            # - Moderate entropy (balanced)
            if 0.4 < pulse.entropy < 0.8:
                quality += 0.15
            
            # - Some novelty but not too much
            if 0.3 < pulse.novelty < 0.7:
                quality += 0.15
            
            quality = min(1.0, max(0.0, quality))
            
            # Create state ID for Phase 4
            current_state_id = state_id_from_metrics(dissonance, pulse.entropy, quality)
            
            # Metrics before and after
            metrics_before = {
                'dissonance': dissonance,
                'entropy': pulse.entropy,
                'novelty': pulse.novelty,
                'arousal': pulse.arousal,
                'quality': 0.5,  # before generation
            }
            
            metrics_after = {
                'dissonance': dissonance,
                'entropy': pulse.entropy,
                'novelty': pulse.novelty,
                'arousal': pulse.arousal,
                'quality': quality,  # after generation
            }
            
            # Phase 4: Record state transition
            bridges.record_state(
                state_id=current_state_id,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                prev_state_id=prev_state_id,
                turn_id=f"turn_{turn}",
                boredom=(dissonance < 0.3 and pulse.entropy < 0.4),
                overwhelm=(dissonance > 0.8 or pulse.arousal > 0.8),
                stuck=(quality < 0.4)
            )
            
            # MathBrain: observe and learn (Leo-style)
            loss = haiku_gen.observe(best_haiku, quality, user_context=user_trigrams)
            
            # Debug: show learning stats
            if os.environ.get('HAIKU_DEBUG') == '1':
                stats = haiku_gen.get_stats()
                print(f"[MathBrain] obs={stats['observations']}, loss={loss:.4f}, avg_loss={stats['running_loss']:.4f}")
                
                # Show Phase 4 suggestions
                suggestions = bridges.suggest_next_states(current_state_id, min_count=1, max_results=3)
                if suggestions:
                    print(f"[Phase4] Suggested next states:")
                    for s in suggestions:
                        print(f"  → {s.to_state} (score={s.score:.3f}, count={s.count})")
            
            # Update prev_state_id for next iteration
            prev_state_id = current_state_id
            
            # Background processes (not shown to user)
            
            # MetaHaiku: internal reflection
            interaction_data = {
                'user': user_input,
                'haiku': best_haiku,
                'dissonance': dissonance,
                'pulse': pulse,
                'turn': turn
            }
            internal_haiku = meta.reflect(interaction_data)
            
            # Debug: optionally show internal voice
            if os.environ.get('HAIKU_DEBUG') == '1':
                print(f"[MetaHaiku internal]: {internal_haiku}\n")
            
            # Overthinkg: cloud expansion with rings
            over.expand(recent_trigrams=user_trigrams)
            
            # Check if we should run a dream dialog
            dream_ctx = HaikuDreamContext(
                last_haiku=best_haiku,
                dissonance=dissonance,
                pulse_entropy=pulse.entropy,
                pulse_novelty=pulse.novelty,
                pulse_arousal=pulse.arousal,
                quality=quality,
                cloud_size=len(haiku_gen.vocab),
                turn_count=turn
            )
            
            if should_run_dream(dream_ctx, 'state/cloud.db'):
                # Run dream dialog in background
                dream_haikus = run_dream_dialog(haiku_gen, best_haiku, 'state/cloud.db')
                
                if os.environ.get('HAIKU_DEBUG') == '1':
                    print(f"[Dream] Generated {len(dream_haikus)} dream haikus")
                    for i, dh in enumerate(dream_haikus[:2]):
                        print(f"  Dream {i+1}: {dh.replace(chr(10), ' / ')}")
                
                # Feed best dream haikus back into cloud
                for dh in dream_haikus:
                    update_dream_fragments(dh, 'state/cloud.db')
                    # Also update haiku generator's chain
                    dream_tokens = tokenizer.tokenize_dual(dh)
                    haiku_gen.update_chain(dream_tokens['trigrams'])
            
            # Create numpy shard
            shard_data = {
                'input': user_input,
                'output': best_haiku,
                'dissonance': dissonance,
                'haiku_temp': haiku_temp,
                'harmonix_temp': harmonix_temp,
                'user_trigrams': user_trigrams,
                'system_trigrams': system_trigrams
            }
            harmonix.create_shard(shard_data)
            
            # Record metrics
            harmonix.record_metrics(
                perplexity=0.5,  # Placeholder
                entropy=pulse.entropy,
                resonance=1.0 - dissonance
            )
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. The cloud fades.")
    
    finally:
        # Show stats before cleanup
        cloud_size = len(haiku_gen.vocab)
        print(f"\n{turn} interactions completed.")
        print(f"Cloud size: {cloud_size} words")
        
        # Cleanup
        harmonix.close()
        over.close()
        db.close()


if __name__ == '__main__':
    main()
