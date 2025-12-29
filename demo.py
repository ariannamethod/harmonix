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


def init_database():
    """Initialize cloud database with seed words."""
    # First, ensure harmonix initializes tables
    harmonix_init = Harmonix('cloud.db')
    harmonix_init.close()
    
    conn = sqlite3.connect('cloud.db')
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
    
    # Initialize components
    db = sqlite3.connect('cloud.db')
    tokenizer = DualTokenizer()
    haiku_gen = HaikuGenerator(SEED_WORDS)
    harmonix = Harmonix('cloud.db')
    rae = RecursiveAdapterEngine()
    meta = MetaHaiku(haiku_gen)
    over = Overthinkg('cloud.db')
    
    # Interaction counter
    turn = 0
    
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
