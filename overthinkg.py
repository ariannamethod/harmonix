"""
Overthinkg: Cloud Expansion Engine
Continuous background processing that grows the word cloud.

Forked from Leo's overthinking.py - uses "rings of thought" strategy.
"""

import sqlite3
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OverthinkingRing:
    """
    One internal "ring of thought" produced after an interaction.
    Inspired by Leo's overthinking rings.
    """
    ring: int  # 0 = echo, 1 = drift, 2 = meta
    trigrams: List[Tuple[str, str, str]]
    coherence: float
    source: str  # 'echo', 'drift', 'meta'

class Overthinkg:
    """
    Expansion engine using Leo's "rings of thought" approach:
    - Ring 0: Echo circle (compact rephrasing)
    - Ring 1: Semantic drift (move sideways through themes)
    - Ring 2: Meta shard (abstract keyword cluster)
    
    Each ring generates new trigrams and potentially adds new words.
    """
    
    def __init__(self, db_path: str = 'cloud.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.coherence_threshold = 0.4
        self.rings_config = {
            0: {'temp': 0.8, 'semantic': 0.2, 'max_trigrams': 5},   # Echo
            1: {'temp': 1.0, 'semantic': 0.5, 'max_trigrams': 7},   # Drift  
            2: {'temp': 1.2, 'semantic': 0.4, 'max_trigrams': 3},   # Meta
        }
    
    def expand(self, recent_trigrams: Optional[List[Tuple[str, str, str]]] = None):
        """
        Background expansion using 3 rings of thought.
        Generates new trigrams and adds coherent words to cloud.
        """
        words = self._get_words()
        existing_trigrams = self._get_trigrams()
        
        if len(words) < 3:
            return
        
        if recent_trigrams is None:
            recent_trigrams = existing_trigrams[-10:] if existing_trigrams else []
        
        # Ring 0: Echo - compact internal rephrasing
        ring0 = self._generate_ring_trigrams(
            words, recent_trigrams, ring_num=0
        )
        self._process_ring(ring0, existing_trigrams)
        
        # Ring 1: Drift - semantic exploration
        ring1 = self._generate_ring_trigrams(
            words, recent_trigrams, ring_num=1
        )
        self._process_ring(ring1, existing_trigrams)
        
        # Ring 2: Meta - abstract keywords
        ring2 = self._generate_ring_trigrams(
            words, recent_trigrams, ring_num=2
        )
        self._process_ring(ring2, existing_trigrams)
    
    def _generate_ring_trigrams(self, words: List[str], 
                               recent_trigrams: List[Tuple[str, str, str]],
                               ring_num: int) -> OverthinkingRing:
        """Generate trigrams for a specific ring."""
        config = self.rings_config[ring_num]
        max_trigrams = config['max_trigrams']
        
        new_trigrams = []
        
        # Generate based on recent context
        for _ in range(max_trigrams):
            if recent_trigrams and random.random() < 0.6:
                # Drift from recent trigrams
                base = random.choice(recent_trigrams)
                # Replace one word randomly
                word_list = list(base)
                idx = random.randint(0, 2)
                word_list[idx] = random.choice(words)
                new_trigrams.append(tuple(word_list))
            else:
                # Pure random exploration
                if len(words) >= 3:
                    trigram = tuple(random.sample(words, 3))
                    new_trigrams.append(trigram)
        
        # Compute coherence for this ring
        coherence = self._compute_ring_coherence(new_trigrams, recent_trigrams)
        
        source_names = {0: 'echo', 1: 'drift', 2: 'meta'}
        
        return OverthinkingRing(
            ring=ring_num,
            trigrams=new_trigrams,
            coherence=coherence,
            source=source_names[ring_num]
        )
    
    def _compute_ring_coherence(self, new_trigrams: List[Tuple[str, str, str]],
                                context_trigrams: List[Tuple[str, str, str]]) -> float:
        """Compute how coherent the ring is with existing context."""
        if not context_trigrams:
            return 0.5
        
        overlap_count = 0
        for new_trig in new_trigrams:
            for context_trig in context_trigrams:
                overlap = len(set(new_trig) & set(context_trig))
                overlap_count += overlap
        
        max_possible = len(new_trigrams) * len(context_trigrams) * 3
        coherence = overlap_count / max(1, max_possible)
        
        return min(1.0, coherence * 10)  # Scale up for sensitivity
    
    def _process_ring(self, ring: OverthinkingRing, 
                     existing_trigrams: List[Tuple[str, str, str]]):
        """Process ring results: add trigrams and new words if coherent."""
        for trigram in ring.trigrams:
            # Check coherence threshold
            trigram_coherence = self.compute_coherence(trigram, existing_trigrams)
            
            if trigram_coherence > self.coherence_threshold:
                self._add_trigram(trigram, trigram_coherence)
                
                # Add new words from this trigram
                for word in trigram:
                    self._add_word(word, f'overthinking:{ring.source}', trigram_coherence)
    
    def _get_words(self) -> List[str]:
        """Get all words from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT word FROM words')
        return [row[0] for row in cursor.fetchall()]
    
    def _get_trigrams(self) -> List[Tuple[str, str, str]]:
        """Get all trigrams from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT word1, word2, word3 FROM trigrams')
        return [tuple(row) for row in cursor.fetchall()]
    
    def compute_coherence(self, word_or_trigram, context_trigrams: List[Tuple[str, str, str]]) -> float:
        """
        Compute coherence score for a word or trigram.
        Measures resonance with existing cloud.
        """
        if isinstance(word_or_trigram, str):
            # Single word
            word = word_or_trigram
            appearances = sum(1 for t in context_trigrams if word in t)
            coherence = appearances / max(1, len(context_trigrams))
        else:
            # Trigram
            w1, w2, w3 = word_or_trigram
            overlap_count = 0
            for existing in context_trigrams:
                overlap = len(set(word_or_trigram) & set(existing))
                overlap_count += overlap
            coherence = overlap_count / max(1, len(context_trigrams) * 3)
        
        return min(1.0, coherence)
    
    def _add_trigram(self, trigram: Tuple[str, str, str], resonance: float):
        """Add new trigram to database."""
        cursor = self.conn.cursor()
        w1, w2, w3 = trigram
        
        cursor.execute('''
            INSERT OR IGNORE INTO trigrams (word1, word2, word3, count, resonance)
            VALUES (?, ?, ?, 1, ?)
        ''', (w1, w2, w3, resonance))
        
        self.conn.commit()
    
    def _add_word(self, word: str, source: str, weight: float):
        """Add new word to database."""
        import time
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO words (word, weight, frequency, last_used, added_by)
            VALUES (?, ?, 0, ?, ?)
        ''', (word, weight, time.time(), source))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
