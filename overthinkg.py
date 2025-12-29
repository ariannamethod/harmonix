"""
Overthinkg: Cloud Expansion Engine
Continuous background processing that grows the word cloud.
"""

import sqlite3
import random
from typing import List, Tuple

class Overthinkg:
    """
    Expansion engine that:
    - Generates new trigrams from existing cloud
    - Adds new words if coherence threshold met
    - Updates co-occurrence patterns
    """
    
    def __init__(self, db_path: str = 'cloud.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.coherence_threshold = 0.4  # Minimum coherence to add new words
    
    def expand(self):
        """
        Background expansion process.
        Generates new trigrams and potentially adds new words.
        """
        # Get existing words and trigrams
        words = self._get_words()
        trigrams = self._get_trigrams()
        
        if len(words) < 3:
            return  # Need at least 3 words to work with
        
        # Generate new potential trigrams from existing words
        new_trigrams = self._generate_new_trigrams(words, n=10)
        
        # Evaluate coherence of new trigrams
        for trigram in new_trigrams:
            coherence = self.compute_coherence(trigram, trigrams)
            
            if coherence > self.coherence_threshold:
                # Add trigram to database
                self._add_trigram(trigram, coherence)
                
                # Check if any words in trigram are new
                for word in trigram:
                    if word not in words:
                        self._add_word(word, 'overthinkg', coherence)
    
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
    
    def _generate_new_trigrams(self, words: List[str], n: int = 10) -> List[Tuple[str, str, str]]:
        """
        Generate new trigram combinations from existing words.
        Returns n candidate trigrams.
        """
        new_trigrams = []
        
        for _ in range(n):
            # Sample 3 random words
            if len(words) >= 3:
                trigram = tuple(random.sample(words, 3))
                new_trigrams.append(trigram)
        
        return new_trigrams
    
    def compute_coherence(self, word_or_trigram, context_trigrams: List[Tuple[str, str, str]]) -> float:
        """
        Compute coherence score for a word or trigram.
        Measures resonance with existing cloud.
        
        Returns float between 0 and 1.
        """
        if isinstance(word_or_trigram, str):
            # Single word
            word = word_or_trigram
            # Count how many times word appears in existing trigrams
            appearances = sum(1 for t in context_trigrams if word in t)
            coherence = appearances / max(1, len(context_trigrams))
        else:
            # Trigram
            w1, w2, w3 = word_or_trigram
            
            # Check how many words overlap with existing trigrams
            overlap_count = 0
            for existing in context_trigrams:
                overlap = len(set(word_or_trigram) & set(existing))
                overlap_count += overlap
            
            # Normalize
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
