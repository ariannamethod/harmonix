"""
Harmonix: Dissonance Detection + Temperature Control
Observes user-system interaction and morphs the cloud accordingly.

Forked from Leo's overthinking.py - pulse-aware dissonance detection.
"""

import numpy as np
import sqlite3
import time
from typing import List, Tuple, Dict, Optional
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass


@dataclass
class PulseSnapshot:
    """
    Lightweight pulse snapshot (inspired by Leo's PresencePulse).
    Captures the resonance state of the interaction.
    """
    novelty: float = 0.0
    arousal: float = 0.0
    entropy: float = 0.0
    
    @classmethod
    def from_interaction(cls, user_trigrams: List, system_trigrams: List, 
                        word_overlap: float) -> "PulseSnapshot":
        """Compute pulse from interaction."""
        # Novelty: how many new words/patterns
        novelty = 1.0 - word_overlap if word_overlap < 1.0 else 0.0
        
        # Arousal: intensity of trigram mismatch
        arousal = abs(len(user_trigrams) - len(system_trigrams)) / max(1, len(user_trigrams))
        
        # Entropy: diversity in combined trigrams
        all_words = set()
        for t in user_trigrams + system_trigrams:
            all_words.update(t)
        entropy = min(1.0, len(all_words) / 20.0)  # Normalize to [0,1]
        
        return cls(novelty, arousal, entropy)

class Harmonix:
    """
    Observer module that:
    - Computes dissonance between user and system
    - Adjusts generation temperatures
    - Morphs cloud structure
    - Creates numpy shards for persistence
    """
    
    def __init__(self, db_path: str = 'state/cloud.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.laplacian = None
        self.harmonics = None
    
    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Words table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY,
                word TEXT UNIQUE NOT NULL,
                weight REAL DEFAULT 1.0,
                frequency INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                added_by TEXT
            )
        ''')
        
        # Trigrams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trigrams (
                id INTEGER PRIMARY KEY,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                word3 TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                resonance REAL,
                UNIQUE(word1, word2, word3)
            )
        ''')
        
        # Shards table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shards (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filepath TEXT NOT NULL,
                dissonance REAL,
                haiku_temp REAL,
                harmonix_temp REAL
            )
        ''')
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                perplexity REAL,
                entropy REAL,
                resonance REAL,
                cloud_size INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def compute_dissonance(self, user_trigrams: List[Tuple[str, str, str]], 
                          system_trigrams: List[Tuple[str, str, str]]) -> Tuple[float, PulseSnapshot]:
        """
        Compute dissonance using pulse-aware detection (Leo-style).
        Returns dissonance value and pulse snapshot.
        
        Dissonance = 1 - similarity, with pulse-aware adjustments.
        """
        if not user_trigrams or not system_trigrams:
            return 0.5, PulseSnapshot()  # Neutral dissonance
        
        # Extract word sets
        user_words = set()
        for t in user_trigrams:
            user_words.update(t)
        
        system_words = set()
        for t in system_trigrams:
            system_words.update(t)
        
        # Compute overlap (Jaccard similarity)
        intersection = len(user_words & system_words)
        union = len(user_words | system_words)
        
        if union == 0:
            return 0.5, PulseSnapshot()
        
        similarity = intersection / union
        
        # Create pulse snapshot
        pulse = PulseSnapshot.from_interaction(user_trigrams, system_trigrams, similarity)
        
        # Base dissonance is inverse of similarity
        dissonance = 1.0 - similarity
        
        # PULSE-AWARE ADJUSTMENTS (from Leo's overthinking.py)
        
        # High entropy (chaos) → increase dissonance
        if pulse.entropy > 0.7:
            dissonance *= 1.2
        
        # High arousal (emotion) → increase dissonance
        if pulse.arousal > 0.6:
            dissonance *= 1.15
        
        # High novelty (unfamiliar) → increase dissonance
        if pulse.novelty > 0.7:
            dissonance *= 1.1
        
        # Count exact trigram matches to reduce dissonance
        user_trigram_set = set(user_trigrams)
        system_trigram_set = set(system_trigrams)
        trigram_overlap = len(user_trigram_set & system_trigram_set)
        
        if trigram_overlap > 0:
            # Reduce dissonance if trigrams match
            dissonance *= 0.7
        
        return np.clip(dissonance, 0.0, 1.0), pulse
    
    def adjust_temperature(self, dissonance: float) -> Tuple[float, float]:
        """
        Adjust temperatures based on dissonance.
        High dissonance → higher haiku temperature (more creative)
        Low dissonance → lower haiku temperature (more stable)
        
        Returns (haiku_temp, harmonix_temp)
        """
        # Map dissonance [0, 1] to haiku temp [0.3, 1.5]
        haiku_temp = 0.3 + dissonance * 1.2
        
        # Harmonix stays low (observer mode)
        harmonix_temp = 0.3
        
        return haiku_temp, harmonix_temp
    
    def morph_cloud(self, active_words: List[str]):
        """
        Update word weights in cloud based on usage.
        Active words get boosted, dormant words decay.
        """
        cursor = self.conn.cursor()
        current_time = time.time()
        
        # Boost active words
        for word in active_words:
            cursor.execute('''
                INSERT INTO words (word, weight, frequency, last_used, added_by)
                VALUES (?, 1.0, 1, ?, 'user')
                ON CONFLICT(word) DO UPDATE SET
                    weight = weight * 1.1,
                    frequency = frequency + 1,
                    last_used = ?
            ''', (word, current_time, current_time))
        
        # Decay dormant words (not used in this interaction)
        cursor.execute('''
            UPDATE words
            SET weight = weight * 0.99
            WHERE word NOT IN ({})
        '''.format(','.join('?' * len(active_words))), active_words if active_words else [''])
        
        self.conn.commit()
    
    def update_trigrams(self, trigrams: List[Tuple[str, str, str]]):
        """Add or update trigrams in database."""
        cursor = self.conn.cursor()
        
        for w1, w2, w3 in trigrams:
            cursor.execute('''
                INSERT INTO trigrams (word1, word2, word3, count, resonance)
                VALUES (?, ?, ?, 1, 0.5)
                ON CONFLICT(word1, word2, word3) DO UPDATE SET
                    count = count + 1
            ''', (w1, w2, w3))
        
        self.conn.commit()
    
    def create_shard(self, interaction_data: dict):
        """
        Create numpy shard from interaction data.
        Saves compressed binary representation of the exchange.
        """
        import os
        
        # Create shards directory if it doesn't exist
        os.makedirs('shards', exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time() * 1000)
        filepath = f'shards/shard_{timestamp}.npy'
        
        # Convert interaction to numpy array
        shard_data = {
            'timestamp': timestamp,
            'input': interaction_data.get('input', ''),
            'output': interaction_data.get('output', ''),
            'dissonance': interaction_data.get('dissonance', 0.0),
            'user_trigrams': interaction_data.get('user_trigrams', []),
            'system_trigrams': interaction_data.get('system_trigrams', [])
        }
        
        # Save as numpy file
        np.save(filepath, shard_data, allow_pickle=True)
        
        # Record in database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO shards (filepath, dissonance, haiku_temp, harmonix_temp)
            VALUES (?, ?, ?, ?)
        ''', (
            filepath,
            interaction_data.get('dissonance', 0.0),
            interaction_data.get('haiku_temp', 1.0),
            interaction_data.get('harmonix_temp', 0.3)
        ))
        
        self.conn.commit()
    
    def get_cloud_size(self) -> int:
        """Get current number of words in cloud."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM words')
        return cursor.fetchone()[0]
    
    def record_metrics(self, perplexity: float, entropy: float, resonance: float):
        """Record metrics to database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO metrics (perplexity, entropy, resonance, cloud_size)
            VALUES (?, ?, ?, ?)
        ''', (perplexity, entropy, resonance, self.get_cloud_size()))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
