"""
SonnetHarmonix: Observer for Shakespeare sonnets

Adapted from HAiKU harmonix.py for 14-line sonnet structure.
Tracks sonnets, dissonance, and morphs sonnet cloud.
"""

import numpy as np
import sqlite3
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PulseSnapshot:
    """
    Pulse snapshot for sonnet interactions.
    Captures resonance state between user input and generated sonnets.
    """
    novelty: float = 0.0
    arousal: float = 0.0
    entropy: float = 0.0

    @classmethod
    def from_interaction(cls, user_words: List[str], sonnet_words: List[str],
                        word_overlap: float) -> "PulseSnapshot":
        """Compute pulse from user-sonnet interaction."""
        # Novelty: how many new words in sonnet
        novelty = 1.0 - word_overlap if word_overlap < 1.0 else 0.0

        # Arousal: intensity of word count mismatch
        arousal = abs(len(user_words) - len(sonnet_words)) / max(1, len(user_words))

        # Entropy: diversity in combined vocabulary
        all_words = set(user_words + sonnet_words)
        entropy = min(1.0, len(all_words) / 50.0)  # Normalize to [0,1]

        return cls(novelty, arousal, entropy)


class SonnetHarmonix:
    """
    Observer module for Shakespeare sonnets:
    - Computes dissonance between user input and generated sonnets
    - Adjusts generation temperatures
    - Maintains sonnet cloud (database of sonnets)
    - Tracks sonnet quality metrics
    """

    def __init__(self, db_path: str = 'state/sonnets.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema for sonnets."""
        cursor = self.conn.cursor()

        # Sonnets table (stores complete 14-line sonnets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sonnets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                quality REAL DEFAULT 0.5,
                dissonance REAL,
                temperature REAL,
                timestamp REAL DEFAULT (julianday('now')),
                added_by TEXT DEFAULT 'user',
                word_count INTEGER,
                line_count INTEGER DEFAULT 14
            )
        ''')

        # Sonnet lines (individual lines from sonnets for analysis)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sonnet_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sonnet_id INTEGER,
                line_num INTEGER,
                text TEXT NOT NULL,
                syllable_count INTEGER,
                FOREIGN KEY (sonnet_id) REFERENCES sonnets(id)
            )
        ''')

        # Sonnet trigrams (for pattern tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sonnet_trigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                word3 TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                resonance REAL,
                UNIQUE(word1, word2, word3)
            )
        ''')

        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL DEFAULT (julianday('now')),
                avg_quality REAL,
                avg_dissonance REAL,
                sonnet_count INTEGER,
                vocab_size INTEGER
            )
        ''')

        self.conn.commit()

    def add_sonnet(self, sonnet_text: str, quality: float = 0.5,
                   dissonance: float = 0.5, temperature: float = 0.8,
                   added_by: str = 'user'):
        """
        Add a sonnet to the database.

        Args:
            sonnet_text: Complete 14-line sonnet
            quality: Quality score (0-1)
            dissonance: Dissonance value
            temperature: Generation temperature
            added_by: Source ('user', 'overthinkng', 'dream', etc.)
        """
        lines = sonnet_text.strip().split('\n')
        word_count = len(sonnet_text.split())
        line_count = len(lines)

        cursor = self.conn.cursor()

        # Insert sonnet
        cursor.execute('''
            INSERT INTO sonnets (text, quality, dissonance, temperature, added_by, word_count, line_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (sonnet_text, quality, dissonance, temperature, added_by, word_count, line_count))

        sonnet_id = cursor.lastrowid

        # Insert individual lines
        for i, line in enumerate(lines, start=1):
            syllable_count = self._count_syllables(line)
            cursor.execute('''
                INSERT INTO sonnet_lines (sonnet_id, line_num, text, syllable_count)
                VALUES (?, ?, ?, ?)
            ''', (sonnet_id, i, line, syllable_count))

        self.conn.commit()
        return sonnet_id

    def _count_syllables(self, line: str) -> int:
        """Estimate syllable count using formatter (more accurate)."""
        # FIX: Use formatter's improved syllable counter
        from formatter import SonnetFormatter
        formatter = SonnetFormatter()
        return formatter.count_syllables(line)

    def compute_dissonance(self, user_input: str, sonnet_text: str) -> Tuple[float, PulseSnapshot]:
        """
        Compute dissonance between user input and generated sonnet.

        Args:
            user_input: User's input text
            sonnet_text: Generated sonnet

        Returns:
            (dissonance, pulse_snapshot)
        """
        # Extract words
        user_words = user_input.lower().split()
        sonnet_words = sonnet_text.lower().split()

        if not user_words or not sonnet_words:
            return 0.5, PulseSnapshot()  # Neutral

        # Compute Jaccard similarity
        user_set = set(user_words)
        sonnet_set = set(sonnet_words)

        intersection = len(user_set & sonnet_set)
        union = len(user_set | sonnet_set)

        if union == 0:
            return 0.5, PulseSnapshot()

        similarity = intersection / union

        # Create pulse snapshot
        pulse = PulseSnapshot.from_interaction(user_words, sonnet_words, similarity)

        # Base dissonance (inverse of similarity)
        dissonance = 1.0 - similarity

        # Pulse-aware adjustment
        # High novelty → increase dissonance slightly
        # High arousal → increase dissonance
        pulse_adjustment = (pulse.novelty * 0.1) + (pulse.arousal * 0.1)
        dissonance = min(1.0, dissonance + pulse_adjustment)

        return dissonance, pulse

    def adjust_temperature(self, dissonance: float) -> Tuple[float, float]:
        """
        Adjust generation temperature based on dissonance.

        Low dissonance (resonance) → lower temperature (precision)
        High dissonance → higher temperature (exploration)

        Returns:
            (sonnet_temp, harmonix_temp)
        """
        # Sonnet generation temperature
        if dissonance < 0.3:
            sonnet_temp = 0.6  # Precision mode
        elif dissonance < 0.7:
            sonnet_temp = 0.8  # Balanced
        else:
            sonnet_temp = 1.0  # Exploration mode

        # Harmonix temperature (for cloud morphing)
        harmonix_temp = 0.5 + (dissonance * 0.5)  # Range: [0.5, 1.0]

        return sonnet_temp, harmonix_temp

    def update_trigrams(self, trigrams: List[Tuple[str, str, str]]):
        """
        Update sonnet trigram counts (for pattern tracking).

        Args:
            trigrams: List of (word1, word2, word3) tuples
        """
        cursor = self.conn.cursor()

        for w1, w2, w3 in trigrams:
            cursor.execute('''
                INSERT INTO sonnet_trigrams (word1, word2, word3, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(word1, word2, word3)
                DO UPDATE SET count = count + 1
            ''', (w1, w2, w3))

        self.conn.commit()

    def get_recent_sonnets(self, limit: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get recent sonnets from database.

        Returns:
            List of (id, text, quality) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM sonnets
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        return cursor.fetchall()

    def get_best_sonnets(self, limit: int = 10, min_quality: float = 0.7) -> List[Tuple[int, str, float]]:
        """
        Get best quality sonnets.

        Returns:
            List of (id, text, quality) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM sonnets
            WHERE quality >= ?
            ORDER BY quality DESC
            LIMIT ?
        ''', (min_quality, limit))

        return cursor.fetchall()

    def get_stats(self) -> Dict:
        """Get sonnet cloud statistics."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM sonnets')
        sonnet_count = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(quality) FROM sonnets')
        avg_quality = cursor.fetchone()[0] or 0.0

        cursor.execute('SELECT AVG(dissonance) FROM sonnets')
        avg_dissonance = cursor.fetchone()[0] or 0.0

        cursor.execute('SELECT COUNT(DISTINCT word1) FROM sonnet_trigrams')
        vocab_size = cursor.fetchone()[0]

        return {
            'sonnet_count': sonnet_count,
            'avg_quality': avg_quality,
            'avg_dissonance': avg_dissonance,
            'vocab_size': vocab_size
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == '__main__':
    # Test SonnetHarmonix
    print("Testing SonnetHarmonix...\n")

    harmonix = SonnetHarmonix()

    # Test sonnet
    test_sonnet = """When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.
The time is come to speak of love and woe.
Proud mark your father's words, and let us go.
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
And by opposing end them. To die, to sleep.
No more; and by a sleep to say we end.
The heart-ache and the thousand natural shocks.
That flesh is heir to: 'tis a consummation.
Devoutly to be wished. To die, to sleep.
To sleep, perchance to dream: ay, there's the rub."""

    user_input = "What is love and death?"

    # Compute dissonance
    dissonance, pulse = harmonix.compute_dissonance(user_input, test_sonnet)
    print(f"Dissonance: {dissonance:.3f}")
    print(f"Pulse: novelty={pulse.novelty:.3f}, arousal={pulse.arousal:.3f}, entropy={pulse.entropy:.3f}")

    # Adjust temperature
    sonnet_temp, harmonix_temp = harmonix.adjust_temperature(dissonance)
    print(f"Temperatures: sonnet={sonnet_temp:.2f}, harmonix={harmonix_temp:.2f}\n")

    # Add sonnet
    sonnet_id = harmonix.add_sonnet(test_sonnet, quality=0.8, dissonance=dissonance,
                                    temperature=sonnet_temp, added_by='test')
    print(f"✓ Added sonnet #{sonnet_id}")

    # Get stats
    stats = harmonix.get_stats()
    print(f"\nStats: {stats}")

    harmonix.close()
    print("\n✓ SonnetHarmonix test complete!")
