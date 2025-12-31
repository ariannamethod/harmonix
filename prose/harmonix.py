"""
ProseHarmonix: Observer for free-form prose

Adapted from Sonnet harmonix.py for prose structure.
Tracks prose, dissonance, semantic density, and morphs prose cloud.
"""

import numpy as np
import sqlite3
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PulseSnapshot:
    """
    Pulse snapshot for prose interactions.
    Captures resonance state between user input and generated prose.
    """
    novelty: float = 0.0
    arousal: float = 0.0
    entropy: float = 0.0

    @classmethod
    def from_interaction(cls, user_words: List[str], prose_words: List[str],
                        word_overlap: float) -> "PulseSnapshot":
        """Compute pulse from user-prose interaction."""
        # Novelty: how many new words in prose
        novelty = 1.0 - word_overlap if word_overlap < 1.0 else 0.0

        # Arousal: intensity of word count mismatch
        arousal = abs(len(user_words) - len(prose_words)) / max(1, len(user_words))

        # Entropy: diversity in combined vocabulary
        all_words = set(user_words + prose_words)
        entropy = min(1.0, len(all_words) / 100.0)  # Normalized (higher than Sonnet: 100 vs 50)

        return cls(novelty, arousal, entropy)


class ProseHarmonix:
    """
    Observer module for free-form prose:
    - Computes dissonance between user input and generated prose
    - Adjusts generation temperatures
    - Maintains prose cloud (database of prose)
    - Tracks prose quality and semantic density
    """

    def __init__(self, db_path: str = 'cloud/prose.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema for prose."""
        cursor = self.conn.cursor()

        # Prose table (stores complete prose texts)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prose (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                quality REAL DEFAULT 0.5,
                dissonance REAL,
                temperature REAL,
                semantic_density REAL,
                timestamp REAL DEFAULT (julianday('now')),
                added_by TEXT DEFAULT 'user',
                word_count INTEGER,
                sentence_count INTEGER,
                paragraph_count INTEGER
            )
        ''')

        # Prose sentences (individual sentences for analysis)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prose_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prose_id INTEGER,
                sentence_num INTEGER,
                text TEXT NOT NULL,
                word_count INTEGER,
                FOREIGN KEY (prose_id) REFERENCES prose(id)
            )
        ''')

        # Prose trigrams (for pattern tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prose_trigrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                word3 TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                resonance REAL,
                UNIQUE(word1, word2, word3)
            )
        ''')

        self.conn.commit()

    def add_prose(self, text: str, quality: float = 0.5,
                 dissonance: float = 0.0, temperature: float = 0.8,
                 added_by: str = 'user') -> int:
        """
        Add prose to cloud.

        Args:
            text: Prose text
            quality: Quality score (0-1)
            dissonance: Dissonance score
            temperature: Generation temperature used
            added_by: Source identifier

        Returns:
            Prose ID
        """
        from formatter import ProseFormatter
        formatter = ProseFormatter()

        # Compute metrics
        metrics = formatter.compute_metrics(text)
        semantic_density = self._compute_semantic_density(text)

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO prose (text, quality, dissonance, temperature, semantic_density,
                             word_count, sentence_count, paragraph_count, added_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text, quality, dissonance, temperature, semantic_density,
              metrics['word_count'], metrics['sentence_count'],
              metrics['paragraph_count'], added_by))

        prose_id = cursor.lastrowid

        # Store sentences
        sentences = formatter.split_sentences(text)
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            cursor.execute('''
                INSERT INTO prose_sentences (prose_id, sentence_num, text, word_count)
                VALUES (?, ?, ?, ?)
            ''', (prose_id, i, sentence, word_count))

        # Store trigrams
        self._update_trigrams(text)

        self.conn.commit()
        return prose_id

    def _compute_semantic_density(self, text: str) -> float:
        """
        Compute semantic density (meaningful words per total words).

        Returns:
            Density score (0-1)
        """
        words = text.split()
        if not words:
            return 0.0

        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
                    'was', 'are', 'were', 'be', 'been', 'being', 'it', 'its'}

        meaningful = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
        unique_meaningful = set(meaningful)

        density = len(unique_meaningful) / max(1, len(words))
        return density

    def _update_trigrams(self, text: str):
        """Update trigram counts from prose text."""
        words = text.lower().split()

        if len(words) < 3:
            return

        cursor = self.conn.cursor()

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]

            # Try to update existing trigram
            cursor.execute('''
                UPDATE prose_trigrams
                SET count = count + 1
                WHERE word1 = ? AND word2 = ? AND word3 = ?
            ''', (w1, w2, w3))

            # If no rows updated, insert new trigram
            if cursor.rowcount == 0:
                cursor.execute('''
                    INSERT OR IGNORE INTO prose_trigrams (word1, word2, word3, count, resonance)
                    VALUES (?, ?, ?, 1, 0.5)
                ''', (w1, w2, w3))

        self.conn.commit()

    def compute_dissonance(self, user_input: str, prose_output: str) -> Tuple[float, PulseSnapshot]:
        """
        Compute dissonance between user input and prose output.

        Higher dissonance = more novel/surprising response.

        Args:
            user_input: User's question/prompt
            prose_output: Generated prose

        Returns:
            (dissonance_score, pulse_snapshot)
        """
        user_words = set(user_input.lower().split())
        prose_words = set(prose_output.lower().split())

        # Word overlap
        overlap = len(user_words & prose_words)
        max_overlap = max(len(user_words), 1)
        overlap_ratio = overlap / max_overlap

        # Compute pulse
        pulse = PulseSnapshot.from_interaction(
            list(user_words), list(prose_words), overlap_ratio
        )

        # Dissonance = weighted combination of pulse metrics
        dissonance = (pulse.novelty * 0.5) + (pulse.arousal * 0.3) + (pulse.entropy * 0.2)

        return dissonance, pulse

    def adjust_temperature(self, dissonance: float, base_temp: float = 0.8) -> float:
        """
        Adjust generation temperature based on dissonance.

        High dissonance → lower temp (more conservative)
        Low dissonance → higher temp (more creative)

        Args:
            dissonance: Current dissonance score
            base_temp: Base temperature

        Returns:
            Adjusted temperature
        """
        if dissonance > 0.7:
            # Very high dissonance, be conservative
            return base_temp * 0.7
        elif dissonance < 0.3:
            # Low dissonance, be creative
            return base_temp * 1.2
        else:
            return base_temp

    def get_stats(self) -> Dict[str, any]:
        """
        Get prose cloud statistics.

        Returns:
            Dictionary with cloud stats
        """
        cursor = self.conn.cursor()

        # Total prose count
        cursor.execute('SELECT COUNT(*) FROM prose')
        total_prose = cursor.fetchone()[0]

        # Average quality
        cursor.execute('SELECT AVG(quality) FROM prose')
        avg_quality = cursor.fetchone()[0] or 0.0

        # Average dissonance
        cursor.execute('SELECT AVG(dissonance) FROM prose')
        avg_dissonance = cursor.fetchone()[0] or 0.0

        # Average semantic density
        cursor.execute('SELECT AVG(semantic_density) FROM prose')
        avg_density = cursor.fetchone()[0] or 0.0

        # Trigram vocabulary
        cursor.execute('SELECT COUNT(*) FROM prose_trigrams')
        trigram_count = cursor.fetchone()[0]

        return {
            'total_prose': total_prose,
            'avg_quality': avg_quality,
            'avg_dissonance': avg_dissonance,
            'avg_semantic_density': avg_density,
            'trigram_vocabulary': trigram_count,
        }

    def get_recent_prose(self, limit: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get recent prose from cloud.

        Args:
            limit: Number of prose to return

        Returns:
            List of (id, text, quality) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM prose
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def get_best_prose(self, limit: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get highest quality prose from cloud.

        Args:
            limit: Number of prose to return

        Returns:
            List of (id, text, quality) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM prose
            ORDER BY quality DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == '__main__':
    # Test ProseHarmonix
    print("Testing ProseHarmonix...\n")

    harmonix = ProseHarmonix()

    # Add test prose
    test_prose = """Resonance is the interplay between the sound, the words, and the message.
It's the way the words come together to create a sound and a feeling.
It's the way you hear the words, and the way they resonate with you.

When we speak of resonance in the context of language, we touch upon
something deeper than mere communication."""

    user_input = "What is resonance?"

    print("Adding test prose...")
    dissonance, pulse = harmonix.compute_dissonance(user_input, test_prose)
    prose_id = harmonix.add_prose(test_prose, quality=0.75, dissonance=dissonance)

    print(f"✓ Prose added (ID: {prose_id})")
    print(f"  Dissonance: {dissonance:.3f}")
    print(f"  Pulse: novelty={pulse.novelty:.3f}, arousal={pulse.arousal:.3f}, entropy={pulse.entropy:.3f}")

    # Get stats
    stats = harmonix.get_stats()
    print(f"\nCloud stats: {stats}")

    harmonix.close()
    print("\n✓ ProseHarmonix test complete!")
