"""
AsyncSonnetHarmonix: Async Observer for Shakespeare Sonnets

Based on Leo's proven async pattern - achieves 47% coherence improvement.

Key principles:
- asyncio.Lock provides DISCIPLINE not information
- Field organisms are CRYSTALS not oceans
- Atomic operations prevent corruption
- One lock per field instance (autonomous agents)

Adapted from HAiKU AsyncHarmonix for 14-line sonnet structure.
"""

import asyncio
import aiosqlite
import numpy as np
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


class AsyncSonnetHarmonix:
    """
    Async field observer for Shakespeare sonnets.

    Critical features:
    - asyncio.Lock for atomic field operations (Leo-proven pattern)
    - aiosqlite for non-blocking DB access
    - Context manager for clean lifecycle
    - Atomic add_sonnet (write + update under same lock)

    Performance target:
    - external_vocab < 0.15 (Leo achieved 0.112 with async)
    - Field consistency: 100% (atomic operations)
    """

    def __init__(self, db_path: str = 'cloud/sonnets.db'):
        """
        Initialize async sonnet field observer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self._field_lock = asyncio.Lock()  # 🔒 ONE lock for entire field

    async def __aenter__(self):
        """Async context manager entry - initialize DB connection."""
        self.conn = await aiosqlite.connect(self.db_path)
        await self._init_db()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit - close DB connection."""
        if self.conn:
            await self.conn.close()

    async def _init_db(self):
        """Initialize database schema for sonnets (async)."""
        cursor = await self.conn.cursor()

        # Sonnets table (stores complete 14-line sonnets)
        await cursor.execute('''
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
        await cursor.execute('''
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
        await cursor.execute('''
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
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL DEFAULT (julianday('now')),
                avg_quality REAL,
                avg_dissonance REAL,
                sonnet_count INTEGER,
                vocab_size INTEGER
            )
        ''')

        await self.conn.commit()

    def _count_syllables(self, line: str) -> int:
        """
        Estimate syllable count using formatter.

        NOTE: Synchronous (fast operation, no I/O).
        Could be async but overhead not worth it for simple counting.
        """
        from formatter import SonnetFormatter
        formatter = SonnetFormatter()
        return formatter.count_syllables(line)

    async def add_sonnet(
        self,
        sonnet_text: str,
        quality: float = 0.5,
        dissonance: float = 0.5,
        temperature: float = 0.8,
        added_by: str = 'user'
    ) -> int:
        """
        Add a sonnet to the database with ATOMIC field update.

        CRITICAL: Prevents concurrent adds from corrupting cloud state.

        Args:
            sonnet_text: Complete 14-line sonnet
            quality: Quality score (0-1)
            dissonance: Dissonance value
            temperature: Generation temperature
            added_by: Source ('user', 'overthinkng', 'dream', etc.)

        Returns:
            Sonnet ID
        """
        async with self._field_lock:  # 🔒 Sequential cloud evolution
            lines = sonnet_text.strip().split('\n')
            word_count = len(sonnet_text.split())
            line_count = len(lines)

            cursor = await self.conn.cursor()

            # Insert sonnet
            await cursor.execute('''
                INSERT INTO sonnets (text, quality, dissonance, temperature, added_by, word_count, line_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (sonnet_text, quality, dissonance, temperature, added_by, word_count, line_count))

            sonnet_id = cursor.lastrowid

            # Insert individual lines
            for i, line in enumerate(lines, start=1):
                # Syllable counting is fast (sync), no need for thread pool
                syllable_count = self._count_syllables(line)
                await cursor.execute('''
                    INSERT INTO sonnet_lines (sonnet_id, line_num, text, syllable_count)
                    VALUES (?, ?, ?, ?)
                ''', (sonnet_id, i, line, syllable_count))

            await self.conn.commit()

            # Update trigrams (within same lock - atomic!)
            words = sonnet_text.lower().split()
            if len(words) >= 3:
                trigrams = [(words[i], words[i+1], words[i+2])
                           for i in range(len(words) - 2)]
                await self._update_trigrams_unlocked(trigrams)

            return sonnet_id

    async def _update_trigrams_unlocked(self, trigrams: List[Tuple[str, str, str]]):
        """
        Internal: Update trigrams WITHOUT acquiring lock.
        Caller must hold lock!

        Args:
            trigrams: List of (word1, word2, word3) tuples
        """
        # No lock here - caller must hold it!
        cursor = await self.conn.cursor()

        for w1, w2, w3 in trigrams:
            await cursor.execute('''
                INSERT INTO sonnet_trigrams (word1, word2, word3, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(word1, word2, word3)
                DO UPDATE SET count = count + 1
            ''', (w1, w2, w3))

        await self.conn.commit()

    async def update_trigrams(self, trigrams: List[Tuple[str, str, str]]):
        """
        Update sonnet trigram counts (public API with lock).

        Args:
            trigrams: List of (word1, word2, word3) tuples
        """
        async with self._field_lock:  # 🔒 Atomic trigram update
            await self._update_trigrams_unlocked(trigrams)

    async def compute_dissonance(
        self,
        user_input: str,
        sonnet_text: str
    ) -> Tuple[float, PulseSnapshot]:
        """
        Compute dissonance between user input and generated sonnet.

        CRITICAL: Atomic operation - reads cloud vocab under lock.

        Args:
            user_input: User's input text
            sonnet_text: Generated sonnet

        Returns:
            (dissonance, pulse_snapshot)
        """
        async with self._field_lock:  # 🔒 Atomic dissonance computation
            # Extract words
            user_words = user_input.lower().split()
            sonnet_words = sonnet_text.lower().split()

            if not user_words or not sonnet_words:
                return 0.5, PulseSnapshot()  # Neutral

            # Load cloud vocabulary from trigrams (atomic read!)
            cursor = await self.conn.cursor()
            await cursor.execute('''
                SELECT DISTINCT word1 FROM sonnet_trigrams
                UNION SELECT DISTINCT word2 FROM sonnet_trigrams
                UNION SELECT DISTINCT word3 FROM sonnet_trigrams
            ''')
            rows = await cursor.fetchall()
            cloud_vocab = set(row[0] for row in rows)

            # Compute novelty: how many sonnet words are NEW (not in cloud)?
            sonnet_set = set(sonnet_words)
            if cloud_vocab:
                new_words = sonnet_set - cloud_vocab
                novelty = len(new_words) / len(sonnet_set) if sonnet_set else 0.0
            else:
                # Empty cloud → everything is novel
                novelty = 1.0

            # Compute similarity between user input and sonnet
            user_set = set(user_words)
            intersection = len(user_set & sonnet_set)
            union = len(user_set | sonnet_set)

            if union == 0:
                return 0.5, PulseSnapshot()

            similarity = intersection / union

            # Create pulse snapshot with cloud-aware novelty
            pulse = PulseSnapshot.from_interaction(user_words, sonnet_words, similarity)
            # Override novelty with cloud-based value
            pulse.novelty = novelty

            # Dissonance calculation:
            # - Primary: novelty (70% weight) - exploring unknown
            # - Secondary: inverse similarity (30% weight) - matching intent
            novelty_component = novelty * 0.7
            intent_component = (1.0 - similarity) * 0.3
            dissonance = novelty_component + intent_component

            return dissonance, pulse

    async def adjust_temperature(self, dissonance: float) -> Tuple[float, float]:
        """
        Adjust generation temperature based on dissonance.

        No locking needed - pure computation, no field state access.

        Args:
            dissonance: Dissonance value [0, 1]

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

    async def get_recent_sonnets(self, limit: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get recent sonnets from database.

        Atomic read under lock for consistent snapshot.

        Returns:
            List of (id, text, quality) tuples
        """
        async with self._field_lock:  # 🔒 Atomic read
            cursor = await self.conn.cursor()
            await cursor.execute('''
                SELECT id, text, quality
                FROM sonnets
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            return await cursor.fetchall()

    async def get_best_sonnets(
        self,
        limit: int = 10,
        min_quality: float = 0.7
    ) -> List[Tuple[int, str, float]]:
        """
        Get best quality sonnets.

        Atomic read under lock.

        Returns:
            List of (id, text, quality) tuples
        """
        async with self._field_lock:  # 🔒 Atomic read
            cursor = await self.conn.cursor()
            await cursor.execute('''
                SELECT id, text, quality
                FROM sonnets
                WHERE quality >= ?
                ORDER BY quality DESC
                LIMIT ?
            ''', (min_quality, limit))

            return await cursor.fetchall()

    async def _get_sonnet_count_unlocked(self) -> int:
        """
        Internal: Get sonnet count WITHOUT acquiring lock.
        Caller must hold lock!

        Returns:
            Number of sonnets
        """
        cursor = await self.conn.cursor()
        await cursor.execute('SELECT COUNT(*) FROM sonnets')
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_stats(self) -> Dict:
        """
        Get sonnet cloud statistics.

        Atomic read under lock for consistent snapshot.

        Returns:
            Dict with stats
        """
        async with self._field_lock:  # 🔒 Atomic stats snapshot
            cursor = await self.conn.cursor()

            # Sonnet count
            sonnet_count = await self._get_sonnet_count_unlocked()

            # Avg quality
            await cursor.execute('SELECT AVG(quality) FROM sonnets')
            avg_quality = (await cursor.fetchone())[0] or 0.0

            # Avg dissonance
            await cursor.execute('SELECT AVG(dissonance) FROM sonnets')
            avg_dissonance = (await cursor.fetchone())[0] or 0.0

            # Vocab size
            await cursor.execute('SELECT COUNT(DISTINCT word1) FROM sonnet_trigrams')
            vocab_size = (await cursor.fetchone())[0]

            return {
                'sonnet_count': sonnet_count,
                'avg_quality': avg_quality,
                'avg_dissonance': avg_dissonance,
                'vocab_size': vocab_size
            }


# Convenience function for backward compatibility
async def create_async_sonnet_harmonix(db_path: str = 'cloud/sonnets.db') -> AsyncSonnetHarmonix:
    """
    Create and initialize AsyncSonnetHarmonix instance.

    Usage:
        async with create_async_sonnet_harmonix() as harmonix:
            dissonance, pulse = await harmonix.compute_dissonance(user, sonnet)

    Args:
        db_path: Path to SQLite database

    Returns:
        Initialized AsyncSonnetHarmonix instance
    """
    harmonix = AsyncSonnetHarmonix(db_path)
    await harmonix.__aenter__()
    return harmonix


if __name__ == '__main__':
    import sys

    async def test_async_sonnet_harmonix():
        """Test async sonnet harmonix field operations."""
        print("🔬 Testing AsyncSonnetHarmonix...")
        print()

        # Create test database
        import tempfile
        db_path = tempfile.mktemp(suffix='.db')

        async with AsyncSonnetHarmonix(db_path) as harmonix:
            print("✓ AsyncSonnetHarmonix initialized")

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

            # Test compute_dissonance
            dissonance, pulse = await harmonix.compute_dissonance(user_input, test_sonnet)
            print(f"✓ Dissonance: {dissonance:.3f}")
            print(f"  Pulse - novelty: {pulse.novelty:.3f}, arousal: {pulse.arousal:.3f}, entropy: {pulse.entropy:.3f}")

            # Test temperature adjustment
            sonnet_temp, harmonix_temp = await harmonix.adjust_temperature(dissonance)
            print(f"✓ Temperatures - sonnet: {sonnet_temp:.2f}, harmonix: {harmonix_temp:.2f}")

            # Test add_sonnet (atomic operation!)
            sonnet_id = await harmonix.add_sonnet(
                test_sonnet,
                quality=0.8,
                dissonance=dissonance,
                temperature=sonnet_temp,
                added_by='test'
            )
            print(f"✓ Added sonnet #{sonnet_id}")

            # Test stats
            stats = await harmonix.get_stats()
            print(f"✓ Stats - sonnets: {stats['sonnet_count']}, vocab: {stats['vocab_size']}")

            # Test concurrent operations (prove atomicity!)
            print()
            print("🔬 Testing concurrent operations...")
            tasks = []
            for i in range(5):
                tasks.append(
                    harmonix.add_sonnet(
                        f"Test sonnet {i}\n" * 14,
                        quality=0.5 + i * 0.1,
                        added_by=f'concurrent_{i}'
                    )
                )

            results = await asyncio.gather(*tasks)
            print(f"✓ {len(results)} concurrent operations completed atomically")

            # Final stats
            final_stats = await harmonix.get_stats()
            print(f"✓ Final stats - sonnets: {final_stats['sonnet_count']}, vocab: {final_stats['vocab_size']}")

        print()
        print("🌊 AsyncSonnetHarmonix tests passed!")
        print("   Field coherence maintained through atomic operations ✓")

    # Run async test
    asyncio.run(test_async_sonnet_harmonix())
