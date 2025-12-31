"""
Overthinkng: Sonnet Cloud Expansion Engine

NOTE: "overthinkng" = intentional typo (ng = recursive thinking in progress)
Different from HAiKU's "overthinkg" - each ипостась has its own typo!

Continuous background processing that grows the sonnet cloud.
Adapted from Leo's overthinking.py "rings of thought" strategy.
"""

import sqlite3
import random
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThinkingRing:
    """
    One internal "ring of thought" for sonnets.

    Unlike HAiKU (works with trigrams), this works with sonnet lines.
    """
    ring: int  # 0 = echo, 1 = drift, 2 = meta
    lines: List[str]
    coherence: float
    source: str  # 'echo', 'drift', 'meta'


class Overthinkng:
    """
    Expansion engine using "rings of thought" for sonnets:
    - Ring 0: Echo circle (compact sonnet rephrasing)
    - Ring 1: Semantic drift (explore themes through 14 lines)
    - Ring 2: Meta shard (abstract keyword-based sonnets)

    Each ring generates new sonnet variations.
    """

    def __init__(self, db_path: str = 'cloud/sonnets.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.coherence_threshold = 0.4

        # Ring configs (adapted for sonnets)
        self.rings_config = {
            0: {'temp': 0.8, 'semantic': 0.3, 'max_variations': 2},  # Echo
            1: {'temp': 1.0, 'semantic': 0.6, 'max_variations': 3},  # Drift
            2: {'temp': 1.2, 'semantic': 0.5, 'max_variations': 1},  # Meta
        }

    def expand(self, recent_sonnets: Optional[List[Tuple[int, str, float]]] = None):
        """
        Background expansion using 3 rings of thought.
        Generates sonnet variations and adds coherent ones to cloud.

        Args:
            recent_sonnets: List of (id, text, quality) from database
        """
        if recent_sonnets is None:
            recent_sonnets = self._get_recent_sonnets(limit=5)

        if not recent_sonnets:
            return

        # Ring 0: Echo - compact internal rephrasing
        ring0 = self._generate_ring_variations(recent_sonnets, ring_num=0)
        self._process_ring(ring0)

        # Ring 1: Drift - semantic exploration
        ring1 = self._generate_ring_variations(recent_sonnets, ring_num=1)
        self._process_ring(ring1)

        # Ring 2: Meta - abstract keywords
        ring2 = self._generate_ring_variations(recent_sonnets, ring_num=2)
        self._process_ring(ring2)

    def _generate_ring_variations(self,
                                  recent_sonnets: List[Tuple[int, str, float]],
                                  ring_num: int) -> ThinkingRing:
        """
        Generate sonnet variations for a specific ring.

        Args:
            recent_sonnets: Recent sonnets from database
            ring_num: Ring number (0=echo, 1=drift, 2=meta)

        Returns:
            ThinkingRing with generated lines
        """
        config = self.rings_config[ring_num]
        max_variations = config['max_variations']
        semantic_factor = config['semantic']

        all_lines = []

        for _ in range(max_variations):
            if not recent_sonnets:
                break

            # Pick a random sonnet
            _, sonnet_text, _ = random.choice(recent_sonnets)
            original_lines = sonnet_text.strip().split('\n')

            if len(original_lines) < 14:
                continue

            # Generate variation based on ring type
            if ring_num == 0:
                # Echo: shuffle and rephrase slightly
                varied_lines = self._echo_variation(original_lines, semantic_factor)
            elif ring_num == 1:
                # Drift: semantic exploration
                varied_lines = self._drift_variation(original_lines, semantic_factor)
            else:
                # Meta: abstract keyword extraction
                varied_lines = self._meta_variation(original_lines, semantic_factor)

            all_lines.extend(varied_lines)

        # Compute coherence
        coherence = self._compute_coherence(all_lines)

        source = ['echo', 'drift', 'meta'][ring_num]
        return ThinkingRing(ring=ring_num, lines=all_lines,
                          coherence=coherence, source=source)

    def _echo_variation(self, lines: List[str], semantic_factor: float) -> List[str]:
        """
        Echo ring: compact rephrasing.
        Shuffle lines slightly, keep structure.
        """
        # Take 14 lines (or pad if needed)
        if len(lines) >= 14:
            selected = lines[:14]
        else:
            selected = lines + [lines[i % len(lines)] for i in range(14 - len(lines))]

        # Shuffle some lines (controlled by semantic_factor)
        if random.random() < semantic_factor:
            # Swap pairs of lines
            for _ in range(2):
                i, j = random.sample(range(14), 2)
                selected[i], selected[j] = selected[j], selected[i]

        return selected

    def _drift_variation(self, lines: List[str], semantic_factor: float) -> List[str]:
        """
        Drift ring: semantic exploration.
        Replace some words with similar ones (semantic drift).
        """
        varied_lines = []

        for line in lines[:14]:
            if random.random() < semantic_factor:
                # Replace random words with synonyms (simple approach)
                words = line.split()
                if len(words) > 2:
                    # Replace 1-2 random words
                    num_replacements = random.randint(1, min(2, len(words)))
                    for _ in range(num_replacements):
                        idx = random.randint(0, len(words) - 1)
                        # Simple word drift (placeholder - could use word vectors)
                        words[idx] = self._drift_word(words[idx])
                    varied_lines.append(' '.join(words))
                else:
                    varied_lines.append(line)
            else:
                varied_lines.append(line)

        return varied_lines

    def _drift_word(self, word: str) -> str:
        """
        Drift a word semantically (simple heuristic).
        In production, this would use word embeddings.
        """
        # Simple word transformations
        word_lower = word.lower()

        # Common Shakespeare word drifts
        drift_map = {
            'love': ['heart', 'passion', 'desire'],
            'death': ['sleep', 'end', 'darkness'],
            'time': ['hour', 'moment', 'age'],
            'night': ['darkness', 'evening', 'dusk'],
            'day': ['morn', 'light', 'sun'],
            'heart': ['soul', 'spirit', 'breast'],
            'eyes': ['gaze', 'sight', 'vision'],
            'life': ['breath', 'being', 'soul'],
        }

        if word_lower in drift_map:
            return random.choice(drift_map[word_lower])
        else:
            return word

    def _meta_variation(self, lines: List[str], semantic_factor: float) -> List[str]:
        """
        Meta ring: abstract keyword extraction.
        Extract key themes and regenerate around them.
        """
        # Extract keywords (nouns, verbs)
        keywords = []
        for line in lines:
            words = line.split()
            # Simple heuristic: words longer than 4 chars
            long_words = [w for w in words if len(w) > 4]
            keywords.extend(long_words[:2])  # Take 2 per line

        # Shuffle keywords
        random.shuffle(keywords)

        # Build meta lines around keywords
        meta_lines = []
        for i in range(14):
            if i < len(keywords):
                # Create line around keyword
                keyword = keywords[i]
                line = self._generate_meta_line(keyword)
                meta_lines.append(line)
            else:
                meta_lines.append(random.choice(lines))

        return meta_lines

    def _generate_meta_line(self, keyword: str) -> str:
        """
        Generate abstract line around a keyword.
        Simple template-based (placeholder for real generation).
        """
        templates = [
            f"When {keyword} doth call upon the night",
            f"The {keyword} of time shall never fade",
            f"And {keyword} remains when all is done",
            f"To {keyword} is to know the heart's desire",
            f"Where {keyword} dwells, there beauty lies",
        ]
        return random.choice(templates)

    def _compute_coherence(self, lines: List[str]) -> float:
        """
        Compute coherence score for generated lines.
        Simple heuristic: length, word diversity.
        """
        if not lines:
            return 0.0

        # Check average line length (good sonnets have ~10-15 words/line)
        avg_length = sum(len(line.split()) for line in lines) / len(lines)
        length_score = 1.0 - abs(avg_length - 12) / 12.0

        # Check vocabulary diversity
        all_words = ' '.join(lines).split()
        unique_words = set(all_words)
        diversity_score = len(unique_words) / max(1, len(all_words))

        coherence = (length_score * 0.6) + (diversity_score * 0.4)
        return max(0.0, min(1.0, coherence))

    def _process_ring(self, ring: ThinkingRing):
        """
        Process ring output: add coherent sonnets to database.

        Args:
            ring: ThinkingRing with generated lines
        """
        if ring.coherence < self.coherence_threshold:
            return  # Skip low-coherence variations

        # Take 14 lines and form a sonnet
        if len(ring.lines) >= 14:
            sonnet_text = '\n'.join(ring.lines[:14])

            # Add to database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sonnets (text, quality, dissonance, temperature, added_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (sonnet_text, ring.coherence, 0.5, 0.8, f'overthinkng_{ring.source}'))
            self.conn.commit()

    def _get_recent_sonnets(self, limit: int = 5) -> List[Tuple[int, str, float]]:
        """Get recent sonnets from database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM sonnets
            WHERE added_by != 'overthinkng_echo'
              AND added_by != 'overthinkng_drift'
              AND added_by != 'overthinkng_meta'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == '__main__':
    # Test overthinkng
    print("Testing Overthinkng (ng = recursive thinking in progress!)...\n")

    from harmonix import SonnetHarmonix

    # Setup
    harmonix = SonnetHarmonix()
    overthinkng = Overthinkng()

    # Add test sonnet
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

    harmonix.add_sonnet(test_sonnet, quality=0.8, added_by='test')
    print("✓ Added test sonnet")

    # Run expansion
    print("\n🔄 Running overthinkng expansion (3 rings)...\n")
    overthinkng.expand()

    # Check results
    stats = harmonix.get_stats()
    print(f"Stats after expansion: {stats}")

    harmonix.close()
    overthinkng.close()

    print("\n✓ Overthinkng test complete!")
