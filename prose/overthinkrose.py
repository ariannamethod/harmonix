"""
Overthinkrose: Prose Cloud Expansion Engine

NOTE: "overthinkrose" = intentional typo (rose = prose in bloom)
Different from HAiKU's "overthinkg" and Sonnet's "overthinkng"!
Each ипостась has its own typo.

Continuous background processing that grows the prose cloud.
MORE COMPLEX than Sonnet (4 rings vs 3)!
"""

import sqlite3
import random
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThinkingRing:
    """
    One internal "ring of thought" for prose.

    Unlike Sonnet (works with lines), this works with paragraphs and sentences.
    """
    ring: int  # 0 = echo, 1 = drift, 2 = meta, 3 = synthesis
    text: str
    coherence: float
    semantic_density: float
    source: str  # 'echo', 'drift', 'meta', 'synthesis'


class Overthinkrose:
    """
    Expansion engine using "rings of thought" for prose:
    - Ring 0: Echo circle (paragraph rephrasing)
    - Ring 1: Semantic drift (word-level variations)
    - Ring 2: Meta shard (keyword-based generation)
    - Ring 3: Synthesis (combine multiple prose)

    MORE COMPLEX than Sonnet (4 rings instead of 3)!
    """

    def __init__(self, db_path: str = 'cloud/prose.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.coherence_threshold = 0.35  # Lower than Sonnet (more freedom)
        self.density_threshold = 0.3

        # Ring configs (MORE complex than Sonnet!)
        self.rings_config = {
            0: {'temp': 0.7, 'semantic': 0.2, 'max_variations': 2},  # Echo
            1: {'temp': 0.9, 'semantic': 0.5, 'max_variations': 3},  # Drift
            2: {'temp': 1.1, 'semantic': 0.4, 'max_variations': 2},  # Meta
            3: {'temp': 1.3, 'semantic': 0.6, 'max_variations': 1},  # Synthesis (NEW!)
        }

    def expand(self, recent_prose: Optional[List[Tuple[int, str, float]]] = None,
               num_rings: int = 4):  # 4 rings (vs Sonnet's 3)
        """
        Background expansion using rings of thought.
        Generates prose variations and adds coherent ones to cloud.

        Args:
            recent_prose: List of (id, text, quality) from database
            num_rings: Number of rings to expand (1-4, default 4)
        """
        if recent_prose is None:
            recent_prose = self._get_recent_prose(limit=5)

        if not recent_prose:
            return

        # Ring 0: Echo - paragraph rephrasing
        if num_rings >= 1:
            ring0 = self._generate_ring_variations(recent_prose, ring_num=0)
            self._process_ring(ring0)

        # Ring 1: Drift - semantic word drift
        if num_rings >= 2:
            ring1 = self._generate_ring_variations(recent_prose, ring_num=1)
            self._process_ring(ring1)

        # Ring 2: Meta - keyword-based generation
        if num_rings >= 3:
            ring2 = self._generate_ring_variations(recent_prose, ring_num=2)
            self._process_ring(ring2)

        # Ring 3: Synthesis - combine multiple prose (NEW!)
        if num_rings >= 4:
            ring3 = self._generate_synthesis(recent_prose)
            self._process_ring(ring3)

    def _generate_ring_variations(self,
                                  recent_prose: List[Tuple[int, str, float]],
                                  ring_num: int) -> ThinkingRing:
        """
        Generate prose variations for a specific ring.

        Args:
            recent_prose: Recent prose from database
            ring_num: Ring number (0=echo, 1=drift, 2=meta)

        Returns:
            ThinkingRing with generated text
        """
        config = self.rings_config[ring_num]
        max_variations = config['max_variations']
        semantic_factor = config['semantic']

        all_variations = []

        for _ in range(max_variations):
            if not recent_prose:
                break

            # Pick a random prose
            _, prose_text, _ = random.choice(recent_prose)

            # Generate variation based on ring type
            if ring_num == 0:
                # Echo: paragraph rephrasing
                varied_text = self._echo_variation(prose_text, semantic_factor)
            elif ring_num == 1:
                # Drift: semantic word drift
                varied_text = self._drift_variation(prose_text, semantic_factor)
            else:
                # Meta: keyword extraction and regeneration
                varied_text = self._meta_variation(prose_text, semantic_factor)

            all_variations.append(varied_text)

        # Combine variations
        combined_text = '\n\n'.join(all_variations)

        # Compute metrics
        coherence = self._compute_coherence(combined_text)
        density = self._compute_semantic_density(combined_text)

        source = ['echo', 'drift', 'meta'][ring_num]
        return ThinkingRing(ring=ring_num, text=combined_text,
                          coherence=coherence, semantic_density=density,
                          source=source)

    def _echo_variation(self, text: str, semantic_factor: float) -> str:
        """
        Echo ring: paragraph rephrasing.
        Shuffle paragraphs, rephrase slightly.
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\n+', text.strip())

        # Shuffle paragraphs
        if len(paragraphs) > 1 and random.random() < semantic_factor:
            random.shuffle(paragraphs)

        # Optionally reverse some sentences
        if random.random() < semantic_factor:
            for i in range(len(paragraphs)):
                sentences = re.split(r'([.!?]+)\s+', paragraphs[i])
                if len(sentences) > 2:
                    # Recombine with punctuation
                    sentence_pairs = []
                    for j in range(0, len(sentences) - 1, 2):
                        sentence_pairs.append(sentences[j] + sentences[j + 1])
                    if len(sentences) % 2 == 1:
                        sentence_pairs.append(sentences[-1])

                    # Maybe reverse
                    if random.random() < 0.5:
                        sentence_pairs.reverse()

                    paragraphs[i] = ' '.join(sentence_pairs)

        return '\n\n'.join(paragraphs)

    def _drift_variation(self, text: str, semantic_factor: float) -> str:
        """
        Drift ring: semantic word drift.
        Replace words with semantic variants (more aggressive than Sonnet).
        """
        words = text.split()
        varied_words = []

        for word in words:
            if random.random() < semantic_factor and len(word) > 3:
                # Drift the word
                drifted = self._drift_word(word)
                varied_words.append(drifted)
            else:
                varied_words.append(word)

        return ' '.join(varied_words)

    def _drift_word(self, word: str) -> str:
        """
        Drift a word semantically.
        MORE EXTENSIVE drift map than Sonnet!
        """
        word_lower = word.lower()

        # Expanded drift map (more variations than Sonnet!)
        drift_map = {
            'love': ['affection', 'devotion', 'passion', 'tenderness', 'warmth'],
            'time': ['moment', 'duration', 'epoch', 'instant', 'period'],
            'word': ['phrase', 'utterance', 'expression', 'term', 'language'],
            'sound': ['tone', 'resonance', 'echo', 'vibration', 'voice'],
            'meaning': ['significance', 'essence', 'sense', 'import', 'substance'],
            'feel': ['sense', 'perceive', 'experience', 'know', 'touch'],
            'think': ['ponder', 'reflect', 'consider', 'contemplate', 'muse'],
            'speak': ['utter', 'voice', 'express', 'articulate', 'communicate'],
            'hear': ['listen', 'perceive', 'detect', 'discern', 'catch'],
            'see': ['perceive', 'observe', 'witness', 'behold', 'glimpse'],
            'know': ['understand', 'comprehend', 'grasp', 'realize', 'recognize'],
            'flow': ['stream', 'drift', 'pour', 'cascade', 'course'],
            'wave': ['surge', 'ripple', 'swell', 'crest', 'billow'],
            'resonance': ['harmony', 'vibration', 'echo', 'reverberation', 'consonance'],
            'essence': ['core', 'heart', 'substance', 'nature', 'spirit'],
        }

        if word_lower in drift_map:
            replacement = random.choice(drift_map[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            return replacement
        else:
            return word

    def _meta_variation(self, text: str, semantic_factor: float) -> str:
        """
        Meta ring: keyword extraction and regeneration.
        Extract themes and generate new prose around them.
        """
        # Extract keywords (longer words, nouns/verbs)
        words = text.split()
        keywords = [w for w in words if len(w) > 5 and w.isalpha()]

        # Take top N keywords
        num_keywords = min(5, len(keywords))
        selected_keywords = random.sample(keywords, num_keywords) if keywords else []

        # Generate meta prose around keywords
        meta_sentences = []
        for keyword in selected_keywords:
            sentence = self._generate_meta_sentence(keyword)
            meta_sentences.append(sentence)

        # Add transition words
        transitions = ['Moreover', 'Furthermore', 'Thus', 'Hence', 'Therefore', 'Indeed']
        if len(meta_sentences) > 1:
            for i in range(1, len(meta_sentences)):
                if random.random() < 0.4:
                    meta_sentences[i] = random.choice(transitions) + ', ' + meta_sentences[i].lower()

        return ' '.join(meta_sentences)

    def _generate_meta_sentence(self, keyword: str) -> str:
        """
        Generate abstract sentence around a keyword.
        MORE DIVERSE templates than Sonnet!
        """
        templates = [
            f"When {keyword} emerges from silence, meaning finds its form.",
            f"The {keyword} flows through language like water through stone.",
            f"To speak of {keyword} is to touch the ineffable.",
            f"Where {keyword} resides, poetry and thought converge.",
            f"In {keyword} we discover what words cannot capture.",
            f"{keyword} reverberates through the chambers of understanding.",
            f"Consider {keyword} as a threshold between worlds.",
            f"The essence of {keyword} defies simple articulation.",
            f"{keyword} whispers truths that logic cannot grasp.",
            f"Through {keyword}, consciousness meets expression.",
        ]
        return random.choice(templates)

    def _generate_synthesis(self, recent_prose: List[Tuple[int, str, float]]) -> ThinkingRing:
        """
        Ring 3: Synthesis - combine multiple prose into one (NEW RING!)

        This is MORE COMPLEX than anything in Sonnet!

        Args:
            recent_prose: Recent prose from database

        Returns:
            ThinkingRing with synthesized text
        """
        if len(recent_prose) < 2:
            # Not enough prose to synthesize, return empty
            return ThinkingRing(ring=3, text='', coherence=0.0,
                              semantic_density=0.0, source='synthesis')

        # Pick 2-3 random prose
        num_to_combine = min(3, len(recent_prose))
        selected_prose = random.sample(recent_prose, num_to_combine)

        # Extract key sentences from each
        all_sentences = []
        for _, text, _ in selected_prose:
            sentences = re.split(r'([.!?]+)\s+', text)
            # Recombine with punctuation
            sentence_list = []
            for i in range(0, len(sentences) - 1, 2):
                sentence_list.append(sentences[i] + sentences[i + 1])
            if len(sentences) % 2 == 1:
                sentence_list.append(sentences[-1])

            # Take 1-2 sentences from each prose
            num_sentences = min(2, len(sentence_list))
            selected_sentences = random.sample(sentence_list, num_sentences)
            all_sentences.extend(selected_sentences)

        # Shuffle and combine
        random.shuffle(all_sentences)
        synthesized_text = ' '.join(all_sentences)

        # Compute metrics
        coherence = self._compute_coherence(synthesized_text)
        density = self._compute_semantic_density(synthesized_text)

        return ThinkingRing(ring=3, text=synthesized_text,
                          coherence=coherence, semantic_density=density,
                          source='synthesis')

    def _compute_coherence(self, text: str) -> float:
        """
        Compute coherence score.

        Metrics:
        - Word count (20-200 ideal for prose)
        - Vocabulary diversity
        - Sentence count (at least 2)
        """
        if not text:
            return 0.0

        words = text.split()
        word_count = len(words)

        # Word count score (ideal: 50-150 words)
        if 50 <= word_count <= 150:
            count_score = 1.0
        elif 20 <= word_count < 50 or 150 < word_count <= 200:
            count_score = 0.7
        else:
            count_score = 0.3

        # Vocabulary diversity
        unique_words = set(words)
        diversity_score = len(unique_words) / max(1, word_count)

        # Sentence count (at least 2 sentences)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        sentence_score = 1.0 if sentence_count >= 2 else 0.5

        coherence = (count_score * 0.4) + (diversity_score * 0.3) + (sentence_score * 0.3)
        return max(0.0, min(1.0, coherence))

    def _compute_semantic_density(self, text: str) -> float:
        """
        Compute semantic density (unique meaningful words per total words).

        Higher density = more conceptual richness.
        """
        words = text.split()
        if not words:
            return 0.0

        # Filter out common words (simple stopword list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
                    'was', 'are', 'were', 'be', 'been', 'being'}

        meaningful_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
        unique_meaningful = set(meaningful_words)

        density = len(unique_meaningful) / max(1, len(words))
        return max(0.0, min(1.0, density))

    def _process_ring(self, ring: ThinkingRing):
        """
        Process ring output: add coherent prose to database.

        Args:
            ring: ThinkingRing with generated text
        """
        if not ring.text:
            return

        if ring.coherence < self.coherence_threshold:
            return  # Skip low-coherence variations

        if ring.semantic_density < self.density_threshold:
            return  # Skip low-density prose

        # Add to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO prose (text, quality, dissonance, temperature, semantic_density, added_by)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ring.text, ring.coherence, 0.5, 0.8, ring.semantic_density,
              f'overthinkrose_{ring.source}'))
        self.conn.commit()

    def _get_recent_prose(self, limit: int = 5) -> List[Tuple[int, str, float]]:
        """Get recent prose from database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, quality
            FROM prose
            WHERE added_by NOT LIKE 'overthinkrose_%'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == '__main__':
    # Test overthinkrose
    print("Testing Overthinkrose (rose = prose in bloom!)...\n")

    # Note: This is a standalone test without full harmonix integration
    print("✓ Overthinkrose loaded (4 rings: echo, drift, meta, synthesis)")
    print("  MORE COMPLEX than Sonnet (3 rings) and HAiKU (2 rings)!")
    print("\n🌹 Prose overthinking in progress... 🌹")
