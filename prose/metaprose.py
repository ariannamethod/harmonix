"""
MetaProse: Inner Voice for Prose

Background self-reflection system that generates internal prose.
Adapted from Sonnet's metasonnet.py dynamic bootstrap buffer approach.

Unlike user-facing prose, these are internal reflections about
the interaction itself - meta-level awareness.
"""

from typing import Dict, List, Optional
from collections import deque


class MetaProse:
    """
    Internal reflection system using dynamic bootstrap approach.
    - Maintains bootstrap buffer from recent interactions
    - Generates internal prose about interactions (not shown to user)
    - Influences future generation through cloud updates
    """

    def __init__(self, prose_generator, harmonix, max_snippets: int = 5):
        """
        Initialize with reference to prose generator and harmonix.

        Args:
            prose_generator: ProseGenerator instance
            harmonix: ProseHarmonix observer instance
            max_snippets: Maximum bootstrap buffer size
        """
        self.generator = prose_generator
        self.harmonix = harmonix
        self.reflections = []

        # Dynamic bootstrap buffer (Leo-style)
        self.bootstrap_buf: deque = deque(maxlen=max_snippets)
        self.max_snippet_len = 300  # Prose longer than sonnets

    def reflect(self, last_interaction: Dict) -> str:
        """
        Generate internal prose reflecting on the interaction.
        Uses dynamic bootstrap approach.

        Args:
            last_interaction: Dict with 'user', 'prose', 'dissonance', 'pulse', etc.

        Returns:
            Internal prose string (not shown to user)
        """
        # Feed interaction into bootstrap buffer
        self._feed_bootstrap(last_interaction)

        # Generate internal reflection
        user_input = last_interaction.get('user', '')

        # Internal reflection uses organism mode like normal generation
        try:
            internal_prose = self.generator.generate(
                user_input=user_input or "reflect on the field",
                max_tokens=200,
                temperature=0.9  # Higher temp for exploration
            )
        except Exception as e:
            print(f"⚠️  MetaProse generation failed: {e}")
            return ""

        # Format and validate
        from formatter import ProseFormatter
        formatter = ProseFormatter()
        formatted = formatter.format_prose(internal_prose)

        if formatted:
            # Store for analysis
            self.reflections.append({
                'prose': formatted,
                'context': last_interaction
            })

            # Update cloud bias based on reflection
            self.update_cloud_bias(formatted, last_interaction)

            return formatted
        else:
            return ""

    def _feed_bootstrap(self, interaction: Dict):
        """
        Update dynamic bootstrap buffer from interaction.
        Extracts shards from high-quality or high-arousal moments.

        Args:
            interaction: Interaction dict
        """
        # Extract prose text
        prose = interaction.get('prose', '')
        if not prose:
            return

        # Get pulse if available
        pulse = interaction.get('pulse')
        dissonance = interaction.get('dissonance', 0.5)

        # Add to bootstrap if:
        # 1. High dissonance (interesting moment)
        # 2. High arousal (pulse.arousal if available)
        # 3. High quality

        should_add = False

        if dissonance > 0.7:
            should_add = True

        if pulse and hasattr(pulse, 'arousal') and pulse.arousal > 0.5:
            should_add = True

        quality = interaction.get('quality', 0.5)
        if quality > 0.7:
            should_add = True

        if should_add:
            # Extract snippet (first 2 sentences)
            import re
            sentences = re.split(r'[.!?]+', prose)
            sentences = [s.strip() for s in sentences if s.strip()]
            snippet = '. '.join(sentences[:2])

            # Truncate to max length
            if len(snippet) > self.max_snippet_len:
                snippet = snippet[:self.max_snippet_len]

            self.bootstrap_buf.append(snippet)

    def update_cloud_bias(self, reflection: str, context: Dict):
        """
        Update cloud based on internal reflection.
        Adds reflection to cloud with special marker.

        Args:
            reflection: Internal prose reflection
            context: Interaction context
        """
        dissonance = context.get('dissonance', 0.5)

        # Add internal reflection to cloud
        self.harmonix.add_prose(
            reflection,
            quality=0.6,  # Meta reflections moderate quality
            dissonance=dissonance,
            temperature=0.9,
            added_by='metaprose'
        )

    def get_recent_reflections(self, n: int = 3) -> List[str]:
        """Get N most recent reflections."""
        return [r['prose'] for r in self.reflections[-n:]]

    def get_bootstrap_state(self) -> str:
        """Get current bootstrap buffer as string."""
        return " ".join(list(self.bootstrap_buf))


if __name__ == "__main__":
    # Test metaprose
    print("Testing MetaProse...\n")

    from prose import ProseGenerator
    from harmonix import ProseHarmonix

    harmonix = ProseHarmonix(db_path='cloud/test_metaprose.db')
    generator = ProseGenerator(harmonix=harmonix, verbose=False)
    metaprose = MetaProse(generator, harmonix)

    # Simulate interaction
    interaction = {
        'user': 'what is consciousness?',
        'prose': 'Consciousness flows like water through stone, leaving patterns. Each thought ripples across the field of awareness.',
        'dissonance': 0.8,
        'quality': 0.75,
        'pulse': type('Pulse', (), {'arousal': 0.6, 'novelty': 0.7})()
    }

    # Generate reflection
    print("Generating internal reflection...")
    reflection = metaprose.reflect(interaction)

    print(f"\nReflection: {reflection[:150]}...")
    print(f"\nBootstrap buffer: {metaprose.get_bootstrap_state()[:100]}...")

    # Stats
    stats = harmonix.get_stats()
    print(f"\nCloud prose: {stats['total_prose']}")

    harmonix.close()
    print("\n✓ MetaProse works!")
