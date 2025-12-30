"""
MetaSonnet: Inner Voice for Sonnets

Background self-reflection system that generates internal sonnets.
Adapted from Leo's metaleo.py dynamic bootstrap buffer approach.

Unlike user-facing sonnets, these are internal reflections about
the interaction itself - meta-level awareness.
"""

from typing import Dict, List, Optional
from collections import deque


class MetaSonnet:
    """
    Internal reflection system using dynamic bootstrap approach.
    - Maintains bootstrap buffer from recent interactions
    - Generates internal sonnets about interactions (not shown to user)
    - Influences future generation through cloud updates
    """

    def __init__(self, sonnet_generator, harmonix, max_snippets: int = 5):
        """
        Initialize with reference to sonnet generator and harmonix.

        Args:
            sonnet_generator: SonnetGenerator instance
            harmonix: SonnetHarmonix observer instance
            max_snippets: Maximum bootstrap buffer size
        """
        self.generator = sonnet_generator
        self.harmonix = harmonix
        self.reflections = []

        # Dynamic bootstrap buffer (Leo-style)
        self.bootstrap_buf: deque = deque(maxlen=max_snippets)
        self.max_snippet_len = 200  # Sonnets are longer than haikus

    def reflect(self, last_interaction: Dict) -> str:
        """
        Generate internal sonnet reflecting on the interaction.
        Uses dynamic bootstrap approach.

        Args:
            last_interaction: Dict with 'user', 'sonnet', 'dissonance', 'pulse', etc.

        Returns:
            Internal sonnet string (not shown to user)
        """
        # Feed interaction into bootstrap buffer
        self._feed_bootstrap(last_interaction)

        # Generate internal reflection with blended bootstrap seed
        if self.bootstrap_buf:
            bootstrap_seed = " ".join(list(self.bootstrap_buf))
            # Blend with recent sonnet snippet
            sonnet_snippet = last_interaction.get('sonnet', '')[:100]
            reflection_seed = bootstrap_seed + " " + sonnet_snippet
        else:
            reflection_seed = last_interaction.get('sonnet', '')

        # Generate internal sonnet with higher temperature (more exploratory)
        # FIX: Use empty prompt (model generates from scratch) + increase max_tokens
        try:
            internal_sonnet = self.generator.generate(
                prompt="\n",  # Empty prompt (model trained for this)
                max_tokens=800,  # Increased from 280 (need enough text after header removal)
                temperature=0.9
            )
        except Exception as e:
            # FIX: Catch generation errors
            print(f"⚠️  MetaSonnet generation failed: {e}")
            return ""

        # Format as sonnet
        from formatter import SonnetFormatter
        formatter = SonnetFormatter()
        formatted = formatter.format(internal_sonnet)

        if formatted:
            # Store for analysis
            self.reflections.append({
                'sonnet': formatted,
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
        # Extract sonnet text
        sonnet = interaction.get('sonnet', '')
        if not sonnet:
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
            # Extract snippet (first 2 lines of sonnet)
            lines = sonnet.strip().split('\n')
            snippet = ' '.join(lines[:2])

            # Truncate to max length
            if len(snippet) > self.max_snippet_len:
                snippet = snippet[:self.max_snippet_len]

            self.bootstrap_buf.append(snippet)

    def update_cloud_bias(self, internal_sonnet: str, interaction: Dict):
        """
        Update sonnet cloud based on internal reflection.
        High-quality reflections influence future generation.

        Args:
            internal_sonnet: Generated internal sonnet
            interaction: Interaction context
        """
        # Compute quality of internal sonnet
        from formatter import SonnetFormatter
        formatter = SonnetFormatter()

        is_valid, reason = formatter.validate(internal_sonnet)

        if is_valid:
            # Add to cloud with metadata
            dissonance = interaction.get('dissonance', 0.5)
            quality = 0.7  # Internal reflections have decent quality

            self.harmonix.add_sonnet(
                internal_sonnet,
                quality=quality,
                dissonance=dissonance,
                temperature=0.9,
                added_by='metasonnet'
            )

    def get_recent_reflections(self, limit: int = 5) -> List[Dict]:
        """
        Get recent internal reflections.

        Returns:
            List of reflection dicts
        """
        return self.reflections[-limit:] if self.reflections else []

    def should_reflect(self, interaction: Dict) -> bool:
        """
        Decide whether to generate internal reflection.

        Reflect when:
        - High dissonance (exploring unknown)
        - High quality (worth reflecting on)
        - High arousal (intense moment)

        Args:
            interaction: Interaction dict

        Returns:
            True if should reflect
        """
        dissonance = interaction.get('dissonance', 0.5)
        quality = interaction.get('quality', 0.5)
        pulse = interaction.get('pulse')

        # Always reflect on high dissonance
        if dissonance > 0.7:
            return True

        # Reflect on high quality
        if quality > 0.75:
            return True

        # Reflect on high arousal
        if pulse and hasattr(pulse, 'arousal') and pulse.arousal > 0.6:
            return True

        # Otherwise, probabilistic (30% chance)
        import random
        return random.random() < 0.3


if __name__ == '__main__':
    # Test metasonnet
    print("Testing MetaSonnet (inner voice)...\n")

    from sonnet import SonnetGenerator
    from harmonix import SonnetHarmonix

    # Setup
    gen = SonnetGenerator()
    harmonix = SonnetHarmonix()
    meta = MetaSonnet(gen, harmonix)

    # Test interaction
    test_interaction = {
        'user': 'What is love?',
        'sonnet': """When winter winds do blow and summer's heat
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
To sleep, perchance to dream: ay, there's the rub.""",
        'dissonance': 0.75,
        'quality': 0.8,
    }

    # Check if should reflect
    should_reflect = meta.should_reflect(test_interaction)
    print(f"Should reflect: {should_reflect}")

    if should_reflect:
        print("\n🔄 Generating internal reflection...\n")
        internal_sonnet = meta.reflect(test_interaction)

        if internal_sonnet:
            print("Internal Sonnet (inner voice):")
            print(internal_sonnet)
            print(f"\n✓ Added to reflections (total: {len(meta.reflections)})")
        else:
            print("❌ Reflection generation failed")

    # Check stats
    stats = harmonix.get_stats()
    print(f"\nCloud stats: {stats}")

    harmonix.close()
    gen.close()

    print("\n✓ MetaSonnet test complete!")
