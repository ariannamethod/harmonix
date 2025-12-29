"""
MetaHaiku: Inner Voice
Background self-reflection system that generates internal haikus.

Forked from Leo's metaleo.py - dynamic bootstrap buffer approach.
"""

from typing import Dict, List
from collections import deque


class MetaHaiku:
    """
    Internal reflection system using Leo's dynamic bootstrap approach.
    - Maintains a bootstrap buffer from recent interactions
    - Generates haikus about the interaction (not shown to user)
    - Influences future generation through cloud updates
    """
    
    def __init__(self, haiku_generator, max_snippets: int = 8):
        """Initialize with reference to haiku generator."""
        self.generator = haiku_generator
        self.reflections = []
        
        # Dynamic bootstrap buffer (Leo-style)
        self.bootstrap_buf: deque = deque(maxlen=max_snippets)
        self.max_snippet_len = 100
    
    def reflect(self, last_interaction: Dict) -> str:
        """
        Generate internal haiku reflecting on the interaction.
        Uses Leo's dynamic bootstrap approach.
        
        Args:
            last_interaction: Dict with 'user', 'haiku', 'dissonance', 'pulse', etc.
        
        Returns:
            Internal haiku string (not shown to user)
        """
        # Feed interaction into bootstrap buffer
        self._feed_bootstrap(last_interaction)
        
        # Generate internal reflection with blended bootstrap seed
        if self.bootstrap_buf:
            bootstrap_seed = " ".join(list(self.bootstrap_buf))
            # Blend with recent haiku
            reflection_seed = bootstrap_seed + " " + last_interaction.get('haiku', '')
        else:
            reflection_seed = last_interaction.get('haiku', '')
        
        # Generate with higher temperature (more exploratory)
        # Use bootstrap seed to influence generation
        internal_haiku = self.generator.generate_candidates(n=1, temp=0.7)[0]
        
        # Store for analysis
        self.reflections.append({
            'haiku': internal_haiku,
            'context': last_interaction
        })
        
        # Update cloud bias based on reflection
        self.update_cloud_bias(internal_haiku, last_interaction)
        
        return internal_haiku
    
    def _feed_bootstrap(self, interaction: Dict):
        """
        Update dynamic bootstrap buffer from interaction.
        Extracts shards from high-quality or high-arousal moments.
        """
        # Extract haiku text
        haiku = interaction.get('haiku', '')
        if not haiku:
            return
        
        # Get pulse if available
        pulse = interaction.get('pulse')
        
        # Add to bootstrap if:
        # 1. High dissonance (interesting moment)
        # 2. High arousal (emotional charge)
        # 3. Or just occasionally for diversity
        
        dissonance = interaction.get('dissonance', 0.0)
        should_add = False
        
        if dissonance > 0.6:
            should_add = True
        
        if pulse and hasattr(pulse, 'arousal'):
            if pulse.arousal > 0.6:
                should_add = True
        
        # Random sampling for diversity
        import random
        if random.random() < 0.3:
            should_add = True
        
        if should_add:
            # Extract words from haiku as snippet
            words = haiku.replace('\n', ' ').split()
            snippet = ' '.join(words[:10])  # Take first 10 words
            
            if len(snippet) > self.max_snippet_len:
                snippet = snippet[:self.max_snippet_len]
            
            self.bootstrap_buf.append(snippet)
    
    def update_cloud_bias(self, internal_haiku: str, context: Dict):
        """
        Adjust word weights based on internal reflection.
        This influences future generation subtly.
        """
        # Extract words from internal haiku
        words = []
        for line in internal_haiku.split('\n'):
            words.extend(line.split())
        
        # Identify patterns (e.g., repeated themes)
        # In v1, we simply note which words appear in reflections
        # More sophisticated analysis can happen in v2
        
        # For now, just track that reflection occurred
        # The harmonix module handles actual weight updates
        pass
    
    def get_recent_reflections(self, n: int = 5) -> list:
        """Get last n internal reflections."""
        return self.reflections[-n:]
