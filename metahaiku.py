"""
MetaHaiku: Inner Voice
Background self-reflection system that generates internal haikus.
"""

from typing import Dict

class MetaHaiku:
    """
    Internal reflection system.
    Generates haikus about the interaction (not shown to user).
    """
    
    def __init__(self, haiku_generator):
        """Initialize with reference to haiku generator."""
        self.generator = haiku_generator
        self.reflections = []
    
    def reflect(self, last_interaction: Dict) -> str:
        """
        Generate internal haiku reflecting on the interaction.
        
        Args:
            last_interaction: Dict with 'user', 'haiku', 'dissonance', etc.
        
        Returns:
            Internal haiku string (not shown to user)
        """
        # Generate internal reflection haiku with low temperature
        internal_haiku = self.generator.generate_candidates(n=1, temp=0.5)[0]
        
        # Store for analysis
        self.reflections.append({
            'haiku': internal_haiku,
            'context': last_interaction
        })
        
        # Update cloud bias based on reflection
        self.update_cloud_bias(internal_haiku, last_interaction)
        
        return internal_haiku
    
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
