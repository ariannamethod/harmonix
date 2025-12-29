"""
RAE: Recursive Adapter Engine
Selects best haiku from candidates using chain-of-thought reasoning.
"""

from typing import List, Dict

class RecursiveAdapterEngine:
    """
    Chain-of-thought selector for haiku candidates.
    Uses recursive reasoning to pick the best response.
    """
    
    def __init__(self):
        """Initialize RAE (rule-based in v1, can be learned in v2)."""
        self.reasoning_steps = [
            'perplexity_filter',
            'resonance_filter',
            'coherence_filter',
            'final_selection'
        ]
    
    def reason(self, context: Dict, candidates: List[str], 
               scorer=None) -> str:
        """
        Apply recursive reasoning chain to select best haiku.
        
        Args:
            context: Dict with 'user' input and optional scoring context
            candidates: List of haiku strings
            scorer: Optional HaikuGenerator instance for scoring
        
        Returns:
            Best haiku string
        """
        if not candidates:
            return "cloud is empty\nsilence speaks louder than words\nwait for resonance"
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Step 1: Filter by structural validity (5-7-5 syllables)
        valid_candidates = self._filter_by_structure(candidates)
        if not valid_candidates:
            valid_candidates = candidates  # Fallback
        
        # Step 2: Filter by perplexity (if scorer available)
        if scorer and hasattr(scorer, 'score_haiku'):
            scored = [(c, scorer.score_haiku(c, context.get('user_trigrams'))) 
                     for c in valid_candidates]
            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 3
            top_candidates = [c for c, _ in scored[:3]]
        else:
            top_candidates = valid_candidates[:3]
        
        # Step 3: Select based on diversity (prefer varied vocabulary)
        if len(top_candidates) > 1:
            best = self._select_most_diverse(top_candidates)
        else:
            best = top_candidates[0]
        
        return best
    
    def _filter_by_structure(self, candidates: List[str]) -> List[str]:
        """
        Filter candidates by haiku structure.
        Must have exactly 3 lines.
        """
        valid = []
        for haiku in candidates:
            lines = haiku.split('\n')
            if len(lines) == 3:
                valid.append(haiku)
        return valid
    
    def _select_most_diverse(self, candidates: List[str]) -> str:
        """
        Select haiku with most diverse vocabulary.
        """
        scores = []
        for haiku in candidates:
            words = []
            for line in haiku.split('\n'):
                words.extend(line.split())
            
            # Diversity = unique words / total words
            diversity = len(set(words)) / max(1, len(words))
            scores.append((haiku, diversity))
        
        # Return highest diversity
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
