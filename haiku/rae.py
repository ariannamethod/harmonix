"""
RAE: Recursive Adapter Engine
Selects best haiku from candidates using chain-of-thought reasoning.

v1.0: Rule-based selector
v1.1: Hybrid - learned recursive selector with rule-based fallback
"""

from typing import List, Dict, Optional

class RecursiveAdapterEngine:
    """
    Chain-of-thought selector for haiku candidates.

    Modes:
    - Rule-based: Uses filters + diversity selection (v1.0)
    - Recursive: Uses micrograd MLP with recursive refinement (v1.1)
    - Hybrid: Try recursive first, fallback to rule-based (default v1.1)
    """

    def __init__(self, use_recursive: bool = True):
        """
        Initialize RAE.

        Args:
            use_recursive: If True, use learned recursive selector (v1.1).
                          Falls back to rule-based if selector not trained.
        """
        self.reasoning_steps = [
            'perplexity_filter',
            'resonance_filter',
            'coherence_filter',
            'final_selection'
        ]

        # Try to load recursive selector (v1.1)
        self.recursive_selector = None
        if use_recursive:
            try:
                from rae_recursive import RecursiveRAESelector
                self.recursive_selector = RecursiveRAESelector()
                print("✓ Loaded recursive RAE selector (v1.1)")
            except Exception as e:
                print(f"⚠️ Recursive selector unavailable: {e}")
                print("⚠️ Using rule-based selector (v1.0)")
    
    def reason(self, context: Dict, candidates: List[str],
               scorer=None) -> str:
        """
        Apply recursive reasoning chain to select best haiku.

        v1.1: Try recursive selector first, fallback to rule-based.

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

        # v1.1: Try recursive selector if available
        if self.recursive_selector is not None:
            try:
                best_haiku, confidence = self.recursive_selector.select_recursive(
                    candidates, context
                )
                return best_haiku
            except Exception as e:
                print(f"⚠️ Recursive selection failed: {e}")
                print("⚠️ Falling back to rule-based selection")

        # v1.0: Rule-based fallback
        return self._reason_rule_based(context, candidates, scorer)

    def _reason_rule_based(self, context: Dict, candidates: List[str],
                          scorer=None) -> str:
        """
        Rule-based selection (v1.0 behavior).

        Process:
        1. Filter by structure (3 lines)
        2. Filter by perplexity (if scorer available)
        3. Select most diverse

        Args:
            context: Context dict
            candidates: Haiku candidates
            scorer: Optional scorer

        Returns:
            Best haiku string
        """
        # Step 1: Filter by structural validity (3 lines)
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

    def observe(self, candidates: List[str], selected_haiku: str,
               quality: float, context: Dict = None):
        """
        Train recursive selector from feedback (v1.1).

        Only trains if recursive selector is available.

        Args:
            candidates: Original candidate list
            selected_haiku: Haiku that was selected
            quality: Quality score (0-1)
            context: Context dict

        Returns:
            Loss (if trained), else None
        """
        if self.recursive_selector is not None:
            try:
                loss = self.recursive_selector.observe(
                    candidates, selected_haiku, quality, context
                )
                return loss
            except Exception as e:
                print(f"⚠️ Recursive training failed: {e}")
                return None
        return None
    
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
