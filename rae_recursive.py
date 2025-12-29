"""
Recursive RAE Selector - Micrograd-based

Inspired by Tiny Recursive Model (TRM) but using micrograd instead of PyTorch.
Uses recursive refinement to select best haiku from candidates.

Philosophy:
- Lightweight (uses existing micrograd MLP from haiku.py)
- Recursive refinement (3-5 iterations)
- No PyTorch dependency
- Trains online like MathBrain
"""

from typing import List, Tuple, Optional
import json
from pathlib import Path

# Import micrograd components from haiku.py
from haiku import MLP, Value


class RecursiveRAESelector:
    """
    Recursive haiku selector using micrograd MLP with refinement loop.

    Architecture:
    - Input: candidate features (perplexity, entropy, resonance, diversity, coherence)
    - Recursive refinement: 3-5 iterations
    - Output: selection scores for each candidate
    - Training: online learning from quality feedback
    """

    def __init__(
        self,
        feature_dim: int = 5,
        hidden_dim: int = 8,
        refinement_steps: int = 3,
        state_path: str = 'rae_brain.json'
    ):
        """
        Initialize recursive selector.

        Args:
            feature_dim: Input feature dimension (5: perplexity, entropy, resonance, diversity, coherence)
            hidden_dim: Hidden layer size
            refinement_steps: Number of recursive refinement steps (3-5)
            state_path: Path to save/load selector state
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.refinement_steps = refinement_steps
        self.state_path = state_path

        # MLP selector (same architecture as MathBrain in haiku.py)
        self.selector = MLP(feature_dim, [hidden_dim, 1])

        # Training state
        self.observations = 0
        self.learning_rate = 0.01

        # Try to load existing state
        if Path(state_path).exists():
            self.load_state()

    def extract_features(self, haiku: str, context: dict = None) -> List[float]:
        """
        Extract features from haiku candidate.

        Features (5D):
        1. perplexity: inverse word frequency
        2. entropy: word diversity
        3. resonance: overlap with context trigrams
        4. diversity: unique word ratio
        5. coherence: semantic flow (simple heuristic)

        Args:
            haiku: Haiku text (3 lines)
            context: Context dict with user_trigrams, etc.

        Returns:
            Feature vector [5 floats]
        """
        words = haiku.lower().split()
        unique_words = set(words)

        # 1. Perplexity (inverse word frequency heuristic)
        # Higher = more rare words
        perplexity = 0.5  # Default

        # 2. Entropy (word diversity)
        if len(words) > 0:
            entropy = len(unique_words) / len(words)
        else:
            entropy = 0.0

        # 3. Resonance (overlap with context)
        resonance = 0.0
        if context and 'user_trigrams' in context:
            user_words = set()
            for t in context['user_trigrams']:
                user_words.update(t)
            if user_words:
                overlap = len(unique_words & user_words)
                resonance = overlap / len(user_words)

        # 4. Diversity (unique word ratio)
        diversity = entropy  # Same as entropy for now

        # 5. Coherence (simple heuristic: line length variance)
        lines = haiku.split('\n')
        if len(lines) == 3:
            lengths = [len(line.split()) for line in lines]
            avg_len = sum(lengths) / 3
            variance = sum((l - avg_len) ** 2 for l in lengths) / 3
            coherence = 1.0 / (1.0 + variance)  # Lower variance = higher coherence
        else:
            coherence = 0.0

        return [perplexity, entropy, resonance, diversity, coherence]

    def select_recursive(
        self,
        candidates: List[str],
        context: dict = None
    ) -> Tuple[str, float]:
        """
        Select best haiku using recursive refinement.

        Process:
        1. Extract features for each candidate
        2. For each refinement step:
           a. Score all candidates
           b. Refine scores based on previous step
        3. Return candidate with highest final score

        Args:
            candidates: List of haiku strings
            context: Context dict with user info

        Returns:
            (best_haiku, confidence_score)
        """
        if not candidates:
            raise ValueError("No candidates provided")

        if len(candidates) == 1:
            return candidates[0], 1.0

        # Extract features
        features = [self.extract_features(h, context) for h in candidates]

        # Initial scores (pass 0)
        scores = []
        for feat in features:
            # Convert to Value objects for micrograd
            feat_values = [Value(f) for f in feat]
            score = self.selector(feat_values)
            scores.append(score.data)

        # Recursive refinement
        for step in range(self.refinement_steps):
            # Normalize scores
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                norm_scores = [0.5] * len(scores)

            # Refine: boost top scores, suppress low scores
            refined_scores = []
            for i, (feat, prev_score) in enumerate(zip(features, norm_scores)):
                # Add previous score as 6th feature (recursive feedback)
                feat_with_feedback = feat + [prev_score]
                feat_values = [Value(f) for f in feat_with_feedback[:5]]  # Keep 5D
                score = self.selector(feat_values)

                # Combine with previous score (refinement)
                refined = 0.7 * score.data + 0.3 * prev_score
                refined_scores.append(refined)

            scores = refined_scores

        # Select best
        best_idx = scores.index(max(scores))
        confidence = max(scores)

        return candidates[best_idx], confidence

    def observe(
        self,
        candidates: List[str],
        selected_haiku: str,
        quality: float,
        context: dict = None
    ) -> float:
        """
        Train selector from feedback (online learning).

        Process:
        1. Extract features for selected haiku
        2. Compute loss: (predicted_score - quality)^2
        3. Backward pass (micrograd autograd)
        4. Update weights with SGD

        Args:
            candidates: Original candidate list
            selected_haiku: Haiku that was selected
            quality: Quality score (0-1) from Harmonix metrics
            context: Context dict

        Returns:
            Loss value
        """
        # Extract features for selected haiku
        features = self.extract_features(selected_haiku, context)
        feat_values = [Value(f) for f in features]

        # Forward pass
        predicted_score = self.selector(feat_values)

        # Target
        target = Value(quality)

        # MSE loss
        loss = (predicted_score - target) ** 2

        # Backward pass
        for param in self.selector.parameters():
            param.grad = 0.0
        loss.backward()

        # SGD update
        for param in self.selector.parameters():
            param.data -= self.learning_rate * param.grad

            # Clamp weights to prevent explosion
            if param.data > 5.0:
                param.data = 5.0
            elif param.data < -5.0:
                param.data = -5.0

        self.observations += 1

        # Save state periodically
        if self.observations % 10 == 0:
            self.save_state()

        return loss.data

    def save_state(self):
        """Save selector state to JSON."""
        state = {
            'observations': self.observations,
            'weights': [p.data for p in self.selector.parameters()]
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        """Load selector state from JSON."""
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            self.observations = state.get('observations', 0)

            # Load weights
            weights = state.get('weights', [])
            params = self.selector.parameters()
            for param, weight in zip(params, weights):
                param.data = weight

            print(f"✓ Loaded recursive selector state: {self.observations} observations")
        except Exception as e:
            print(f"⚠️ Failed to load recursive selector state: {e}")
