"""
Tests for rae_recursive.py - Recursive RAE Selector

Tests:
- Feature extraction
- Recursive selection
- Online learning (observe)
- State persistence
"""

import pytest
from pathlib import Path
from rae_recursive import RecursiveRAESelector


class TestRecursiveRAESelector:
    """Test RecursiveRAESelector class."""

    @pytest.fixture
    def selector(self, tmp_path):
        state_path = tmp_path / "test_rae_brain.json"
        return RecursiveRAESelector(state_path=str(state_path))

    def test_selector_init(self, selector):
        """Test selector initialization."""
        assert selector.feature_dim == 5
        assert selector.hidden_dim == 8
        assert selector.refinement_steps == 3
        assert selector.observations == 0

    def test_extract_features(self, selector):
        """Test feature extraction from haiku."""
        haiku = "words dance in cloud\nresonance finds its own path\nconstraint births form"
        features = selector.extract_features(haiku)

        # Should return 5 features
        assert len(features) == 5

        # All features should be floats in [0, 1]
        assert all(isinstance(f, float) for f in features)
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_extract_features_with_context(self, selector):
        """Test feature extraction with context."""
        haiku = "what is resonance\nin the cloud space\nwords dance here"
        context = {
            'user': 'what is resonance',
            'user_trigrams': [('what', 'is', 'resonance')]
        }

        features = selector.extract_features(haiku, context)

        # Should have resonance > 0 (overlap with context)
        resonance_idx = 2
        assert features[resonance_idx] > 0.0

    def test_select_recursive_single_candidate(self, selector):
        """Test recursive selection with single candidate."""
        candidates = ["one\ntwo\nthree"]

        best, confidence = selector.select_recursive(candidates)

        assert best == candidates[0]
        assert confidence == 1.0

    def test_select_recursive_multiple_candidates(self, selector):
        """Test recursive selection with multiple candidates."""
        candidates = [
            "first\nhaiku\nhere",
            "second\npoem\ntext",
            "third\nverse\nwords"
        ]

        best, confidence = selector.select_recursive(candidates)

        # Should select one of the candidates
        assert best in candidates
        # Confidence is float (may be negative before training due to random init)
        assert isinstance(confidence, float)

    def test_select_recursive_with_context(self, selector):
        """Test recursive selection considers context."""
        candidates = [
            "words dance in cloud\nresonance finds path\nmeaning emerges",
            "random unrelated text\nno connection here\njust words"
        ]
        context = {
            'user': 'what is resonance',
            'user_trigrams': [('what', 'is', 'resonance'), ('is', 'resonance', 'in')]
        }

        best, confidence = selector.select_recursive(candidates, context)

        # Should prefer haiku with resonance overlap
        # (Not guaranteed but likely after training)
        assert best in candidates

    def test_observe_trains_selector(self, selector):
        """Test observe() updates selector weights."""
        candidates = ["test\nhaiku\nhere"]
        selected = candidates[0]
        quality = 0.7

        initial_obs = selector.observations

        loss = selector.observe(candidates, selected, quality)

        # Should increment observations
        assert selector.observations == initial_obs + 1

        # Should return loss value
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_observe_multiple_times(self, selector):
        """Test selector learns from multiple observations."""
        candidates = ["good\nhaiku\ntext", "bad\nlow\nquality"]

        # Train on high quality
        for _ in range(5):
            selector.observe(candidates, candidates[0], quality=0.9)

        # Train on low quality
        for _ in range(5):
            selector.observe(candidates, candidates[1], quality=0.2)

        # Should have 10 observations
        assert selector.observations == 10

    def test_save_load_state(self, selector, tmp_path):
        """Test state persistence."""
        # Train selector
        candidates = ["test\nhaiku\nhere"]
        for i in range(15):  # Trigger save at 10
            selector.observe(candidates, candidates[0], quality=0.7)

        # Save state
        selector.save_state()

        # Check file exists
        assert Path(selector.state_path).exists()

        # Create new selector and load
        new_selector = RecursiveRAESelector(state_path=selector.state_path)

        # Should have same observations
        assert new_selector.observations == selector.observations

    def test_refinement_iterations(self, selector):
        """Test recursive refinement runs multiple steps."""
        # Set refinement steps
        selector.refinement_steps = 5

        candidates = [
            "first\nhaiku\nhere",
            "second\npoem\ntext"
        ]

        # Should complete without error
        best, confidence = selector.select_recursive(candidates)
        assert best in candidates


class TestRAEIntegration:
    """Test RAE integration with recursive selector."""

    def test_rae_with_recursive_selector(self, tmp_path):
        """Test RAE uses recursive selector when available."""
        from rae import RecursiveAdapterEngine

        rae = RecursiveAdapterEngine(use_recursive=True)

        # Should have recursive selector
        assert rae.recursive_selector is not None

        # Test selection
        context = {'user': 'test', 'user_trigrams': []}
        candidates = ["one\ntwo\nthree", "four\nfive\nsix"]

        result = rae.reason(context, candidates)
        assert result in candidates

    def test_rae_observe(self, tmp_path):
        """Test RAE observe() trains recursive selector."""
        from rae import RecursiveAdapterEngine

        rae = RecursiveAdapterEngine(use_recursive=True)

        if rae.recursive_selector is not None:
            candidates = ["test\nhaiku\nhere"]
            selected = candidates[0]
            quality = 0.7

            loss = rae.observe(candidates, selected, quality, context={})

            # Should return loss (or None if selector unavailable)
            assert loss is None or isinstance(loss, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
