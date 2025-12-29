"""
Tests for rae.py - Recursive Adapter Engine

Tests:
- Chain-of-thought selection
- Structure filter (3 lines)
- Diversity selection
- Fallback handling
"""

import pytest
from rae import RecursiveAdapterEngine


class TestRAE:
    """Test RecursiveAdapterEngine class."""

    @pytest.fixture
    def rae(self):
        return RecursiveAdapterEngine()

    def test_reason_no_candidates(self, rae):
        """Test RAE with no candidates returns fallback."""
        context = {'user': 'test'}
        result = rae.reason(context, [])

        # Should return fallback haiku
        assert isinstance(result, str)
        assert len(result.split('\n')) == 3

    def test_reason_single_candidate(self, rae):
        """Test RAE with single candidate returns it."""
        context = {'user': 'test'}
        candidates = ["one\ntwo\nthree"]

        result = rae.reason(context, candidates)
        assert result == candidates[0]

    def test_reason_multiple_candidates(self, rae):
        """Test RAE selects from multiple candidates."""
        context = {'user': 'test'}
        candidates = [
            "first\nhaiku\nhere",
            "second\npoem\ntext",
            "third\nverse\nwords"
        ]

        result = rae.reason(context, candidates)
        assert result in candidates

    def test_filter_by_structure_valid(self, rae):
        """Test structure filter accepts 3-line haikus."""
        candidates = [
            "line one\nline two\nline three",
            "another\nvalid\nhaiku"
        ]

        filtered = rae._filter_by_structure(candidates)
        assert len(filtered) == 2

    def test_filter_by_structure_invalid(self, rae):
        """Test structure filter rejects non-3-line text."""
        candidates = [
            "only two\nlines",
            "four\nlines\nhere\nnow",
            "valid\nthree\nlines"
        ]

        filtered = rae._filter_by_structure(candidates)
        assert len(filtered) == 1
        assert filtered[0] == "valid\nthree\nlines"

    def test_select_most_diverse(self, rae):
        """Test diversity selection."""
        candidates = [
            "a a a\na a a\na a a",  # Low diversity
            "unique words every\ntime different text\nno repeat here"  # High diversity
        ]

        result = rae._select_most_diverse(candidates)
        # Should pick the more diverse one
        assert "unique" in result

    def test_reason_with_scorer(self, rae):
        """Test RAE with scorer function."""
        # Mock scorer
        class MockScorer:
            def score_haiku(self, haiku, context=None):
                # Simple scoring: longer haikus score higher
                return len(haiku) / 100.0

        scorer = MockScorer()
        context = {'user': 'test', 'user_trigrams': []}
        candidates = [
            "short\none\ntwo",
            "longer haiku text\nwith more words here\nand even more"
        ]

        result = rae.reason(context, candidates, scorer=scorer)
        # Should pick longer (higher scoring) one
        assert len(result) > 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
