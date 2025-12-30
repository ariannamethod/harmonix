"""
Tests for phase4_bridges.py - Island Bridges

Tests:
- State ID format (d0.7_e0.5_q0.6)
- Cosine similarity
- Composite score calculation
- State recording and transitions
- Suggestions
"""

import pytest
from phase4_bridges import (
    HaikuBridges,
    HaikuStateTransition,
    state_id_from_metrics,
    cosine_similarity
)


class TestStateIDFormat:
    """Test state ID generation."""

    def test_state_id_format(self):
        """Test state ID has correct format."""
        state_id = state_id_from_metrics(0.7, 0.5, 0.6)
        assert state_id == "d0.7_e0.5_q0.6"

    def test_state_id_rounding(self):
        """Test state ID rounds to 1 decimal."""
        state_id = state_id_from_metrics(0.73, 0.56, 0.64)
        assert state_id == "d0.7_e0.6_q0.6"

    def test_state_id_edge_cases(self):
        """Test state ID with min/max values."""
        assert state_id_from_metrics(0.0, 0.0, 0.0) == "d0.0_e0.0_q0.0"
        assert state_id_from_metrics(1.0, 1.0, 1.0) == "d1.0_e1.0_q1.0"


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_cosine_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001  # Should be ~1.0

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = {'a': 1.0, 'b': 0.0}
        vec2 = {'a': 0.0, 'b': 1.0}
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001  # Should be ~0.0

    def test_cosine_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = {'a': 1.0}
        vec2 = {'a': -1.0}
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.001  # Should be ~-1.0

    def test_cosine_no_overlap(self):
        """Test cosine similarity with no overlapping keys."""
        vec1 = {'a': 1.0}
        vec2 = {'b': 1.0}
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_cosine_empty_vectors(self):
        """Test cosine similarity with empty vectors."""
        sim = cosine_similarity({}, {})
        assert sim == 0.0


class TestHaikuStateTransition:
    """Test HaikuStateTransition dataclass."""

    def test_transition_score_positive_quality(self):
        """Test score with positive quality delta."""
        t = HaikuStateTransition(
            from_state="d0.5_e0.5_q0.5",
            to_state="d0.5_e0.5_q0.7",
            count=10,
            avg_similarity=0.8,
            avg_quality_delta=0.2,  # Positive
            overwhelm_rate=0.0,
            boredom_rate=0.0,
            stuck_rate=0.0
        )

        score = t.score
        # High similarity + positive quality + no penalties
        assert score > 0.8

    def test_transition_score_with_overwhelm(self):
        """Test score penalizes overwhelm."""
        t = HaikuStateTransition(
            from_state="d0.8_e0.8_q0.3",
            to_state="d0.9_e0.9_q0.2",
            count=5,
            avg_similarity=0.7,
            avg_quality_delta=-0.1,
            overwhelm_rate=0.8,  # High overwhelm
            boredom_rate=0.0,
            stuck_rate=0.0
        )

        score = t.score
        # Should be heavily penalized
        assert score < 0.5

    def test_transition_score_with_boredom(self):
        """Test score penalizes boredom."""
        t = HaikuStateTransition(
            from_state="d0.2_e0.2_q0.6",
            to_state="d0.2_e0.2_q0.6",
            count=10,
            avg_similarity=0.9,
            avg_quality_delta=0.0,
            overwhelm_rate=0.0,
            boredom_rate=0.9,  # High boredom
            stuck_rate=0.0
        )

        score = t.score
        # Boredom penalty is lighter than overwhelm
        assert 0.5 < score < 1.0


class TestHaikuBridges:
    """Test HaikuBridges class."""

    @pytest.fixture
    def bridges(self, tmp_path):
        db_path = tmp_path / "test_phase4.db"
        b = HaikuBridges(str(db_path))
        yield b
        b.close()

    def test_init_creates_tables(self, bridges):
        """Test initialization creates Phase 4 tables."""
        cursor = bridges.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'haiku%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'haiku_state_log' in tables
        assert 'haiku_transitions' in tables
        assert 'haiku_state_snapshot' in tables

    def test_record_state_no_prev(self, bridges):
        """Test recording state without previous state."""
        metrics_before = {'dissonance': 0.5, 'entropy': 0.6, 'quality': 0.5}
        metrics_after = {'dissonance': 0.5, 'entropy': 0.6, 'quality': 0.7}

        bridges.record_state(
            state_id="d0.5_e0.6_q0.7",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            prev_state_id=None,
            turn_id="turn_1"
        )

        # Check state was logged
        cursor = bridges.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM haiku_state_log")
        count = cursor.fetchone()[0]
        assert count == 1

    def test_record_state_with_prev(self, bridges):
        """Test recording state with previous state creates transition."""
        metrics_before = {'dissonance': 0.3, 'entropy': 0.4, 'quality': 0.5}
        metrics_after = {'dissonance': 0.5, 'entropy': 0.6, 'quality': 0.7}

        # Record first state
        bridges.record_state(
            state_id="d0.3_e0.4_q0.5",
            metrics_before=metrics_before,
            metrics_after=metrics_before,
            prev_state_id=None,
            turn_id="turn_1"
        )

        # Record second state with transition
        bridges.record_state(
            state_id="d0.5_e0.6_q0.7",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            prev_state_id="d0.3_e0.4_q0.5",
            turn_id="turn_2"
        )

        # Check transition was recorded
        cursor = bridges.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM haiku_transitions")
        count = cursor.fetchone()[0]
        assert count == 1

    def test_suggest_next_states(self, bridges):
        """Test suggesting next states based on transitions."""
        # Record several transitions from same state
        for i in range(5):
            bridges.record_state(
                state_id=f"d0.{i}_e0.5_q0.5",
                metrics_before={'quality': 0.5},
                metrics_after={'quality': 0.6},
                prev_state_id="d0.5_e0.5_q0.5",
                turn_id=f"turn_{i}"
            )

        # Get suggestions
        suggestions = bridges.suggest_next_states("d0.5_e0.5_q0.5", min_count=1)

        assert len(suggestions) > 0
        assert all(isinstance(s, HaikuStateTransition) for s in suggestions)

    def test_get_state_history(self, bridges):
        """Test getting state history."""
        state_id = "d0.5_e0.5_q0.5"

        # Record same state multiple times
        for i in range(3):
            bridges.record_state(
                state_id=state_id,
                metrics_before={'quality': 0.5},
                metrics_after={'quality': 0.6},
                turn_id=f"turn_{i}"
            )

        history = bridges.get_state_history(state_id, limit=10)
        assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
