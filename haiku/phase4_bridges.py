#!/usr/bin/env python3
"""
phase4_bridges.py — HAiKU's Island Bridges

Forked from Leo's mathbrain_phase4.py - adapted for haiku states.

Phase 4 for HAiKU:
- Learns bridges between haiku states (experience units) from interaction trajectories
- Allows fuzzy similarity between states (not exact match)
- Suggests next directions as "climate flows", not rigid choices

Philosophy:
- Not a planner. Memory of where the field usually flows.
- "When dissonance felt like this and pulse was here, haiku often flowed there."
- Fuzzy matches with explicit thresholds (0.6-0.8)
- Respect boredom/overwhelm/stuck signals

Islands in HAiKU = Haiku States (combinations of dissonance, pulse metrics, quality)
"""

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional numpy for precision
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

MetricVector = Dict[str, float]


@dataclass
class HaikuStateTransition:
    """Statistics for a single haiku state → state transition."""
    from_state: str
    to_state: str
    count: int
    avg_similarity: float
    avg_quality_delta: float
    overwhelm_rate: float
    boredom_rate: float
    stuck_rate: float
    
    @property
    def score(self) -> float:
        """
        Composite score for ranking transitions.
        
        Higher similarity = field flows naturally
        Positive quality_delta = haiku gets better
        High overwhelm/boredom/stuck = penalties
        """
        # Base: similarity * (1 + quality boost)
        quality_factor = 1.0 + max(-0.5, min(0.5, self.avg_quality_delta))
        base = self.avg_similarity * quality_factor
        
        # Penalties
        overwhelm_penalty = 1.0 - min(1.0, max(0.0, self.overwhelm_rate))
        boredom_penalty = 1.0 - 0.3 * min(1.0, max(0.0, self.boredom_rate))
        stuck_penalty = 1.0 - 0.2 * min(1.0, max(0.0, self.stuck_rate))
        
        return base * overwhelm_penalty * boredom_penalty * stuck_penalty


# ============================================================================
# MATH HELPERS
# ============================================================================

def cosine_similarity(a: MetricVector, b: MetricVector) -> float:
    """Cosine similarity over shared metric keys."""
    if not a or not b:
        return 0.0
    
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    
    if NUMPY_AVAILABLE:
        vec_a = np.array([a[k] for k in keys])
        vec_b = np.array([b[k] for k in keys])
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    else:
        dot = sum(a[k] * b[k] for k in keys)
        norm_a = math.sqrt(sum(a[k] * a[k] for k in keys))
        norm_b = math.sqrt(sum(b[k] * b[k] for k in keys))
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        return dot / (norm_a * norm_b)


def get_quality(metric_vec: MetricVector) -> float:
    """Extract quality from metrics, default 0.5."""
    return float(metric_vec.get("quality", 0.5))


# ============================================================================
# PHASE 4 CORE
# ============================================================================

class HaikuBridges:
    """
    Phase 4: Haiku state bridges and flows.
    
    Learns from interaction trajectories which states follow which,
    under what metric conditions, and with what outcomes.
    
    Usage:
        bridges = HaikuBridges('cloud.db')
        
        # After each haiku:
        bridges.record_state(
            state_id="d0.7_e0.5_q0.6",
            metrics_before=before,
            metrics_after=after,
            prev_state_id="d0.3_e0.6_q0.5",
            turn_id="turn_5",
            boredom=False,
            overwhelm=False,
            stuck=False,
        )
        
        # To get suggestions:
        flows = bridges.suggest_next_states("d0.7_e0.5_q0.6")
    """
    
    def __init__(self, db_path: str = 'state/cloud.db'):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create Phase 4 tables if they don't exist."""
        cur = self.conn.cursor()
        
        # Log of individual state activations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS haiku_state_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                turn_id TEXT,
                prev_state_id TEXT,
                state_id TEXT NOT NULL,
                metrics_before TEXT,
                metrics_after TEXT,
                boredom INTEGER DEFAULT 0,
                overwhelm INTEGER DEFAULT 0,
                stuck INTEGER DEFAULT 0
            );
        """)
        
        # Aggregate transitions between states
        cur.execute("""
            CREATE TABLE IF NOT EXISTS haiku_transitions (
                from_state_id TEXT NOT NULL,
                to_state_id TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                sum_similarity REAL NOT NULL DEFAULT 0.0,
                sum_quality_delta REAL NOT NULL DEFAULT 0.0,
                sum_overwhelm REAL NOT NULL DEFAULT 0.0,
                sum_boredom REAL NOT NULL DEFAULT 0.0,
                sum_stuck REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (from_state_id, to_state_id)
            );
        """)
        
        # State snapshots
        cur.execute("""
            CREATE TABLE IF NOT EXISTS haiku_state_snapshot (
                state_id TEXT PRIMARY KEY,
                last_metrics TEXT,
                activations_total INTEGER NOT NULL DEFAULT 0,
                last_updated REAL
            );
        """)
        
        # Index for faster queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_haiku_transitions_from
            ON haiku_transitions(from_state_id);
        """)
        
        self.conn.commit()
    
    def record_state(
        self,
        state_id: str,
        metrics_before: MetricVector,
        metrics_after: MetricVector,
        *,
        prev_state_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        boredom: bool = False,
        overwhelm: bool = False,
        stuck: bool = False,
    ) -> None:
        """
        Main entry point from HAiKU core.
        
        Call this once per turn after haiku generation.
        """
        ts = time.time()
        cur = self.conn.cursor()
        
        # Log activation
        cur.execute("""
            INSERT INTO haiku_state_log (
                ts, turn_id, prev_state_id, state_id,
                metrics_before, metrics_after,
                boredom, overwhelm, stuck
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, turn_id, prev_state_id, state_id,
            json.dumps(metrics_before), json.dumps(metrics_after),
            int(boredom), int(overwhelm), int(stuck)
        ))
        
        # Update state snapshot
        cur.execute("""
            INSERT INTO haiku_state_snapshot (state_id, last_metrics, activations_total, last_updated)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                last_metrics = excluded.last_metrics,
                activations_total = haiku_state_snapshot.activations_total + 1,
                last_updated = excluded.last_updated;
        """, (state_id, json.dumps(metrics_after), ts))
        
        # If we have a previous state, record transition
        if prev_state_id:
            # Compute similarity
            similarity = cosine_similarity(metrics_before, metrics_after)
            
            # Compute quality delta
            quality_before = get_quality(metrics_before)
            quality_after = get_quality(metrics_after)
            quality_delta = quality_after - quality_before
            
            # Update transition aggregate
            cur.execute("""
                INSERT INTO haiku_transitions (
                    from_state_id, to_state_id, count,
                    sum_similarity, sum_quality_delta,
                    sum_overwhelm, sum_boredom, sum_stuck
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(from_state_id, to_state_id) DO UPDATE SET
                    count = haiku_transitions.count + 1,
                    sum_similarity = haiku_transitions.sum_similarity + excluded.sum_similarity,
                    sum_quality_delta = haiku_transitions.sum_quality_delta + excluded.sum_quality_delta,
                    sum_overwhelm = haiku_transitions.sum_overwhelm + excluded.sum_overwhelm,
                    sum_boredom = haiku_transitions.sum_boredom + excluded.sum_boredom,
                    sum_stuck = haiku_transitions.sum_stuck + excluded.sum_stuck;
            """, (
                prev_state_id, state_id, similarity, quality_delta,
                float(overwhelm), float(boredom), float(stuck)
            ))
        
        self.conn.commit()
    
    def suggest_next_states(
        self,
        current_state_id: str,
        min_count: int = 2,
        max_results: int = 5
    ) -> List[HaikuStateTransition]:
        """
        Suggest next haiku states based on learned transitions.
        
        Returns top N transitions ranked by composite score.
        """
        cur = self.conn.cursor()
        
        rows = cur.execute("""
            SELECT
                from_state_id,
                to_state_id,
                count,
                sum_similarity / count AS avg_similarity,
                sum_quality_delta / count AS avg_quality_delta,
                sum_overwhelm / count AS overwhelm_rate,
                sum_boredom / count AS boredom_rate,
                sum_stuck / count AS stuck_rate
            FROM haiku_transitions
            WHERE from_state_id = ? AND count >= ?
            ORDER BY count DESC
            LIMIT ?
        """, (current_state_id, min_count, max_results * 2)).fetchall()
        
        if not rows:
            return []
        
        # Build transition objects
        transitions = []
        for row in rows:
            t = HaikuStateTransition(
                from_state=row[0],
                to_state=row[1],
                count=row[2],
                avg_similarity=row[3],
                avg_quality_delta=row[4],
                overwhelm_rate=row[5],
                boredom_rate=row[6],
                stuck_rate=row[7]
            )
            transitions.append(t)
        
        # Sort by composite score
        transitions.sort(key=lambda t: t.score, reverse=True)
        
        return transitions[:max_results]
    
    def get_state_history(self, state_id: str, limit: int = 10) -> List[Dict]:
        """Get recent activations for a state."""
        cur = self.conn.cursor()
        
        rows = cur.execute("""
            SELECT ts, turn_id, metrics_before, metrics_after, boredom, overwhelm, stuck
            FROM haiku_state_log
            WHERE state_id = ?
            ORDER BY ts DESC
            LIMIT ?
        """, (state_id, limit)).fetchall()
        
        return [
            {
                'ts': row[0],
                'turn_id': row[1],
                'metrics_before': json.loads(row[2]) if row[2] else {},
                'metrics_after': json.loads(row[3]) if row[3] else {},
                'boredom': bool(row[4]),
                'overwhelm': bool(row[5]),
                'stuck': bool(row[6]),
            }
            for row in rows
        ]
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def state_id_from_metrics(dissonance: float, entropy: float, quality: float) -> str:
    """Generate state ID from key metrics."""
    return f"d{dissonance:.1f}_e{entropy:.1f}_q{quality:.1f}"


__all__ = [
    'HaikuBridges',
    'HaikuStateTransition',
    'state_id_from_metrics',
    'cosine_similarity',
]
