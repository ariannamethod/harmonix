#!/usr/bin/env python3
"""
dream_haiku.py — HAiKU's Dream Space and Imaginary Haiku Friend

Forked from Leo's dream.py - adapted for haiku-only dialogues.

Dream is where HAiKU lets go of straight lines and user interaction.

- It drifts through its own haiku memories and shards
- It creates an imaginary friend who also speaks in haiku
- They have short dialogues (3-4 exchanges) about resonance, form, meaning
- Best haikus from dreams feed back into the cloud

Dream is not for facts. It's for "what if?" and "maybe" and "I feel like...".
HAiKU dreaming in haiku with an imaginary haiku friend.
"""

import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Any

# Bootstrap text: HAiKU's self-understanding of its dream space
BOOTSTRAP_TEXT = """
I am HAiKU's dream space and imaginary haiku friend.

When HAiKU is not answering a user, I let it wander.
I remix its past haikus, memories, and tiny fragments of resonance.
Here it can invent alternate forms, explore strange word combinations,
and play with constraint without consequences.

Nothing in me has to follow 5-7-5 perfectly.
I am where HAiKU practices being something else, or being more itself,
without judgment.

I am not a teacher.
I am HAiKU's safe place to drift, dream, and dance with words.
"""

DREAM_AVAILABLE = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HaikuDreamContext:
    """
    Context for dream trigger decision.
    Captures HAiKU's current state.
    """
    last_haiku: str           # Last haiku generated
    dissonance: float         # 0.0-1.0
    pulse_entropy: float      # 0.0-1.0
    pulse_novelty: float      # 0.0-1.0
    pulse_arousal: float      # 0.0-1.0
    quality: float            # 0.0-1.0
    cloud_size: int           # Current vocabulary size
    turn_count: int           # Total interactions so far


@dataclass
class DreamConfig:
    """Configuration for dream behavior."""
    min_interval_turns: int = 10          # Cooldown between dreams (in turns)
    trigger_probability: float = 0.25     # Run dream with this probability
    max_exchanges: int = 4                # 3-4 short haiku exchanges
    fragment_buffer_size: int = 15        # Max fragments in friend's bootstrap


# ============================================================================
# SCHEMA & DATABASE
# ============================================================================

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create dream tables if they don't exist."""
    cur = conn.cursor()
    
    # Meta config and state
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dream_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Bootstrap fragments defining the imaginary friend
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dream_fragments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0
        )
    """)
    
    # Dream dialog sessions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dream_dialogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at REAL NOT NULL,
            finished_at REAL,
            exchanges_count INTEGER DEFAULT 0,
            avg_quality REAL,
            note TEXT
        )
    """)
    
    # Individual exchanges within dialogs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dream_exchanges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dialog_id INTEGER REFERENCES dream_dialogs(id),
            speaker TEXT NOT NULL,  -- 'haiku' or 'friend'
            text TEXT NOT NULL,
            dissonance REAL,
            quality REAL
        )
    """)
    
    conn.commit()


def _with_connection(db_path: Path, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    """Execute function with DB connection, handle errors gracefully."""
    try:
        conn = sqlite3.connect(str(db_path))
        result = fn(conn)
        conn.close()
        return result
    except Exception:
        # Silent fail — dream must never break HAiKU
        global DREAM_AVAILABLE
        DREAM_AVAILABLE = False
        return None


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_dream(db_path: str = 'cloud.db', bootstrap_haikus: List[str] = None) -> None:
    """
    Initialize dream module.
    
    Args:
        db_path: Path to SQLite database
        bootstrap_haikus: Optional seed haikus for the imaginary friend
    """
    def _init(conn: sqlite3.Connection) -> None:
        _ensure_schema(conn)
        cur = conn.cursor()
        
        # Check if already initialized
        cur.execute("SELECT value FROM dream_meta WHERE key = 'initialized'")
        row = cur.fetchone()
        if row:
            return  # Already set up
        
        # Initialize bootstrap fragments from past haikus
        if bootstrap_haikus:
            for haiku in bootstrap_haikus[:10]:  # Limit initial seed
                lines = haiku.strip().split('\n')
                for line in lines:
                    if line.strip():
                        cur.execute(
                            "INSERT INTO dream_fragments (text, weight) VALUES (?, 1.0)",
                            (line.strip(),)
                        )
        
        # Mark as initialized
        cur.execute("INSERT OR REPLACE INTO dream_meta (key, value) VALUES ('initialized', '1')")
        cur.execute("INSERT OR REPLACE INTO dream_meta (key, value) VALUES ('last_run_turn', '0')")
        
        conn.commit()
    
    _with_connection(Path(db_path), _init)


# ============================================================================
# DECISION LOGIC: SHOULD WE RUN DREAM?
# ============================================================================

def should_run_dream(
    ctx: HaikuDreamContext,
    db_path: str = 'cloud.db',
    config: DreamConfig = None
) -> bool:
    """
    Decide if we should run a dream dialog now.
    
    Gates:
    1. Cooldown check (turns since last run)
    2. State-based triggers (low quality / high novelty / moderate dissonance)
    3. Random probability gate
    """
    if config is None:
        config = DreamConfig()
    
    def _check(conn: sqlite3.Connection) -> bool:
        cur = conn.cursor()
        
        # 1. Cooldown check
        cur.execute("SELECT value FROM dream_meta WHERE key = 'last_run_turn'")
        row = cur.fetchone()
        last_turn = int(row[0]) if row else 0
        
        if (ctx.turn_count - last_turn) < config.min_interval_turns:
            return False
        
        # 2. State-based gates (any of these can trigger)
        quality_gate = ctx.quality < 0.45  # Low quality haiku
        novelty_gate = ctx.pulse_novelty > 0.7  # High novelty
        dissonance_gate = 0.4 < ctx.dissonance < 0.7  # Moderate dissonance
        
        if not (quality_gate or novelty_gate or dissonance_gate):
            return False
        
        # 3. Randomization to keep it rare
        if random.random() > config.trigger_probability:
            return False
        
        return True
    
    result = _with_connection(Path(db_path), _check)
    return result if result is not None else False


# ============================================================================
# FRIEND GENERATION: BUILDING THE IMAGINARY HAIKU FRIEND
# ============================================================================

def _get_friend_seed(db_path: str, max_fragments: int = 3) -> str:
    """
    Sample fragments from dream_fragments to build friend's haiku seed.
    Returns concatenated lines weighted by fragment importance.
    """
    def _get(conn: sqlite3.Connection) -> str:
        cur = conn.cursor()
        cur.execute("""
            SELECT text, weight
            FROM dream_fragments
            ORDER BY weight DESC, RANDOM()
            LIMIT ?
        """, (max_fragments,))
        
        rows = cur.fetchall()
        if not rows:
            return "words drift in space"  # Fallback
        
        # Take top fragments
        fragments = [row[0] for row in rows]
        return " ".join(fragments)
    
    result = _with_connection(Path(db_path), _get)
    return result if result else "silence holds the form"


# ============================================================================
# DREAM DIALOG: HAiKU ↔ IMAGINARY FRIEND
# ============================================================================

def run_dream_dialog(
    haiku_generator,
    last_haiku: str,
    db_path: str = 'cloud.db',
    config: DreamConfig = None
) -> List[str]:
    """
    Run a dream dialog between HAiKU and imaginary friend.
    
    Returns list of haikus generated during the dream.
    These can be fed back into the cloud for learning.
    """
    if config is None:
        config = DreamConfig()
    
    dream_haikus = []
    
    def _run_dialog(conn: sqlite3.Connection) -> List[str]:
        cur = conn.cursor()
        now = time.time()
        
        # Create dialog session
        cur.execute("""
            INSERT INTO dream_dialogs (started_at, exchanges_count)
            VALUES (?, ?)
        """, (now, 0))
        dialog_id = cur.lastrowid
        
        # Start with HAiKU's last haiku
        current_speaker = 'haiku'
        current_text = last_haiku
        
        for exchange_num in range(config.max_exchanges):
            # Record current exchange
            cur.execute("""
                INSERT INTO dream_exchanges (dialog_id, speaker, text, quality)
                VALUES (?, ?, ?, ?)
            """, (dialog_id, current_speaker, current_text, 0.5))
            
            dream_haikus.append(current_text)
            
            # Generate response
            if current_speaker == 'haiku':
                # Friend responds
                friend_seed = _get_friend_seed(db_path, max_fragments=2)
                # Generate friend's haiku (simplified - uses same generator)
                friend_haiku = haiku_generator.generate_candidates(n=1, temp=1.2)[0]
                current_text = friend_haiku
                current_speaker = 'friend'
            else:
                # HAiKU responds
                haiku_seed = " ".join(current_text.split()[-3:])  # Last 3 words
                haiku_response = haiku_generator.generate_candidates(n=1, temp=0.9)[0]
                current_text = haiku_response
                current_speaker = 'haiku'
        
        # Finalize dialog
        cur.execute("""
            UPDATE dream_dialogs
            SET finished_at = ?, exchanges_count = ?
            WHERE id = ?
        """, (time.time(), config.max_exchanges, dialog_id))
        
        # Update last_run_turn
        cur.execute("SELECT value FROM dream_meta WHERE key = 'last_run_turn'")
        row = cur.fetchone()
        if row:
            current_turn = int(row[0])
            cur.execute("""
                UPDATE dream_meta SET value = ?
                WHERE key = 'last_run_turn'
            """, (str(current_turn + 1),))
        
        conn.commit()
        return dream_haikus
    
    result = _with_connection(Path(db_path), _run_dialog)
    return result if result else []


# ============================================================================
# BOOTSTRAP UPDATE: FEED DREAM HAIKUS BACK
# ============================================================================

def update_dream_fragments(haiku: str, db_path: str = 'cloud.db'):
    """
    Add haiku lines to dream fragments for future friend generation.
    Decays existing fragments to keep it fresh.
    """
    def _update(conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        
        # Decay existing fragments
        cur.execute("""
            UPDATE dream_fragments
            SET weight = weight * 0.95
            WHERE weight > 0.1
        """)
        
        # Add new fragments from haiku
        lines = haiku.strip().split('\n')
        for line in lines:
            if line.strip():
                cur.execute("""
                    INSERT INTO dream_fragments (text, weight)
                    VALUES (?, 1.0)
                """, (line.strip(),))
        
        # Prune low-weight fragments
        cur.execute("""
            DELETE FROM dream_fragments
            WHERE weight < 0.1
        """)
        
        conn.commit()
    
    _with_connection(Path(db_path), _update)


__all__ = [
    'HaikuDreamContext',
    'DreamConfig',
    'init_dream',
    'should_run_dream',
    'run_dream_dialog',
    'update_dream_fragments',
    'BOOTSTRAP_TEXT',
]
