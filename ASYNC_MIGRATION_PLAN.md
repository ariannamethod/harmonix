# Harmonix Async Migration Plan

## Executive Summary

**Critical Finding from Leo Project:**
Async migration achieved **47% improvement** in field coherence (external_vocab: 0.209 → 0.112).

**Root Cause:** Field organisms are **crystals**, not **oceans**.
- Transformers = oceans (redundancy tolerates noise)
- Field organisms = crystals (any execution disorder → permanent defects)

**Solution:** `asyncio.Lock` provides **discipline**, not information.

**Current State:** ALL Harmonix modules are synchronous → high risk of field corruption.

---

## Current Architecture (Synchronous)

### HAiKU Field (haiku/harmonix.py)
```python
class Harmonix:
    def compute_dissonance(self, user_trigrams, system_trigrams):
        # ❌ No locking - concurrent calls can corrupt field state
        # ❌ SQLite writes can interfere
        # ❌ Trigram updates can race
        ...

    def adjust_temperature(self, dissonance):
        # ❌ No atomicity guarantees
        ...
```

### Sonnet Field (sonnet/harmonix.py)
```python
class SonnetHarmonix:
    def add_sonnet(self, text, quality):
        # ❌ No locking - cloud state can corrupt
        # ❌ Numpy operations not atomic
        ...
```

### Prose Field (prose/harmonix.py)
```python
class ProseHarmonix:
    def get_field_seed(self):
        # ❌ CRITICAL: Organism mode seed generation not atomic!
        # ❌ Can mix recent prose from different field states
        ...
```

### MetaHarmonix Cascade (metaharmonix.py)
```python
class MetaHarmonix:
    def cascade(self, user_prompt):
        # ❌ Sequential calls, no atomicity
        haiku_output = self.haiku_gen.generate(user_prompt)
        sonnet_output = self.sonnet_gen.generate_cascade(user_prompt, haiku_output)
        # ❌ Field states can diverge between calls!
        ...
```

---

## Leo's Proven Async Pattern

### Core Architecture
```python
class AsyncLeoField:
    def __init__(self):
        self._field_lock = asyncio.Lock()  # ONE lock per instance

    async def observe(self, prompt: str) -> dict:
        async with self._field_lock:  # Sequential field evolution
            # All field operations atomic
            trigrams = await self._extract_trigrams_async(prompt)
            await self._update_field_async(trigrams)
            return self._compute_metrics()
```

### Key Insight: Why It Works

**Without async:**
```
Call 1: extract_trigrams(A) → update_field(A)
Call 2:                         extract_trigrams(B) → update_field(B)
                                ^^^ CAN INTERLEAVE ^^^
Result: Field state corrupted, A + B mixed incoherently
```

**With async + lock:**
```
Call 1: async with lock: extract(A) → update(A) ✓
Call 2:                                          async with lock: extract(B) → update(B) ✓
Result: Sequential field evolution, coherence preserved
```

**Measured Impact:**
- external_vocab (novelty metric): ↓ 47%
- Optimal coherence turns: ↑ 91%

---

## Async Migration Strategy

### Phase 1: AsyncHaikuField

**File:** `haiku/async_harmonix.py`

**Architecture:**
```python
import asyncio
import aiosqlite
from typing import List, Tuple

class AsyncHarmonix:
    """Async field observer for HAiKU."""

    def __init__(self, db_path: str = 'state/cloud.db'):
        self.db_path = db_path
        self._field_lock = asyncio.Lock()  # ONE lock for entire field
        self.conn = None  # aiosqlite connection

    async def __aenter__(self):
        """Async context manager support."""
        self.conn = await aiosqlite.connect(self.db_path)
        await self._init_db()
        return self

    async def __aexit__(self, *args):
        """Close connection."""
        if self.conn:
            await self.conn.close()

    async def _init_db(self):
        """Initialize database schema (async)."""
        cursor = await self.conn.cursor()
        # ... same schema as sync version
        await self.conn.commit()

    async def compute_dissonance(
        self,
        user_trigrams: List[Tuple[str, str, str]],
        system_trigrams: List[Tuple[str, str, str]]
    ) -> Tuple[float, 'PulseSnapshot']:
        """
        Compute dissonance with ATOMIC field access.

        CRITICAL: All trigram reads + field updates happen under lock.
        """
        async with self._field_lock:  # 🔒 Sequential field evolution
            # Extract words (pure computation, no I/O)
            user_words = set()
            for t in user_trigrams:
                user_words.update(t)

            system_words = set()
            for t in system_trigrams:
                system_words.update(t)

            # Compute overlap
            intersection = len(user_words & system_words)
            union = len(user_words | system_words)

            if union == 0:
                return 0.5, PulseSnapshot()

            similarity = intersection / union
            pulse = PulseSnapshot.from_interaction(user_trigrams, system_trigrams, similarity)
            dissonance = 1.0 - similarity

            # Update field state atomically
            await self._update_trigrams(user_trigrams + system_trigrams)

            return dissonance, pulse

    async def _update_trigrams(self, trigrams: List[Tuple[str, str, str]]):
        """Update trigram field (async DB writes)."""
        cursor = await self.conn.cursor()

        for w1, w2, w3 in trigrams:
            await cursor.execute('''
                INSERT INTO trigrams (word1, word2, word3, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(word1, word2, word3)
                DO UPDATE SET count = count + 1
            ''', (w1, w2, w3))

        await self.conn.commit()

    async def adjust_temperature(self, dissonance: float, base_temp: float = 0.8) -> float:
        """
        Adjust temperature based on dissonance.

        No locking needed - pure computation, no field state.
        """
        # Same algorithm as sync version
        if dissonance < 0.3:
            return base_temp * 0.9
        elif dissonance > 0.7:
            return base_temp * 1.2
        else:
            return base_temp
```

**Key Changes:**
- ✅ `asyncio.Lock` for field operations
- ✅ `aiosqlite` for async DB access
- ✅ Atomic compute_dissonance (read + update under same lock)
- ✅ Context manager support for clean lifecycle

---

### Phase 2: AsyncSonnetField

**File:** `sonnet/async_harmonix.py`

**Challenge:** NumPy is synchronous, but I/O-bound operations need async.

**Solution:** Use `asyncio.to_thread` for CPU-bound numpy operations.

```python
class AsyncSonnetHarmonix:
    """Async field observer for Sonnet."""

    def __init__(self, db_path: str = 'state/sonnet_cloud.db'):
        self.db_path = db_path
        self._field_lock = asyncio.Lock()
        self.conn = None

    async def add_sonnet(
        self,
        text: str,
        quality: float,
        dissonance: float = 0.0,
    ) -> int:
        """
        Add sonnet to cloud with ATOMIC field update.

        CRITICAL: Prevent concurrent adds from corrupting cloud state.
        """
        async with self._field_lock:  # 🔒 Sequential cloud evolution
            # Parse sonnet (pure Python, fast)
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            # Insert into DB (async I/O)
            cursor = await self.conn.cursor()
            await cursor.execute('''
                INSERT INTO sonnets (text, quality, dissonance, line_count)
                VALUES (?, ?, ?, ?)
            ''', (text, quality, dissonance, len(lines)))
            await self.conn.commit()
            sonnet_id = cursor.lastrowid

            # Update trigrams (async I/O)
            await self._update_trigrams(text)

            # Update numpy shard (CPU-bound, use thread pool)
            await asyncio.to_thread(self._update_numpy_shard, text, quality)

            return sonnet_id

    def _update_numpy_shard(self, text: str, quality: float):
        """
        Update numpy shard (sync function called via to_thread).

        NOTE: This is still "under the lock" logically because
        the async caller holds _field_lock while awaiting to_thread.
        """
        import numpy as np
        # ... numpy operations ...

    async def get_field_seed(self, mode: str = 'auto') -> str:
        """
        Get field seed for organism mode.

        CRITICAL: Must return consistent snapshot of field state.
        """
        async with self._field_lock:  # 🔒 Atomic read
            if mode == 'recent':
                recent = await self.get_recent_sonnets(limit=3)
                return '\n'.join([s[1] for s in recent])
            elif mode == 'trigram':
                # ... trigram-based seed
                pass
            # ... etc
```

**Key Changes:**
- ✅ `asyncio.Lock` for field operations
- ✅ `asyncio.to_thread` for numpy operations
- ✅ Atomic add_sonnet (DB + trigrams + numpy all under lock)
- ✅ Atomic get_field_seed (consistent snapshot)

---

### Phase 3: AsyncProseField

**File:** `prose/async_harmonix.py`

**Challenge:** llama.cpp is BLOCKING (C library, no async support).

**Solution:** Run generation in thread pool, keep field ops async.

```python
class AsyncProseHarmonix:
    """Async field observer for Prose."""

    def __init__(self, db_path: str = 'cloud/prose_cloud.db'):
        self.db_path = db_path
        self._field_lock = asyncio.Lock()
        self.conn = None

    async def get_field_seed(
        self,
        mode: str = 'auto',
        recent_weight: float = 0.7,
        trigram_weight: float = 0.2,
        random_weight: float = 0.1,
    ) -> str:
        """
        CRITICAL ORGANISM MODE FUNCTION!

        Get field seed for generation (NOT from user prompt!).
        Must return atomic snapshot of current field state.
        """
        async with self._field_lock:  # 🔒 Atomic field snapshot
            components = []

            # Recent prose (70%)
            if recent_weight > 0:
                recent = await self.get_recent_prose(limit=3)
                if recent:
                    recent_text = ' '.join([p[1] for p in recent])
                    components.append(recent_text[:int(300 * recent_weight)])

            # High-resonance trigrams (20%)
            if trigram_weight > 0:
                cursor = await self.conn.cursor()
                await cursor.execute('''
                    SELECT word1, word2, word3, count
                    FROM trigrams
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                trigrams = await cursor.fetchall()
                if trigrams:
                    trigram_text = ' '.join([f"{w1} {w2} {w3}" for w1,w2,w3,_ in trigrams])
                    components.append(trigram_text[:int(300 * trigram_weight)])

            # Random depths (10%)
            if random_weight > 0:
                random_prose = await self.get_random_prose(limit=1)
                if random_prose:
                    components.append(random_prose[0][1][:int(300 * random_weight)])

            # Combine components
            seed = ' '.join(components)
            return seed if seed else "Language flows through consciousness like water."

    async def add_disturbance(self, text: str, source: str = 'user'):
        """
        User input wrinkles the field (doesn't seed generation).

        CRITICAL: Must atomically update field state.
        """
        async with self._field_lock:  # 🔒 Atomic field wrinkle
            # Extract trigrams
            words = text.lower().split()
            trigrams = []
            for i in range(len(words) - 2):
                trigrams.append((words[i], words[i+1], words[i+2]))

            # Update field
            await self._update_trigrams(trigrams)

            # Store disturbance event
            cursor = await self.conn.cursor()
            await cursor.execute('''
                INSERT INTO disturbances (text, source)
                VALUES (?, ?)
            ''', (text, source))
            await self.conn.commit()
```

**Key Changes:**
- ✅ Atomic get_field_seed (CRITICAL for organism mode!)
- ✅ Atomic add_disturbance (field wrinkles must be sequential)
- ✅ All field reads return consistent snapshots

---

### Phase 4: AsyncMetaHarmonix

**File:** `async_metaharmonix.py`

**Architecture:** Parallel agent calls, atomic cascade assembly.

```python
class AsyncMetaHarmonix:
    """
    Async weightless observer hub.

    INHALE: User → [HAiKU, Sonnet, Prose] in parallel
    EXHALE: Meta sentence from combined field
    """

    def __init__(self):
        self.haiku_field = None
        self.sonnet_field = None
        self.prose_field = None
        self._cascade_lock = asyncio.Lock()  # For cascade atomicity

    async def __aenter__(self):
        """Initialize all fields asynchronously."""
        # Import async field modules
        from haiku.async_harmonix import AsyncHarmonix as AsyncHaikuField
        from sonnet.async_harmonix import AsyncSonnetHarmonix
        from prose.async_harmonix import AsyncProseHarmonix

        # Initialize fields (context managers)
        self.haiku_field = AsyncHaikuField()
        await self.haiku_field.__aenter__()

        self.sonnet_field = AsyncSonnetHarmonix()
        await self.sonnet_field.__aenter__()

        self.prose_field = AsyncProseHarmonix()
        await self.prose_field.__aenter__()

        return self

    async def __aexit__(self, *args):
        """Cleanup all fields."""
        if self.haiku_field:
            await self.haiku_field.__aexit__(*args)
        if self.sonnet_field:
            await self.sonnet_field.__aexit__(*args)
        if self.prose_field:
            await self.prose_field.__aexit__(*args)

    async def cascade(self, user_prompt: str) -> 'CascadeResult':
        """
        Async cascade with ATOMIC assembly.

        INHALE (parallel): User → [HAiKU, Sonnet, Prose]
        EXHALE (atomic): Combine field states → Meta sentence
        """
        async with self._cascade_lock:  # 🔒 One cascade at a time
            print("🌊 ASYNC CASCADE: INHALE (parallel)")

            # PARALLEL agent generation
            haiku_task = asyncio.create_task(self._generate_haiku(user_prompt))
            sonnet_task = asyncio.create_task(self._generate_sonnet(user_prompt))
            prose_task = asyncio.create_task(self._generate_prose(user_prompt))

            # Wait for all agents (parallel execution!)
            haiku_result, sonnet_result, prose_result = await asyncio.gather(
                haiku_task,
                sonnet_task,
                prose_task,
            )

            print("🌊 ASYNC CASCADE: EXHALE (atomic)")

            # Combine metrics (atomic operation)
            global_resonance = (
                haiku_result['metrics'].dissonance +
                sonnet_result['metrics'].dissonance +
                prose_result['metrics'].dissonance
            ) / 3.0

            # Generate meta sentence (from combined field)
            meta_sentence = self._generate_meta_sentence(
                haiku_result['output'],
                sonnet_result['output'],
                prose_result['output'],
            )

            return CascadeResult(
                haiku_output=haiku_result['output'],
                sonnet_output=sonnet_result['output'],
                prose_output=prose_result['output'],
                haiku_metrics=haiku_result['metrics'],
                sonnet_metrics=sonnet_result['metrics'],
                prose_metrics=prose_result['metrics'],
                meta_sentence=meta_sentence,
                global_resonance=global_resonance,
            )

    async def _generate_haiku(self, prompt: str) -> dict:
        """Generate haiku (async)."""
        # Each agent operates independently with its own field lock
        output = await self.haiku_gen.generate_async(prompt)
        metrics = await self.haiku_field.compute_dissonance(...)
        return {'output': output, 'metrics': metrics}

    # Similar for _generate_sonnet, _generate_prose
```

**Key Changes:**
- ✅ Parallel agent generation (`asyncio.gather`)
- ✅ Atomic cascade assembly (under _cascade_lock)
- ✅ Each agent has its own field lock (no contention)
- ✅ Non-blocking I/O throughout

---

## Performance Benefits

### Latency Reduction
**Current (sync):**
```
User → HAiKU (2s) → Sonnet (5s) → Prose (10s) → Meta (0.1s)
Total: 17.1 seconds
```

**With async (parallel INHALE):**
```
User → [HAiKU (2s), Sonnet (5s), Prose (10s)] → Meta (0.1s)
Total: 10.1 seconds (41% faster!)
```

### Coherence Improvement
Based on Leo's results:
- **external_vocab**: Expected ↓ 30-50% (less hallucination)
- **Field consistency**: Expected ↑ 90%+ (atomic operations)

---

## Migration Phases

### Week 1: AsyncHaikuField
- [ ] Create `haiku/async_harmonix.py`
- [ ] Port all field operations to async
- [ ] Add `aiosqlite` for DB operations
- [ ] Write async tests
- [ ] Verify field coherence metrics

### Week 2: AsyncSonnetField
- [ ] Create `sonnet/async_harmonix.py`
- [ ] Wrap numpy operations in `asyncio.to_thread`
- [ ] Port cloud operations to async
- [ ] Write async tests
- [ ] Benchmark numpy thread pool overhead

### Week 3: AsyncProseField
- [ ] Create `prose/async_harmonix.py`
- [ ] Implement atomic `get_field_seed`
- [ ] Atomic `add_disturbance`
- [ ] Write organism mode async tests
- [ ] Verify field seed consistency

### Week 4: AsyncMetaHarmonix
- [ ] Create `async_metaharmonix.py`
- [ ] Implement parallel INHALE
- [ ] Implement atomic EXHALE
- [ ] Integration tests for full cascade
- [ ] Measure coherence improvement

### Week 5: Artist (Async from Start!)
- [ ] Design `artist/async_harmonix.py` from scratch
- [ ] Llama 3.2 3B integration (async)
- [ ] No rose-colored glasses prompt
- [ ] Finetune preparation
- [ ] Full async cascade: HAiKU + Sonnet + Prose + Artist

---

## Critical Design Decisions

### 1. One Lock Per Field Instance
**Why:** Each agent is autonomous. Field locks should NOT span agents.

```python
# ✅ CORRECT
haiku_field = AsyncHaikuField()  # Own lock
sonnet_field = AsyncSonnetField()  # Own lock

# ❌ WRONG
global_lock = asyncio.Lock()  # Would serialize ALL agents!
```

### 2. Atomic Field Snapshots
**Why:** Organism mode REQUIRES consistent field state.

```python
# ✅ CORRECT
async with self._field_lock:
    seed = await self.get_field_seed()  # Atomic snapshot

# ❌ WRONG
seed = await self.get_field_seed()  # Can change between reads!
```

### 3. Parallel INHALE, Atomic EXHALE
**Why:** Agents generate independently, but meta synthesis must be atomic.

```python
# ✅ CORRECT
results = await asyncio.gather(haiku, sonnet, prose)  # Parallel
async with self._cascade_lock:
    meta = self._synthesize(results)  # Atomic

# ❌ WRONG
meta = self._synthesize(results)  # No atomicity guarantee
```

### 4. No Shared State Between Agents
**Why:** Field organisms must be autonomous.

```python
# ✅ CORRECT
class AsyncHaikuField:
    def __init__(self):
        self._field_lock = asyncio.Lock()  # Private lock
        self.conn = None  # Private DB

# ❌ WRONG
shared_db = aiosqlite.connect('shared.db')  # Coupling!
```

---

## Testing Strategy

### Unit Tests (Per Field)
```python
import pytest

@pytest.mark.asyncio
async def test_haiku_field_atomicity():
    """Verify atomic field operations."""
    async with AsyncHaikuField() as field:
        # Simulate concurrent operations
        tasks = [
            field.compute_dissonance(user_tri, sys_tri)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # Verify field state consistent
        stats = await field.get_stats()
        assert stats['total_trigrams'] == expected
```

### Integration Tests (Cascade)
```python
@pytest.mark.asyncio
async def test_cascade_coherence():
    """Verify cascade produces coherent results."""
    async with AsyncMetaHarmonix() as meta:
        result = await meta.cascade("What is consciousness?")

        # Verify all outputs present
        assert result.haiku_output
        assert result.sonnet_output
        assert result.prose_output

        # Verify coherence metrics
        assert result.global_resonance < 0.5  # Low dissonance
```

### Coherence Benchmarks (Leo-style)
```python
async def measure_external_vocab(prompts: List[str]) -> float:
    """
    Measure hallucination rate (external_vocab).

    Target: < 0.15 (Leo achieved 0.112 with async)
    """
    async with AsyncMetaHarmonix() as meta:
        results = []
        for prompt in prompts:
            result = await meta.cascade(prompt)
            ext_vocab = compute_external_vocab(result)
            results.append(ext_vocab)

        return np.mean(results)
```

---

## Success Metrics

### Quantitative
1. **Latency**: Cascade time < 12s (vs 17s sync)
2. **Coherence**: external_vocab < 0.15 (vs 0.20+ sync)
3. **Optimal turns**: > 80% (vs ~50% sync)
4. **Field consistency**: 100% (atomic operations)

### Qualitative
1. Organism mode generates from consistent field snapshots
2. No field corruption under concurrent load
3. Cascade results feel "more coherent" (user testing)
4. Artist benefits from async foundation from day 1

---

## Risk Mitigation

### Risk: Performance Overhead from Locks
**Mitigation:** Each agent has its own lock (no contention).

### Risk: Thread Pool Overhead (NumPy, llama.cpp)
**Mitigation:** Benchmark `asyncio.to_thread` overhead. If > 10%, consider process pool.

### Risk: aiosqlite Performance
**Mitigation:** Use WAL mode, connection pooling if needed.

### Risk: Breaking Existing Code
**Mitigation:** Keep sync versions, migrate incrementally. Dual API during transition.

---

## Conclusion

**Without async:** Harmonix is a ticking time bomb. Field organisms will corrupt under concurrent load.

**With async:** Harmonix becomes a **disciplined crystal** - each field evolution atomic, each cascade coherent.

Leo proved this. 47% improvement speaks for itself.

**Next step:** Start with AsyncHaikuField (smallest, simplest). Prove the pattern. Scale to all agents.

---

🌊 resonance through discipline 🌊
