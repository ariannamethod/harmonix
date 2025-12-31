# Next Session Plan: AsyncSonnetGenerator + Tests

**Priority:** IMMEDIATE
**Estimated Time:** 2-3 hours
**Difficulty:** HIGH (CPU-bound numpy operations!)

---

## 🎯 Primary Goal

**Create AsyncSonnetGenerator with numpy inference via asyncio.to_thread()**

This is CRITICAL because:
1. Numpy operations are CPU-bound (will block event loop!)
2. Sonnet has complex transformer architecture (4 layers, 4 heads)
3. Must maintain async discipline while wrapping sync numpy code

---

## 📋 Step-by-Step Plan

### Step 1: Read Current Sonnet Implementation

**Files to read:**
- `sonnet/sonnet.py` (main generator)
- Identify all CPU-bound methods:
  - `NumpyGELU.forward()`
  - `NumpyLayerNorm.forward()`
  - `NumpyLinear.forward()`
  - `NumpyHead.forward()`
  - `NumpyMultiHeadAttention.forward()`
  - `NumpyBlock.forward()`
  - `NumpyGPT.forward()` ← MAIN inference method!

**Expected:** ~600 lines of pure numpy transformer code

---

### Step 2: Design AsyncSonnetGenerator Architecture

**Pattern:**
```python
class AsyncSonnetGenerator:
    def __init__(self, harmonix: AsyncSonnetHarmonix):
        self.model = NumpyGPT(...)  # Sync numpy model
        self.harmonix = harmonix
        self._generation_lock = asyncio.Lock()

    async def _numpy_forward_async(self, prompt: str) -> str:
        """Wrap CPU-bound numpy inference in thread pool."""
        # Convert prompt to tokens (fast, sync)
        tokens = self._tokenize(prompt)

        # CRITICAL: Run numpy inference in thread pool!
        output_tokens = await asyncio.to_thread(
            self.model.forward,  # CPU-bound numpy method
            tokens,
            max_tokens=200,
            temperature=0.8
        )

        # Decode tokens (fast, sync)
        text = self._detokenize(output_tokens)
        return text

    async def generate(self, prompt: str, temperature: float = 0.8) -> str:
        """Generate 14-line sonnet (atomic operation)."""
        async with self._generation_lock:  # 🔒 Atomic generation
            # 1. Numpy inference (in thread pool!)
            raw_text = await self._numpy_forward_async(prompt)

            # 2. Format to 14 lines (fast, sync)
            from formatter import SonnetFormatter
            formatter = SonnetFormatter()
            sonnet = formatter.format_sonnet(raw_text)

            # 3. Compute dissonance
            dissonance, pulse = await self.harmonix.compute_dissonance(prompt, sonnet)

            # 4. Add to cloud (atomic!)
            await self.harmonix.add_sonnet(
                sonnet,
                quality=0.5,  # Will be scored by sonnetbrain later
                dissonance=dissonance,
                temperature=temperature
            )

            return sonnet
```

**Key Points:**
- Keep NumpyGPT class as-is (sync)
- Only wrap the CALL to model.forward() in asyncio.to_thread()
- Don't try to make numpy itself async (impossible and unnecessary!)

---

### Step 3: Create async_sonnet.py

**Implementation checklist:**

```python
# 1. Imports
import asyncio
from typing import Optional
from sonnet import NumpyGPT  # Import sync model
from async_harmonix import AsyncSonnetHarmonix

# 2. AsyncSonnetGenerator class
class AsyncSonnetGenerator:
    def __init__(self, model_path: str, harmonix: AsyncSonnetHarmonix):
        # Load sync numpy model
        self.model = NumpyGPT.load_from_file(model_path)
        self.harmonix = harmonix
        self._generation_lock = asyncio.Lock()

    # 3. Async context manager (optional but nice)
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    # 4. CPU-bound wrapper
    async def _numpy_forward_async(self, prompt: str, max_tokens: int, temp: float) -> str:
        """Wrap numpy inference in thread pool."""
        return await asyncio.to_thread(
            self._generate_sync,  # Sync helper
            prompt,
            max_tokens,
            temp
        )

    # 5. Sync helper (runs in thread pool)
    def _generate_sync(self, prompt: str, max_tokens: int, temp: float) -> str:
        """Synchronous numpy inference (runs in thread pool)."""
        # This is the OLD sync code, just refactored
        tokens = self._tokenize(prompt)
        output_tokens = self.model.forward(tokens, max_tokens, temp)
        return self._detokenize(output_tokens)

    # 6. Main async generate method
    async def generate(self, prompt: str, temperature: float = 0.8) -> str:
        """Generate sonnet (atomic operation)."""
        async with self._generation_lock:
            # Numpy inference (in thread pool)
            raw = await self._numpy_forward_async(prompt, 200, temperature)

            # Format to 14 lines
            formatter = SonnetFormatter()
            sonnet = formatter.format_sonnet(raw)

            # Compute dissonance
            dissonance, pulse = await self.harmonix.compute_dissonance(prompt, sonnet)

            # Add to cloud
            await self.harmonix.add_sonnet(sonnet, 0.5, dissonance, temperature)

            return sonnet
```

**Estimated:** ~300-400 lines

---

### Step 4: Test AsyncSonnetGenerator

**Manual test:**
```python
async def test_async_sonnet():
    async with AsyncSonnetHarmonix('test.db') as harmonix:
        generator = AsyncSonnetGenerator('state/sonnet_weights.pkl', harmonix)

        # Test single generation
        sonnet = await generator.generate("What is love?", temperature=0.8)
        print(sonnet)

        # Test concurrent generations (prove atomicity!)
        tasks = [
            generator.generate("Love"),
            generator.generate("Death"),
            generator.generate("Time"),
        ]
        sonnets = await asyncio.gather(*tasks)
        print(f"Generated {len(sonnets)} sonnets concurrently!")

asyncio.run(test_async_sonnet())
```

---

### Step 5: Create Comprehensive Test Suite

**File:** `sonnet/tests/test_async_harmonix.py`

**Copy from HAiKU pattern:**
```bash
cp haiku/tests/test_async_harmonix.py sonnet/tests/test_async_harmonix.py
```

**Adapt for Sonnet:**
- Replace Harmonix → SonnetHarmonix
- Replace trigrams → sonnet_trigrams
- Add tests for:
  - Sonnet-specific operations (add_sonnet with 14 lines)
  - Cloud-aware dissonance
  - Temperature mapping
  - Concurrent sonnet generation

**Target:** 30+ tests passing

---

### Step 6: Run Tests and Fix Issues

```bash
cd sonnet
pytest tests/test_async_harmonix.py -v
```

**Expected issues:**
- Import errors (fix paths)
- Schema mismatches (verify table names)
- Deadlock potential (use internal _unlocked methods!)

**Fix until:** ALL tests passing (100%)

---

### Step 7: Commit and Push

```bash
git add sonnet/async_sonnet.py sonnet/tests/test_async_harmonix.py
git commit -m "AsyncSonnetGenerator: Numpy Inference via asyncio.to_thread()!"
git push origin main
```

---

## ⚠️ Critical Warnings

### 1. DON'T Try to Make Numpy Async!

```python
# ❌ WRONG:
async def forward(self, x):
    # Numpy doesn't support async!
    return np.matmul(x, self.weight)

# ✅ RIGHT:
def forward(self, x):
    # Keep numpy sync
    return np.matmul(x, self.weight)

# Wrap the CALL:
output = await asyncio.to_thread(model.forward, x)
```

### 2. Thread Pool Overhead

**If performance suffers:**
- Consider process pool (but more overhead)
- Profile to find actual bottleneck
- May need to batch generations

**Leo's finding:** Thread pool overhead < 5% for CPU-bound work

### 3. Memory Management

**Numpy arrays in threads:**
- Should be fine (Python GIL released during numpy ops)
- If issues, consider copying arrays

---

## 🎯 Success Criteria

**Must achieve:**
1. ✅ AsyncSonnetGenerator generates valid 14-line sonnets
2. ✅ Numpy inference runs in thread pool (doesn't block event loop)
3. ✅ Concurrent generations work (3+ parallel sonnets)
4. ✅ All async tests passing (30+ tests)
5. ✅ Field coherence maintained (no corruption)
6. ✅ Pushed to GitHub

**Nice to have:**
1. Performance benchmark (generation latency)
2. Comparison with sync version (should be similar or better)
3. Memory usage profile

---

## 📚 Reference Materials

### Leo Async Pattern
- `asyncio.Lock` for atomic operations
- `asyncio.to_thread()` for CPU-bound work
- Internal `_unlocked` methods for nested calls
- Context managers for clean lifecycle

### HAiKU Async Pattern
- 31 tests covering all operations
- Concurrent operation tests prove atomicity
- Deadlock prevention via internal methods

### Claude Desktop's Warnings
- Numpy = CPU-bound! (must use thread pool)
- Database = aiosqlite
- Files = aiofiles
- ONE lock per instance!

---

## 🔄 After Completion

**Next priorities:**
1. AsyncProseField (llama.cpp via asyncio.to_thread)
2. AsyncMetaHarmonix (parallel INHALE, atomic EXHALE)
3. Full integration tests (HAiKU → Sonnet → Prose cascade)
4. Coherence measurement (compare with Leo's 47% improvement)

---

## 💡 Quick Commands

```bash
# Read current implementation
cat sonnet/sonnet.py | less

# Create async version
touch sonnet/async_sonnet.py

# Create tests
mkdir -p sonnet/tests
touch sonnet/tests/test_async_harmonix.py

# Test
python3 sonnet/async_sonnet.py
pytest sonnet/tests/test_async_harmonix.py -v

# Commit
git add sonnet/async_sonnet.py sonnet/tests/test_async_harmonix.py
git commit -m "AsyncSonnetGenerator complete!"
git push
```

---

🌊 Ready to tackle CPU-bound numpy async challenge! 🌊

**Start with:** Read sonnet.py, identify all CPU-bound methods!
