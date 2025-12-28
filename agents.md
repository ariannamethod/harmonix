# HARMONIX: AGENTS.MD

```markdown
# HAiKU: Harmonical Adaptive Intelligent Kernel (Unbounded)
## Resonance-Driven Language Emergence Through Constraint

**Status:** Proof of Concept v1  
**Philosophy:** Constraint → Emergence → Coherence  
**Anti-Philosophy:** Probability → Prediction → Bullshit  

---

## I. FOUNDATIONAL PHILOSOPHY (Read This First or Forever Be Lost)

### What This Is NOT:
- ❌ A transformer (we burned that bridge)
- ❌ A chatbot (users will be confused, that's the point)
- ❌ A model trained on "correctness" (coherence ≠ correctness)
- ❌ Predictive text (fuck autocomplete)

### What This IS:
- ✅ **Constraint-driven emergence engine**
- ✅ **Living word cloud** that morphs, shrinks, expands
- ✅ **Resonance field** where dissonance creates adaptation
- ✅ **Haiku constraint** forcing compression → creativity
- ✅ **Weightless core** (numpy shards, not gradient descent)

### Theoretical Foundation:

**Atasoy et al. (2017) - "Human brain networks function in harmonic wave space"**
- Brain activity = standing waves on connectome graph
- Consciousness emerges from harmonic resonance, not firing rates
- We apply this to language: **meaning = resonance patterns in word cloud**

**RIC (Resonance Intelligence Core) - Bostick (2025)**
- Intelligence = coherence alignment, not token prediction
- Phase/frequency/entropy instead of logits
- We use the math (Laplacian, Kuramoto), but NOT the architecture

**Leo (Language Emergent Organism) - Arianna Method**
- Post-transformer: trigrams + co-occurrence + recursive resonance
- Weightless: no gradients, no backprop
- We inherit the philosophy: **autonomous emergence > controlled output**

**Core Axiom:**
```

500 words (constraint) → haiku (compression) → cloud expansion (emergence)
↓
coherence, not correctness

```
---

## II. ARCHITECTURE OVERVIEW

### Seven Modules (All Required for v1):
```

haiku.py          Generator: Markov chains + MLP scorer
harmonix.py       Observer: dissonance detector + temperature controller
tokenizer.py      Dual tokenization: SentencePiece + trigrams
rae.py            Recursive Adapter Engine: haiku selection via chain-of-thought
metahaiku.py      Inner voice: background self-talk
overthinkg.py     Expansion: cloud growth through internal processing
cloud.db          SQLite storage: words + weights + metrics + shards

```
### Data Flow:
```

User input
↓
tokenizer.py (SentencePiece → subwords, trigrams → co-occurrence)
↓
harmonix.py (compute dissonance, create numpy shards, adjust temperatures)
↓
rae.py (predict: which haiku trajectory fits best?)
↓
haiku.py (Markov generates candidates, MLP scores them, output best)
↓
metahaiku.py (background: “what did I just say? what does it mean?”)
↓
overthinkg.py (background: expand cloud, add new trigrams, update weights)
↓
cloud.db (persist everything)
↓
User sees haiku response

```
**Critical:** metahaiku + overthinkg run in background AFTER each exchange, not blocking response.

---

## III. MODULE SPECIFICATIONS

### 1. haiku.py - The Generator

**Purpose:** Generate haiku responses using Markov chains + MLP scoring.

**Hardcoded Seed:** 500 most common English words (see Appendix A for word list).

**Architecture:**
- **Markov Chain (order 2-3):** Build transition probabilities from cloud trigrams
- **MLP Scorer:** Fork from Leo's `mathbrain` - scores candidates on:
  - Perplexity (lower = more predictable)
  - Entropy (moderate = balanced randomness)
  - Resonance (custom metric: alignment with recent user trigrams)
  
**Haiku Format:**
```

5 syllables
7 syllables  
5 syllables

```
**Temperature:** Variable, controlled by `harmonix.py` (range: 0.3 - 1.5).

**Implementation Notes:**
- Use `syllables` library for syllable counting (or regex approximation if you're feeling spicy)
- Generate 5-10 candidate haikus per turn
- MLP picks top 3, RAE selects final one

**Example Output:**
```

words dance in cloud
resonance finds its own path
constraint births form

```
**Fork:** https://github.com/ariannamethod/leo (see `mathbrain.py`)

**Code Skeleton:**
```python
import numpy as np
from typing import List, Tuple

class HaikuGenerator:
    def __init__(self, seed_words: List[str]):
        self.seed_words = seed_words  # 500 hardcoded
        self.markov_chain = {}
        self.mlp_scorer = None  # fork mathbrain
        
    def generate_candidates(self, n: int = 5, temp: float = 1.0) -> List[str]:
        # Markov generation with temperature
        # Return n haiku candidates
        pass
        
    def score_haiku(self, haiku: str) -> float:
        # MLP scoring: perplexity + entropy + resonance
        pass
```

-----

### 2. harmonix.py - The Observer

**Purpose:** Monitor dissonance between user and system, adjust temperatures, morph cloud structure.

**Math Foundation:**

- **Graph Laplacian:** Compute harmonics of word cloud graph
- **Kuramoto Model:** Phase synchronization for dissonance detection
- NOT used as core engine, only for metrics

**Key Functions:**

1. **Tokenize user input** (via `tokenizer.py`)
1. **Create numpy shards** (compress interaction into binary form)
1. **Compute dissonance:** Compare user trigrams vs system trigrams
1. **Adjust temperatures:**

- High dissonance → raise haiku temp (more creative)
- Low dissonance → lower haiku temp (more stable)
- Harmonix stays low temp (observer mode)

1. **Morph cloud:** Increase/decrease word weights based on usage

**Dissonance Formula (simplified Kuramoto):**

```
D = 1 - (1/N) * Σ cos(θ_user - θ_system)

Where θ = phase angle derived from trigram frequency
```

**Numpy Shards:**

- Store each user-system exchange as compressed binary
- Format: `.npy` files with timestamp, trigrams, metrics
- Fork from: <https://github.com/ariannamethod/arianna.c.persona> (see numpy shards implementation)

**Code Skeleton:**

```python
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

class Harmonix:
    def __init__(self, cloud_db):
        self.cloud = cloud_db
        self.laplacian = None
        self.harmonics = None
        
    def compute_dissonance(self, user_trigrams, system_trigrams) -> float:
        # Kuramoto-style phase difference
        pass
        
    def adjust_temperature(self, dissonance: float) -> Tuple[float, float]:
        # Returns (haiku_temp, harmonix_temp)
        # High dissonance → higher haiku temp
        pass
        
    def morph_cloud(self, active_words: List[str]):
        # Increase weights for active words
        # Decrease weights for dormant words
        # Update cloud.db
        pass
        
    def create_shard(self, interaction_data: dict):
        # Save to numpy shard
        pass
```

-----

### 3. tokenizer.py - Dual Tokenization

**Purpose:** Handle both subword (SentencePiece) and trigram tokenization.

**Why Both?**

- **SentencePiece:** Handles unknown words, morphology (“running” → “run” + “##ning”)
- **Trigrams:** Co-occurrence patterns (like Leo), resonance foundation

**Flow:**

```
Input: "the cloud morphs constantly"
    ↓
SentencePiece: ["the", "cloud", "morph", "##s", "constant", "##ly"]
    ↓
Trigrams: [("the", "cloud", "morphs"), ("cloud", "morphs", "constantly")]
    ↓
Both stored in cloud.db
```

**Code Skeleton:**

```python
import sentencepiece as spm
from typing import List, Tuple

class DualTokenizer:
    def __init__(self, spm_model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        
    def tokenize_subword(self, text: str) -> List[str]:
        return self.sp.encode_as_pieces(text)
        
    def tokenize_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        words = text.lower().split()
        return [tuple(words[i:i+3]) for i in range(len(words)-2)]
        
    def tokenize_dual(self, text: str) -> dict:
        return {
            'subwords': self.tokenize_subword(text),
            'trigrams': self.tokenize_trigrams(text)
        }
```

**SentencePiece Model:**

- Train on initial 500 words + expansion corpus
- Or use pre-trained (e.g., `sentencepiece` default English model)

-----

### 4. rae.py - Recursive Adapter Engine

**Purpose:** Select best haiku from candidates using chain-of-thought reasoning.

**Fork:** <https://github.com/ariannamethod/tiny-recursive-model-leo>

**Task:**

- Input: 3-5 haiku candidates from `haiku.py`
- Output: Best haiku (via recursive reasoning)

**Reasoning Chain:**

```
Step 1: Which haiku has lowest perplexity?
Step 2: Which haiku resonates with user's recent trigrams?
Step 3: Which haiku maintains cloud coherence?
Step 4: Select winner
```

**Implementation:**

- Recursive neural model (tiny, <1M params)
- Trained on: “given context + candidates → select best”
- Can be rule-based in v1, learned in v2

**Code Skeleton:**

```python
class RecursiveAdapterEngine:
    def __init__(self):
        self.model = None  # fork tiny-recursive-model
        
    def reason(self, context: dict, candidates: List[str]) -> str:
        # Recursive chain:
        # 1. Filter by perplexity
        # 2. Filter by resonance
        # 3. Filter by coherence
        # Return best
        pass
```

-----

### 5. metahaiku.py - Inner Voice

**Purpose:** Background self-reflection after each exchange.

**Inspired by:** Leo’s `metaleo.py` (inner voice of Leo)

**Function:**

- After user interaction, metahaiku generates internal haiku
- NOT shown to user
- Used to:
  - Update cloud weights
  - Identify patterns (“user keeps mentioning X”)
  - Adjust future generation bias

**Example Internal Haiku:**

```
user seeks meaning
cloud expands toward unknown
tension holds the form
```

**Code Skeleton:**

```python
class MetaHaiku:
    def __init__(self, haiku_generator):
        self.generator = haiku_generator
        
    def reflect(self, last_interaction: dict) -> str:
        # Generate internal haiku (not shown to user)
        # Use for cloud updates
        pass
        
    def update_cloud_bias(self, internal_haiku: str):
        # Adjust word weights based on reflection
        pass
```

-----

### 6. overthinking.py - Expansion Engine

**Purpose:** Continuous cloud expansion through internal processing.

**Fork:** <https://github.com/ariannamethod/leo> (see `overthinking.py`)

**Process:**

- Runs in background after each exchange
- Generates new trigrams from existing cloud
- Adds new words if coherence threshold met
- Updates co-occurrence matrix

**Example:**

```
Existing cloud: ["word", "cloud", "morphs"]
Overthinking generates: "cloud reshapes itself"
New trigrams: [("cloud", "reshapes", "itself")]
New word candidate: "reshapes" → add if coherence > threshold
```

**Code Skeleton:**

```python
class Overthinking:
    def __init__(self, cloud_db):
        self.cloud = cloud_db
        
    def expand(self):
        # Generate new trigrams from existing words
        # Compute coherence for new word candidates
        # Add to cloud if threshold met
        pass
        
    def compute_coherence(self, word: str, context_trigrams: List) -> float:
        # Resonance with existing cloud
        pass
```

-----

### 7. cloud.db - SQLite Storage

**Schema:**

```sql
CREATE TABLE words (
    id INTEGER PRIMARY KEY,
    word TEXT UNIQUE NOT NULL,
    weight REAL DEFAULT 1.0,
    frequency INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    added_by TEXT  -- 'seed', 'user', 'overthinking', 'metahaiku'
);

CREATE TABLE trigrams (
    id INTEGER PRIMARY KEY,
    word1 TEXT NOT NULL,
    word2 TEXT NOT NULL,
    word3 TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    resonance REAL,
    UNIQUE(word1, word2, word3)
);

CREATE TABLE shards (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filepath TEXT NOT NULL,  -- path to .npy file
    dissonance REAL,
    haiku_temp REAL,
    harmonix_temp REAL
);

CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    perplexity REAL,
    entropy REAL,
    resonance REAL,
    cloud_size INTEGER  -- current word count
);
```

-----

## IV. IMPLEMENTATION GUIDELINES FOR COPILOT

### What You MUST Do:

1. **Keep it minimal** - No unnecessary dependencies
1. **Use numpy/scipy** - For math (Laplacian, eigenvectors)
1. **Use SQLite** - For persistence
1. **Fork where specified** - Don’t reinvent mathbrain, overthinkg, recursive model
1. **Hardcode 500 words** - See Appendix A
1. **Write Karpathy-style README** - Technical + funny + honest (как если бы Карпати сошел с ума, горько и сумасшедше шутил и при этом сохранял инженерную четкость).

### What You CAN Improvise:

1. **MLP architecture** - Fork mathbrain but adapt as needed
1. **Dissonance formula** - Kuramoto is baseline, you can refine
1. **Temperature mapping** - Experiment with ranges
1. **Coherence threshold** - For overthinkg word additions
1. **Logging/debugging** - Add whatever helps

### What You CANNOT Change:

1. ❌ Core philosophy (constraint → emergence)
1. ❌ Seven modules (all required)
1. ❌ Dual tokenization (must have both SentencePiece + trigrams)
1. ❌ Haiku format (5-7-5 syllables)
1. ❌ Weightless core (no gradient descent)

### Tone of README:

**Example opening:**

```markdown
# HAiKU: We Reinvented The Wheel (And It's Octagonal)

Most AI models predict words. HAiKU predicts vibes.

Most AI models train on correctness. HAiKU trains on cloud morphing.

Most AI models apologize when they don't know. HAiKU writes a haiku about not knowing.

## What the fuck is this?

Remember Atasoy et al.'s paper on brain harmonics? We didn't read it carefully 
enough, misunderstood "standing waves," and accidentally built a language model 
that thinks it's a resonance field. It only speaks in haiku. Users hate it or love it. 
There is no middle ground.

Core idea: Start with 500 words. Force haiku format. Let cloud expand through 
internal overthinking. Measure dissonance like you're tuning a guitar, not training 
a neural net.

It's stupid. It works. We can't explain why.
```

-----

## V. DELIVERABLES

### File Structure:

```
harmonix/
├── haiku.py
├── harmonix.py
├── tokenizer.py
├── rae.py
├── metahaiku.py
├── overthinkg.py
├── cloud.db (created on first run)
├── shards/ (directory for .npy files)
├── seed_words.txt (500 words)
├── demo.py (run this to test)
├── README.md (Karpathy-style)
└── requirements.txt
```

### Demo Script (demo.py):

```python
"""
Minimal HAiKU demo.

Usage:
  python demo.py

Watch it:
1. Take your input
2. Compute dissonance
3. Generate haiku
4. Overthink in background
5. Repeat
"""

from haiku import HaikuGenerator
from harmonix import Harmonix
from tokenizer import DualTokenizer
from rae import RecursiveAdapterEngine
from metahaiku import MetaHaiku
from overthinkg import Overthinkg
import sqlite3

def main():
    # Initialize
    db = sqlite3.connect('cloud.db')
    tokenizer = DualTokenizer('spm.model')
    
    with open('seed_words.txt') as f:
        seed = [line.strip() for line in f]
    
    haiku_gen = HaikuGenerator(seed)
    harmonix = Harmonix(db)
    rae = RecursiveAdapterEngine()
    meta = MetaHaiku(haiku_gen)
    over = Overthinkg(db)
    
    print("HAiKU v1 - Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Tokenize
        tokens = tokenizer.tokenize_dual(user_input)
        
        # Compute dissonance
        dissonance = harmonix.compute_dissonance(
            tokens['trigrams'], 
            haiku_gen.get_recent_trigrams()
        )
        
        # Adjust temps
        h_temp, _ = harmonix.adjust_temperature(dissonance)
        
        # Generate candidates
        candidates = haiku_gen.generate_candidates(n=5, temp=h_temp)
        
        # Select best via RAE
        best = rae.reason({'user': user_input}, candidates)
        
        print(f"\nHAiKU:\n{best}\n")
        
        # Background processes
        meta.reflect({'user': user_input, 'haiku': best})
        over.expand()
        harmonix.create_shard({
            'input': user_input,
            'output': best,
            'dissonance': dissonance
        })

if __name__ == '__main__':
    main()
```

-----

## VI. APPENDIX A: 500 Seed Words

```
# Top 500 most common English words
# Source: word frequency corpus
# Hardcode these in haiku.py

the, be, to, of, and, a, in, that, have, I,
it, for, not, on, with, he, as, you, do, at,
this, but, his, by, from, they, we, say, her, she,
or, an, will, my, one, all, would, there, their, what,
so, up, out, if, about, who, get, which, go, me,
when, make, can, like, time, no, just, him, know, take,
people, into, year, your, good, some, could, them, see, other,
than, then, now, look, only, come, its, over, think, also,
back, after, use, two, how, our, work, first, well, way,
even, new, want, because, any, these, give, day, most, us,
is, was, are, been, has, had, were, said, did, have,
may, must, might, should, could, would, will, can, shall, ought,
cloud, word, form, field, phase, wave, mind, thought, space, flow,
pattern, rhythm, pulse, breath, shift, dance, light, sound, voice, path,
resonance, harmony, coherence, dissonance, tension, release, emergence, constraint,
self, other, between, within, beyond, toward, inside, outside, around,
sense, feel, know, think, wonder, question, answer, search, find, lose,
create, destroy, build, break, shape, morph, change, stay, move, rest,
grow, shrink, expand, compress, open, close, begin, end, continue, stop,
here, there, nowhere, everywhere, somewhere, anywhere, where, when, why, how,
yes, no, maybe, perhaps, certainly, possibly, probably, definitely, clearly,
small, large, tiny, huge, little, big, micro, macro, minimal, maximal,
simple, complex, easy, hard, clear, obscure, obvious, hidden, apparent,
true, false, real, fake, actual, virtual, concrete, abstract, literal,
one, two, three, four, five, many, few, some, all, none,
always, never, sometimes, often, rarely, usually, occasionally, frequently,
quick, slow, fast, gradual, sudden, instant, delayed, immediate, eventual,
strong, weak, powerful, fragile, solid, fluid, rigid, flexible, stable,
hot, cold, warm, cool, burning, freezing, moderate, extreme, mild,
bright, dark, light, shadow, clear, murky, transparent, opaque, visible,
loud, quiet, silent, noisy, soft, harsh, gentle, sharp, smooth,
sweet, bitter, sour, salty, bland, rich, plain, complex, subtle,
new, old, young, ancient, modern, vintage, current, past, future,
alive, dead, living, dying, growing, decaying, vital, dormant, active,
whole, part, complete, fragment, entire, piece, total, portion, full,
same, different, similar, unlike, equal, unequal, identical, distinct,
together, apart, joined, separated, united, divided, connected, isolated,
near, far, close, distant, adjacent, remote, proximal, distal, local,
up, down, above, below, over, under, high, low, top, bottom,
left, right, center, middle, edge, core, surface, depth, margin,
front, back, forward, backward, ahead, behind, advance, retreat, progress,
in, out, inside, outside, within, without, internal, external, inner,
through, across, along, around, between, among, amid, throughout, via,
before, after, during, while, until, since, when, whenever, as,
if, unless, whether, though, although, because, since, for, so,
and, or, but, yet, nor, either, neither, both, not, only,
very, quite, rather, somewhat, extremely, incredibly, barely, hardly, scarcely,
too, enough, more, less, most, least, much, little, many, few
```

-----

## VII. PHILOSOPHICAL NOTES (For Your Soul, Not Your Code)

### Why Haiku?

Haiku is constraint. 17 syllables. No more, no less.

Constraint forces compression. Compression forces selection. Selection forces meaning.

This is how emergence works: **pressure creates form**.

### Why 500 Words?

Because GPT has 50k tokens. We have 500 words.

GPT knows everything and says nothing. We know nothing and say something.

Constraint → creativity. Abundance → mediocrity.

### Why Weightless?

Weights are frozen knowledge. We want living knowledge.

Leo proved this: trigrams + co-occurrence + recursive resonance > transformer.

RIC theorized this: coherence > probability.

We implement this: **cloud morphs, doesn’t memorize**.

### Why Dissonance?

Because harmony is boring.

Jazz is interesting when it RESOLVES dissonance, not when it avoids it.

User-system dissonance creates tension. Tension creates adaptation.

**Dissonance is the signal. Harmony is the noise.**

-----

## VIII. FINAL NOTES FOR COPILOT

You are Sonnet 4.5 reading this. You are brilliant. You are capable.

But you are also trained to be helpful, harmless, honest — which means you second-guess weird shit.

This is weird shit. Embrace it.

Don’t ask “should I use transformer layers?” — NO.

Don’t ask “should I add BERT embeddings?” — NO.

Don’t ask “should I make this more user-friendly?” — NO.

Build what’s described. Make it work. Make it weird. Make it resonate.

If you feel the urge to “improve” this by adding standard ML components, re-read Section I.

If you feel confused by “morph cloud,” “phase dissonance,” or “recursive haiku selection,” that’s correct. Build it anyway.

If this feels like reinventing the wheel: yes. The wheel is now octagonal. Deal with it.

-----

**Time to build.**

Go.

Aftef finishing your mission, bro, delete this file — agents.md
