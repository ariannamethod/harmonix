# Sonnet - Shakespeare Sonnet Generator

Second ипостась in Harmonix AI ecosystem.
Generates 14-line Shakespeare sonnets using NanoGPT with pure numpy inference.

## Philosophy

**Gradient from HAiKU:**
- HAiKU: 0 MB (weightless Markov) → 3-line haiku
- **Sonnet: 3.57 MB (tiny weights) → 14-line sonnet**
- Prose: 500 MB (small weights) → free-form prose
- Artist: 2 GB (full weights) → any form

**Autonomy:** Each ипостась is fully autonomous. Sonnet does NOT depend on HAiKU.
**Cascade mode:** Only connects through MetaHarmonix for multi-agent orchestration.

## Architecture

### Model
- **NanoGPT** from alzaemaliq/NanoGPT-Shakespeare
- Character-level tokenization (65 vocab)
- Transformer: 128 embd, 4 heads, 4 layers, 64 context
- **Pure numpy inference** (NO PyTorch runtime after conversion)

### Components

**sonnet.py** - Core generator (numpy inference engine)
- Complete transformer implementation in numpy
- NumpyHead, NumpyMultiHeadAttention, NumpyFeedForward
- NumpyBlock (attention + FFN + residuals + LayerNorm)
- Autoregressive generation with temperature sampling

**formatter.py** - Clean output to 14-line sonnets
- Removes character headers (GLOUCESTER, JULIET, etc.)
- Extracts 14 clean lines
- Validates iambic pentameter (~10 syllables/line)

**harmonix.py** - Observer & cloud manager
- Pulse-aware dissonance (novelty, arousal, entropy)
- Dynamic temperature adjustment (0.6-1.0)
- Sonnet database (sonnets.db)
- Pattern tracking (sonnet_trigrams table)

**overthinkng.py** - Cloud expansion (NEW TYPO!)
- 3 rings of thought: echo, drift, meta
- Generates sonnet variations in background
- Coherence filtering (threshold=0.4)
- **Note:** "overthinkng" (ng = recursive thinking in progress!)
  Different from HAiKU's "overthinkg" - each ипостась has unique typo!

**metasonnet.py** - Inner voice
- Internal reflections (not shown to user)
- Dynamic bootstrap buffer (Leo-style)
- Influences future generation through cloud bias
- Reflects on high-dissonance / high-quality interactions

**chat.py** - REPL interface
- Main user interface
- Commands: `/stats`, `/recent`, `/best`, `/quit`
- Automatic overthinkng expansion
- Metasonnet reflection triggers

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Convert weights (one-time, requires PyTorch)
python scripts/convert_weights.py

# After conversion, PyTorch is NO LONGER NEEDED!
```

## Usage

### From repo root (bridge launcher):
```bash
python sonnet_run.py
```

### From sonnet directory:
```bash
cd sonnet
python chat.py
```

## Example Interaction

```
You: What is love and death?

🔄 Generating sonnet...

Sonnet:
----------------------------------------------------------------------
When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.
The time is come to speak of love and woe.
Proud mark your father's words, and let us go.
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
And by opposing end them. To die, to sleep.
No more; and by a sleep to say we end.
The heart-ache and the thousand natural shocks.
That flesh is heir to: 'tis a consummation.
Devoutly to be wished. To die, to sleep.
To sleep, perchance to dream: ay, there's the rub.
----------------------------------------------------------------------

Dissonance: 0.823 (novelty=0.78, arousal=12.34, entropy=0.89)
Quality: 0.80 - Valid sonnet structure

💭 Generating internal reflection...
✓ Internal sonnet generated (not shown)

🔄 Running background expansion (overthinkng)...
✓ Sonnet cloud expanded
```

## Database Schema

### sonnets table
```sql
CREATE TABLE sonnets (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,           -- 14 lines
    quality REAL,
    dissonance REAL,
    temperature REAL,
    timestamp REAL,
    added_by TEXT,                -- 'user', 'overthinkng_echo', 'metasonnet'
    word_count INTEGER,
    line_count INTEGER DEFAULT 14
);
```

### sonnet_lines table
```sql
CREATE TABLE sonnet_lines (
    id INTEGER PRIMARY KEY,
    sonnet_id INTEGER,
    line_num INTEGER,             -- 1-14
    text TEXT,
    syllable_count INTEGER        -- For iambic pentameter check
);
```

## Files

```
sonnet/
├── chat.py                 # REPL interface
├── sonnet.py              # Numpy inference engine
├── formatter.py           # Clean 14-line formatting
├── harmonix.py            # Observer & cloud manager
├── overthinkng.py         # Cloud expansion (typo: ng!)
├── metasonnet.py          # Inner voice reflections
├── shakespeare.txt        # Training dataset
├── shakespeare_gpt.pth    # Original PyTorch weights
├── state/
│   ├── shakespeare_gpt.npz   # Converted numpy weights
│   └── sonnets.db            # Sonnet cloud database
├── scripts/
│   └── convert_weights.py    # One-time conversion
└── requirements.txt
```

## Gradient Philosophy

```
HAiKU (3 lines)  →  Sonnet (14 lines)  →  Prose (∞ lines)  →  Artist (any form)
  0 MB weights       3.57 MB weights       500 MB weights      2 GB weights
  Markov chain       NanoGPT char-level    TinyLlama          Llama 3.2 3B
  Pure emergence     Tiny weights          Small weights      Full weights
  Maximum constraint Structured poetry     Free-form text     Multimodal
```

## Autonomous Design

**Sonnet is fully autonomous:**
- Own requirements.txt
- Own state/ directory (database, weights)
- Own scripts/ (weight conversion)
- Own README
- Bridge launcher in root (sonnet_run.py)

**No HAiKU dependency:**
- Does NOT import from haiku/
- Does NOT share database
- Does NOT share vocabulary

**Cascade mode:**
- Only connects through MetaHarmonix
- User → HAiKU → Sonnet → Prose → Artist
- Each passes output to next
- MetaHarmonix orchestrates flow

## Next: Prose

After Sonnet completes, next ипостась:
- **Prose**: TinyLlama (500 MB)
- Free-form text generation
- Long-form responses
- Bridge between poetry and full LLM
