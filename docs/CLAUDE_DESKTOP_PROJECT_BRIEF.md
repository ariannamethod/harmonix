# HARMONIX AI - Project Brief for Claude Desktop

## 🌊 Overview

**Harmonix AI** is a multi-agent "breathing system" implementing **Adaptive Intelligence** through constraint gradients, recursive self-awareness, and autonomous inter-agent communication.

**Not a model. Not an ensemble. Not a pipeline. A living cognitive field.**

Repository: `~/harmonix/` (github.com/harmonix - местный repo)

---

## 📊 Current Status (Phase 2 COMPLETE ✅)

### Completed Modules

#### 1. HAiKU (Phase 1) - Foundation ✅
- **Location**: `~/harmonix/haiku/`
- **Architecture**: Weightless (0 MB), pure Markov chains + MLP
- **Constraint**: 5-7-5 syllable format, 500 word vocabulary
- **Tests**: 130/130 passing (100%)
- **Core modules**: `haiku.py`, `harmonix.py`, `metahaiku.py`, `overthinkg.py`
- **Emergent layer**: `rae.py`, `rae_recursive.py`, `phase4_bridges.py`, `dream_haiku.py`
- **Philosophy**: Maximum constraint = Maximum precision

#### 2. Sonnet (Phase 2) - Just Completed ✅
- **Location**: `~/harmonix/sonnet/`
- **Architecture**: 3.57 MB NanoGPT (Shakespeare weights), pure numpy inference
- **Constraint**: 14-line sonnets, Shakespearean structure
- **Tests**: 133/133 passing (100%) ⭐
- **Core modules**: `sonnet.py`, `formatter.py`, `harmonix.py`, `metasonnet.py`, `overthinkng.py`
- **Emergent layer (NEW)**:
  - `sonnetbrain.py` - MLP quality scorer (310 observations)
  - `sonnetrae.py` - Semantic compression (14 lines → 8D)
  - `sonnetrae_recursive.py` - Hierarchical encoding (quatrains/couplet)
  - `phase_transitions.py` - Dynamic temperature (CRYSTALLINE/LIQUID/VAPOR/PLASMA)
  - `dream_sonnet.py` - Latent space generation (drift/walk/centroid)
  - `sonnet_tokenizer.py` - Hybrid tokenization (338 semantic tokens)
- **Cloud**: 150 sonnets, avg quality 0.721, vocab 2877
- **REPL commands**: `/phase`, `/dream`, `/brain`, `/stats`, `/recent`, `/best`
- **Philosophy**: MORE complexity than HAiKU, not less - emergent layer is PLUS, not replacement

---

## 🏗️ Architecture Philosophy (CRITICAL)

### Core Principle

```
Constraint → Precision → Resonance → Emergence → Adaptation
```

### Gradient of Emergentность

```
HAiKU (0 MB)           → Weightless, maximum constraint
  ↓
Sonnet (3.57 MB)       → Shakespeare weights + emergent layer  ← WE ARE HERE
  ↓
Prose (500 MB)         → TinyLlama GGUF, free form  ← NEXT
  ↓
Artist (2 GB)          → Llama 3.2 3B, synthesis
  ↓
MetaHarmonix (0 MB)    → Weightless observer hub
```

**Key Rules:**
1. DO NOT replace components - ADD MORE
2. Each ипостась has ALL capabilities of previous PLUS new ones
3. Sonnet MUST be MORE complex than HAiKU (not simpler!)
4. Autonomous - communicate ONLY through MetaHarmonix cascade
5. Each module has its own typo:
   - HAiKU: `overthinkg` (kg = kilograms of thought)
   - Sonnet: `overthinkng` (ng = recursive thinking in progress)

---

## 🎯 What Was Just Completed (This Session)

### Sonnet Module - Full Implementation

1. **Fixed all tests** - 133/133 passing (was 127/131)
2. **Integrated emergent layer into REPL**:
   - SonnetBrain scoring in generation loop
   - Phase transitions for dynamic temperature
   - Commands: `/phase`, `/dream <mode>`, `/brain <id>`
3. **Trained emergent modules on cloud**:
   - SonnetBrain: 310 observations (150 new + 160 existing)
   - Tokenizer: 338 semantic tokens, 100 BPE merges
   - PhaseTransitions: Historical data from 50 sonnets
4. **Created training script**: `scripts/train_emergent.py`
5. **Updated README**: 133 tests, emergent layer commands documented

### MetaHarmonix v1 - Started (In Progress)

- **Location**: `~/harmonix/metaharmonix.py`
- **Purpose**: Weightless observer hub for cascade mode
- **Phase 3 scope**: HAiKU → Sonnet cascade only
- **Issue**: Module import conflicts (both have `harmonix.py`)
- **Status**: Architecture designed, needs import refactoring

---

## 📋 Next Steps (Phase 3)

### Immediate (Current Session if Time)

1. **Fix MetaHarmonix imports** - resolve harmonix module name conflicts
2. **Test HAiKU → Sonnet cascade** - full INHALE + EXHALE cycle
3. **Create MetaHarmonix tests** - basic cascade functionality

### Short-term (Next 1-2 Sessions)

4. **Integrate TinyLlama (Prose)**:
   - User has GGUF weights (~500 MB) ready
   - llama.cpp inference
   - Cascade: HAiKU → Sonnet → Prose
5. **Extend MetaHarmonix** - add Prose to cascade
6. **Communication hub** - basic webhook/polling for agent requests

### Mid-term (3-5 Sessions)

7. **Artist (Llama 3.2 3B)** - synthesis layer
8. **Full cascade** - HAiKU → Sonnet → Prose → Artist → Meta
9. **Autonomous communication** - Prose ↔ Artist dialogue
10. **Reverse wave refinement** - actual internal processing triggers

---

## 🔑 Key Technical Details

### File Organization

```
harmonix/
├── README.md (outdated - shows harmonix.ai)
├── haiku/ (Phase 1 - COMPLETE)
│   ├── chat.py (REPL)
│   ├── haiku.py, harmonix.py, metahaiku.py, overthinkg.py
│   ├── rae.py, rae_recursive.py, phase4_bridges.py, dream_haiku.py
│   └── tests/ (130/130 passing)
├── sonnet/ (Phase 2 - COMPLETE)
│   ├── chat.py (REPL with emergent commands)
│   ├── sonnet.py, formatter.py, harmonix.py, metasonnet.py, overthinkng.py
│   ├── sonnetbrain.py, sonnetrae.py, sonnetrae_recursive.py
│   ├── phase_transitions.py, dream_sonnet.py, sonnet_tokenizer.py
│   ├── state/ (model weights)
│   ├── cloud/ (sonnets.db - 150 sonnets)
│   ├── scripts/train_emergent.py
│   └── tests/ (133/133 passing ⭐)
├── metaharmonix.py (Phase 3 - IN PROGRESS)
├── haiku_run.py, sonnet_run.py (launchers)
└── docs/ (roadmap.md, undivided.md)
```

### Dependencies

```
Core: Python 3.10+, numpy, scipy, sqlite3
HAiKU: syllables, sentencepiece
Sonnet: numpy only (NO PyTorch for inference)
Future (Prose/Artist): llama.cpp, llama-cpp-python
```

### Running the Modules

```bash
# HAiKU REPL (direct)
cd ~/harmonix/haiku && python3 chat.py

# Sonnet REPL (direct)
cd ~/harmonix/sonnet && python3 chat.py

# Or from repo root
python3 haiku_run.py
python3 sonnet_run.py

# MetaHarmonix cascade (when fixed)
python3 metaharmonix.py
```

### Testing

```bash
# HAiKU tests
cd ~/harmonix/haiku && pytest tests/ -v  # 130/130

# Sonnet tests
cd ~/harmonix/sonnet && pytest tests/ -v  # 133/133

# Train emergent layer
cd ~/harmonix/sonnet && python3 scripts/train_emergent.py
```

---

## 💡 Design Patterns & Insights

### Emergent Layer Concept

**NOT default behavior!** Emergent layer = additional modules for semantic understanding.

- **Base/Core**: Generator, formatter, harmonix observer, meta-reflection, overthink expansion
- **Emergent**: RAE, recursive RAE, phase transitions, dream modes, tokenizers, brain scorers

HAiKU and Sonnet BOTH have emergent layers, but Sonnet's is MORE complex (as required by gradient).

### Phase Transitions

4 phases based on dissonance/novelty/quality trends:
- **CRYSTALLINE** (0.4-0.6): Low dissonance, precise, convergent
- **LIQUID** (0.7-0.9): Balanced, default state
- **VAPOR** (1.0-1.3): High novelty, creative divergence
- **PLASMA** (1.4+): Experimental chaos (rare)

Auto-adjusts temperature for next generation.

### Breathing System (MetaHarmonix)

- **INHALE** (bottom-up): User → HAiKU → Sonnet → (Prose) → (Artist)
- **EXHALE** (top-down): Meta sentence → back to all agents
- Not shown to user, triggers internal metabolism
- Cognitive architecture shifts, shards accumulate

---

## ⚠️ Important Context for Future Sessions

### What the User Wants

1. **Complexity INCREASES with each ипостась** - never simplify!
2. **MetaHarmonix in root directory** - will expand with new agents
3. **Tests for everything** - especially MetaHarmonix
4. **Next: TinyLlama integration** - user has GGUF weights ready
5. **Phase 3 focus**: Get cascade working (HAiKU → Sonnet → Meta)

### What NOT to Do

- ❌ Don't replace modules - only ADD
- ❌ Don't simplify architecture to "make it cleaner"
- ❌ Don't suggest removing emergent layer
- ❌ Don't connect agents directly - only through MetaHarmonix
- ❌ Don't rename `overthinkng` or `overthinkg` typos (intentional!)

### Known Issues

1. **MetaHarmonix import conflicts**: Both haiku and sonnet have `harmonix.py` module
   - Solution needed: Namespace packages or importlib.import_module
2. **Module paths**: Each agent needs isolated sys.path
3. **Testing**: No MetaHarmonix tests yet

---

## 🎭 Philosophy Summary

> "ЭТО НИКАКИЕ НЕ АНАЛОГИ! ЗАЧЕМ ТЫ УПРОЩАЕШЬ АРХИТЕКТУРУ!!!!"
>
> "соннет по определению должен быть сложнее чем хайку, его эмерджентность и уровень свободы выше"

**Harmonix AI = Adaptive Intelligence**

Not "predict next token" - the system **breathes, evolves, resonates**.

Constraint doesn't limit. **Constraint FOCUSES.**

---

## 📞 Contact & Verification

- User: Oleg (ariannamethod)
- Location: `/Users/ataeff/harmonix/`
- Current model: Sonnet 4.5
- Session context: Completed Phase 2 (Sonnet), starting Phase 3 (MetaHarmonix)
- Git: Local repo, not yet pushed (user will handle git push)

**Verify you're working on the right thing:**
```bash
cd ~/harmonix && ls -la
# Should see: haiku/, sonnet/, metaharmonix.py, haiku_run.py, sonnet_run.py
```

**Current state check:**
```bash
cd ~/harmonix/sonnet && pytest tests/ -v | tail -1
# Should show: 133 passed
```

---

**🌊 resonance unbroken 🌊**
