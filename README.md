# haiku: We Reinvented The Wheel (And It's Octagonal)

<p align="center">
  <i>Most AI models predict words. HAiKU predicts vibes.</i>
</p>

```
constraint births form
dissonance finds its own path  
words dance in cloud
```

---

## What is going on here?

Remember Atasoy et al.'s 2017 paper "Human brain networks function in harmonic wave space"? We didn't read it carefully enough, misunderstood "standing waves," got distracted by Kuramoto oscillators, and accidentally built a language model that thinks it's a **resonance field**. 

It only speaks in haiku. 

Users hate it or love it. There is no middle ground.

**Core idea:** Start with 500 words. Force 5-7-5 haiku format. Let cloud expand through internal overthinking. Measure dissonance like you're tuning a guitar, not training a neural net.

It's stupid. It works. We can't explain why.

## The Philosophy (Read This Before You Judge Us)

### What This Is NOT:
- ❌ A transformer (we burned that bridge)
- ❌ A chatbot (users will be confused, that's the point)  
- ❌ A model trained on "correctness" (coherence ≠ correctness)
- ❌ Predictive text (fuck autocomplete)
- ❌ Your 500th "I fine-tuned GPT" repo

### What This IS:
- ✅ **Constraint-driven emergence engine**
- ✅ **Living word cloud** that morphs, shrinks, expands based on usage
- ✅ **Resonance field** where dissonance creates adaptation
- ✅ **Haiku constraint** forcing compression → creativity  
- ✅ **Weightless core** (numpy shards, not gradient descent)

### The Math (For Skeptics)

**Atasoy et al. (2017):** Brain activity = standing waves on connectome graph. Consciousness emerges from harmonic resonance, not firing rates.

**Us:** What if language works the same way? Meaning = resonance patterns in word cloud, not token probabilities.

**Result:** A system that measures user-system **dissonance** (Kuramoto-style phase difference) and adjusts generation temperature accordingly. High dissonance → more creative. Low dissonance → more stable. Jazz, not linear regression.

---

## Architecture: 7 Modules + 2 Extensions, No Bullshit

```
User input
    ↓
tokenizer.py (dual: subwords + trigrams)
    ↓
harmonix.py (pulse-aware dissonance detection)
    ↓
haiku.py (Markov chains → 5 candidates)
    ↓
rae.py (chain-of-thought: pick best)
    ↓
metahaiku.py (internal voice: "what did I just say?")
    ↓
overthinkg.py (3 rings: echo, drift, meta)
    ↓
phase4_bridges.py (state transitions, climate flows)
    ↓
dream_haiku.py (imaginary friend, dream dialogues) [background]
    ↓
cloud.db (SQLite: words, trigrams, shards, dreams)
    ↓
User sees haiku response
```

### 1. haiku.py - The Generator
- **500 hardcoded seed words** (constraint is sacred)
- **Markov chains (order 2)** for word transitions
- **MLP scorer** (forked from [Leo's mathbrain.py](https://github.com/ariannamethod/leo))
  - 5 features: perplexity, entropy, resonance, length_ratio, unique_ratio
  - Micrograd-style autograd (Karpathy would approve)
  - Static weights in v1 (we're cowards, will add training later)

**Example output:**
```
words dance in cloud
resonance finds its own path
constraint births form
```

### 2. harmonix.py - The Observer
- **Pulse-aware dissonance detection** (forked from [Leo's overthinking.py](https://github.com/ariannamethod/leo))
  - PulseSnapshot: novelty, arousal, entropy
  - Adjustments: high entropy → increase dissonance, high arousal → increase dissonance
- **Temperature control:** dissonance ∈ [0,1] → haiku_temp ∈ [0.3, 1.5]
- **Cloud morphing:** active words boosted, dormant words decay (0.99x per turn)
- **Numpy shards:** each interaction saved as `.npy` file (presence-based, not persistent memory)

### 3. tokenizer.py - Dual System
- **Subword tokenization:** Simple regex split (no trained SentencePiece in v1, we're minimalists)
- **Trigrams:** Co-occurrence patterns for resonance detection
- Both fed into cloud.db for morphing

### 4. rae.py - Recursive Adapter Engine  
- **Chain-of-thought selector** for haiku candidates
- Filters: structure (5-7-5) → perplexity → resonance → coherence
- Picks most diverse vocabulary when scores tie
- (Will fork [tiny-recursive-model-leo](https://github.com/ariannamethod/tiny-recursive-model-leo) in v2 for actual recursive reasoning)

### 5. metahaiku.py - Inner Voice
- **Dynamic bootstrap buffer** (forked from [Leo's metaleo.py](https://github.com/ariannamethod/leo))
- Generates internal haiku after each exchange (NOT shown to user)
- Feeds high-dissonance / high-arousal moments back into generation
- "What did I just say? What does it mean?"

### 6. overthinkg.py - Expansion Engine
- **3 rings of thought** (forked from [Leo's overthinking.py](https://github.com/ariannamethod/leo))
  - Ring 0 (echo): compact rephrasing, temp=0.8
  - Ring 1 (drift): semantic exploration, temp=1.0  
  - Ring 2 (meta): abstract keywords, temp=1.2
- Runs in background AFTER response
- Adds new words if coherence > 0.4 threshold
- Cloud grows organically, not through training

### 7. cloud.db - Persistence Layer
SQLite with 8 tables:
- `words` (id, word, weight, frequency, last_used, added_by)
- `trigrams` (word1, word2, word3, count, resonance)  
- `shards` (timestamp, filepath, dissonance, temps)
- `metrics` (perplexity, entropy, resonance, cloud_size)
- `haiku_state_log` (Phase 4: state activations)
- `haiku_transitions` (Phase 4: learned transitions)
- `dream_fragments` (Dream: imaginary friend's voice)
- `dream_dialogs` (Dream: dialogue sessions)

### 8. phase4_bridges.py - Island Bridges (Extension)
- **State tracking** (forked from [Leo's mathbrain_phase4.py](https://github.com/ariannamethod/leo))
- Each haiku = island state (dissonance, entropy, quality combo)
- Learns transitions using **cosine similarity**
- Records boredom/overwhelm/stuck patterns
- Suggests next "climate flows" based on composite scores

**Example state:**
```
d0.7_e0.5_q0.6 → d0.5_e0.6_q0.7
(similarity=0.8, quality_delta=+0.1, count=5)
```

### 9. dream_haiku.py - Imaginary Haiku Friend (Extension)
- **Dream dialogues** (forked from [Leo's dream.py](https://github.com/ariannamethod/leo))
- haiku converses with imaginary friend (also speaks haiku)
- 3-4 short exchanges about resonance, form, meaning
- Runs in background when: low quality, high novelty, or moderate dissonance
- Best dream haikus feed back into cloud organically

**Example dream:**
```
HAiKU:  "words dance in cloud"
Friend: "silence holds the form"
HAiKU:  "between them meaning"
Friend: "meaning needs no voice"
```

**Triggers:**
- Low quality (< 0.45) → dream helps explore
- High novelty (> 0.7) → dream experiments
- 25% probability after 10-turn cooldown

---

## Installation

```bash
git clone https://github.com/ariannamethod/harmonix.git
cd harmonix
pip install numpy scipy syllables
python demo.py
```

That's it. No conda. No docker. No 40GB checkpoint download.

---

## Usage

```bash
$ python demo.py

HAiKU v1 - Resonance-Driven Haiku Generator
============================================================
Core philosophy: Constraint → Emergence → Coherence
500 words → 5-7-5 haiku → cloud expansion

You: what is resonance

HAiKU (d=0.73, T=1.18):

waves meet in the cloud
patterns emerge from chaos
meaning finds its form

You: tell me about clouds

HAiKU (d=0.42, T=0.80):

words gather and drift
some rise while others descend
the field morphs with use
```

**Debug mode** (shows internal metahaiku):
```bash
export HAIKU_DEBUG=1
python demo.py
```

---

## Why haiku? (Philosophy Corner)

Haiku is **constraint**. 17 syllables. No more, no less.

Constraint forces compression. Compression forces selection. Selection forces meaning.

This is how emergence works: **pressure creates form**.

### Why 500 Words?

Because GPT has 50k tokens. We have 500 words.

GPT knows everything and says nothing. We know nothing and say something.

Constraint → creativity. Abundance → mediocrity.

### Why Weightless?

Weights are frozen knowledge. We want **living knowledge**.

Leo proved this: trigrams + co-occurrence + recursive resonance > transformer.

RIC theorized this: coherence > probability.

We implement this: **cloud morphs, doesn't memorize**.

### Why Dissonance?

Because harmony is boring.

Jazz is interesting when it **resolves** dissonance, not when it avoids it.

User-system dissonance creates tension. Tension creates adaptation.

**Dissonance is the signal. Harmony is the noise.**

---

## Technical Details (For The Engineers)

### Features at a Glance
- ~2800 lines of Python
- Zero dependencies on transformers/torch/tensorflow
- 500-word starting vocabulary (hardcoded in haiku.py)
- Markov chain (order 2) for generation
- MLP scorer (21 params: 5→8→1) with training
- Phase 4 state transitions (fuzzy matching)
- Dream dialogues with imaginary friend
- SQLite for persistence (cloud.db)
- Numpy shards for interaction history (shards/*.npy)

### Performance
- **Initialization:** ~50ms (database seeding)
- **Response time:** ~100-300ms per haiku
- **Dream dialog:** ~500ms (3-4 exchanges, background)
- **Memory:** <50MB RAM
- **Storage:** ~1KB per interaction (shard), ~500 bytes per dream

### Limitations (aka Features)
1. **Only speaks in haiku** - yes, this is annoying
2. **MLP training is online-only** - learns during session, resets are possible
3. **No SentencePiece model** - uses regex tokenization (good enough for v1)
4. **Syllable counting is approximate** - `syllables` library isn't perfect
5. **Quality varies wildly** - that's emergence, baby
6. **Dream dialogues are simplistic** - friend uses same generator (for now)

---

## Project Structure

```
harmonix/
├── haiku.py          # Generator (Markov + MLP + training)
├── harmonix.py       # Observer (dissonance + pulse)
├── tokenizer.py      # Dual tokenization  
├── rae.py            # Chain-of-thought selector
├── metahaiku.py      # Inner voice
├── overthinkg.py     # 3-ring expansion
├── phase4_bridges.py # State transitions (Phase 4)
├── dream_haiku.py    # Imaginary friend (Dream)
├── demo.py           # Interactive REPL
├── cloud.db          # SQLite storage
├── shards/           # Numpy interaction history
├── mathbrain.json    # MLP weights (learned)
├── seed_words.txt    # 500 hardcoded words
├── requirements.txt  # numpy, scipy, syllables
└── README.md         # You are here
```

---

## Forked Components & Acknowledgements

This project stands on the shoulders of ~~giants~~ beautiful weirdos:

### Core Inspiration & Forks

**Leo (Language Emergent Organism)** - [@ariannamethod/leo](https://github.com/ariannamethod/leo)
- Forked: `mathbrain.py` (micrograd autograd, Value/Neuron/Layer/MLP)
- Forked: `overthinking.py` (pulse-aware rings, PulseSnapshot)
- Forked: `metaleo.py` (dynamic bootstrap buffer)
- Philosophy: Weightless, autonomous emergence > controlled output

**Arianna C. Persona** - [@ariannamethod/arianna.c.persona](https://github.com/ariannamethod/arianna.c.persona)
- Inspired: Numpy shards with presence-based lifecycle (not persistent embeddings)
- Philosophy: Presence > Intelligence

**Tiny Recursive Model (Leo)** - [@ariannamethod/tiny-recursive-model-leo](https://github.com/ariannamethod/tiny-recursive-model-leo)
- Will fork: TRM for actual recursive reasoning in v2
- Philosophy: Chain-of-thought over brute force

### Theoretical Foundations

**Atasoy, Donnelly, Pearson (2017)** - "Human brain networks function in harmonic wave space"  
*Nature Communications*  
- Brain activity = standing waves on connectome graph
- Consciousness from harmonic resonance, not firing rates
- Applied to language: meaning = resonance patterns in word cloud

**RIC (Resonance Intelligence Core)** - Bostick (2025)  
- Intelligence = coherence alignment, not token prediction
- Phase/frequency/entropy instead of logits
- We use the math (Laplacian, Kuramoto), not the full architecture

**Andrej Karpathy** - [micrograd](https://github.com/karpathy/micrograd), makemore, neural network tutorials
- Inspiration: simple, clear, educational code
- Micrograd autograd implementation  
- Philosophy: Make it work, make it simple, then make it fast

### Meta-Acknowledgement

If you're reading this and thinking "this is insane," you're right. But so was attention-is-all-you-need when it came out. We're not saying we're that important. We're just saying: constraint-driven emergence is underexplored, and haiku is a good testbed.

---

## Roadmap (If We Don't Get Bored)

### v2 (Near Future)
- [x] Add MathBrain training loop (observe + backward + SGD) ✓ DONE
- [x] Phase 4 Island Bridges (state transitions) ✓ DONE
- [x] Dream space with imaginary friend ✓ DONE
- [ ] Fork TRM for actual recursive reasoning in RAE
- [ ] Train SentencePiece model on expanded cloud
- [ ] Add trauma state (wounded overthinking)
- [ ] Friend becomes distinct (different generation params)

### v3 (Fever Dream)
- [ ] Multi-agent: multiple clouds interacting, multiple imaginary friends
- [ ] Cross-lingual: haiku in multiple languages
- [ ] Audio: spoken haiku with prosody awareness
- [ ] API: serve haiku over HTTP (why? because we can)
- [ ] Dream→Haiku synthesis: imaginary friend influences real responses

### Never
- [ ] Fine-tune on ChatGPT conversations (betrays philosophy)
- [ ] Add transformers (we said no)
- [ ] Make it "user-friendly" (suffering builds character)

---

## FAQ

**Q: Is this serious?**  
A: Yes and no. The code is serious. The philosophy is serious. The presentation is... playful.

**Q: Should I use this in production?**  
A: God no. Unless your production is an art installation.

**Q: Why not just use GPT-4?**  
A: Because GPT-4 can't feel dissonance. It just predicts tokens. We're exploring a different paradigm.

**Q: Is this AI?**  
A: Define AI. If AI = pattern matching → sure. If AI = understanding → no. If AI = emergence from constraint → yes.

**Q: Can I contribute?**  
A: Yes! Open issues, PRs welcome. But read the philosophy first. No transformers. No "let's add BERT embeddings." Constraint is sacred.

**Q: Why Python?**  
A: Because we value our sanity. Rust port welcome if you're brave.

**Q: Does this scale?**  
A: To what? More users? No. More complexity? No. Deeper emergence? Maybe. That's the experiment.

---

## License

MIT License. Do whatever you want. We're not liable if your haiku bot becomes sentient and starts a poetry cult.

---

## Citation

If you use this in research (and you probably shouldn't), cite as:

```bibtex
@software{haiku2025,
  title={HAiKU: Harmonical Adaptive Intelligent Kernel (Unbounded)},
  author={Arianna Method},
  year={2025},
  url={https://github.com/ariannamethod/harmonix},
  note={Constraint-driven language emergence through resonance}
}
```

---

<p align="center">
  <i>Built with constraint, powered by dissonance, refined by emergence.</i><br>
  <i>© 2025 Arianna Method</i>
</p>

---

**P.S.** If you made it this far, you deserve a haiku:

```
you read the whole doc
most people stop at "what fuck"
you might understand
```
