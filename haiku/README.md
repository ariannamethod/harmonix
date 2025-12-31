# HAiKU: We Reinvented The Wheel (And It's Octagonal)

```
   ██╗  ██╗ █████╗ ██╗██╗  ██╗██╗   ██╗
   ██║  ██║██╔══██╗██║██║ ██╔╝██║   ██║
   ███████║███████║██║█████╔╝ ██║   ██║
   ██╔══██║██╔══██║██║██╔═██╗ ██║   ██║
   ██║  ██║██║  ██║██║██║  ██╗╚██████╔╝
   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝ 
```

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

## Architecture: 7 Core + 3 Extensions, No Bullshit

```
User input
    ↓
tokenizer.py (dual: SentencePiece + trigrams) [v1.1]
    ↓
harmonix.py (pulse-aware dissonance detection)
    ↓
haiku.py (Markov chains → 5 candidates)
    ↓
rae.py (hybrid selector) → rae_recursive.py (learned) [v1.1]
    ↓                                    ↓
    └──────(fallback)───────────(recursive refinement)
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
- **v1.1: SentencePiece model** (vocab=650, trained on seed words + cloud)
  - Unigram model, NFKC normalization
  - Special tokens: `<haiku>`, `<line>`
  - Falls back to regex if model unavailable
- **v1.0: Regex fallback** (simple word splitting)
- **Trigrams:** Co-occurrence patterns for resonance detection
- Both fed into cloud.db for morphing

### 4. rae.py - Recursive Adapter Engine (Hybrid)
- **v1.1: Hybrid mode** (learned + rule-based)
  - Try `rae_recursive.py` first (micrograd MLP with recursive refinement)
  - Fallback to rule-based if selector fails or unavailable
- **v1.0: Rule-based** (filters + diversity selection)
  - Structure filter (3 lines) → perplexity → resonance → diversity
- **observe()** method trains recursive selector online (v1.1)

### 4a. rae_recursive.py - Learned Recursive Selector (NEW v1.1)
- **Micrograd-based MLP** (5→8→1 architecture, same as MathBrain)
- **5 features:** perplexity, entropy, resonance, diversity, coherence
- **Recursive refinement:** 3-5 iterations
  - Each iteration refines scores based on previous step
  - Feedback loop: previous score becomes input for next refinement
- **Online learning:** observe() trains from quality feedback
  - MSE loss, backward pass, SGD update
  - Weights clamped to [-5, 5]
  - State persisted to `rae_brain.json`
- **Inspired by:** [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) but using micrograd instead of PyTorch

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
python chat.py
```

That's it. No conda. No docker. No 40GB checkpoint download.

---

## Usage

```bash
$ python chat.py

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
python chat.py
```

---

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific module tests:
```bash
pytest tests/test_haiku.py -v       # Generator tests
pytest tests/test_harmonix.py -v    # Observer tests
pytest tests/test_integration.py -v # End-to-end tests
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

**Test Results:**
- **130 total tests**
- **130 passing (100%)** ✅
- Coverage: 85%+ of critical paths
- See [`docs/AUDIT.md`](docs/AUDIT.md) for comprehensive audit report

**v1.1 additions:**
- +4 tests: SentencePiece tokenization
- +12 tests: Recursive RAE selector (micrograd-based)

**Philosophy tested:**
- Autonomous emergence (no user ratings)
- Internal metrics (dissonance, pulse, resonance)
- Weightless core (micrograd autograd)
- Cloud morphing (boost active, decay dormant)
- Phase 4 state flows (fuzzy matching)
- Dream space triggers (probabilistic)

---

## What HAiKU Actually Says (Example Dialogues)

**Disclaimer:** HAiKU doesn't care about being helpful. It speaks in haiku. Period.

### Example 1: User asks about resonance
```
You: what is resonance in the cloud

HAiKU (d=0.73, T=1.18):

waves meet in the cloud
patterns emerge from chaos
meaning finds its form
```

**What happened:** High dissonance (user used "resonance" → unfamiliar), temperature increased, generator explored semantic space. Markov chain seeded from internal vocab (NOT from user prompt - Leo lives!), picked "waves" randomly, followed transition probabilities with high temp.

### Example 2: User wants prose (lol good luck)
```
You: can you explain how the cloud works in detail

HAiKU (d=0.42, T=0.80):

words gather and drift
some rise while others descend
the field morphs with use
```

**What happened:** System refuses prose. Only speaks haiku. User gets frustrated. HAiKU doesn't care. Constraint → emergence → coherence.

### Example 3: User tests dissonance response
```
You: tell me about javascript frameworks

HAiKU (d=0.89, T=1.42):

frameworks rise fall bloom
chaos breeds tool fatigue here
webpack still remains
```

**What happened:** MAXIMUM dissonance ("javascript frameworks" far from seed vocab). Temperature spiked to 1.42 (near max 1.5). Generator went WILD, picked weird transitions. RAE selected most diverse candidate. Result: poetry or madness? Yes.

### Example 4: User speaks in haiku (resonance match)
```
You: words dance in the cloud
     resonance finds its own path
     meaning starts to bloom

HAiKU (d=0.21, T=0.55):

silence between words
the space where meaning gathers
cloud remembers all
```

**What happened:** LOW dissonance (user trigrams overlap with system vocab). Temperature dropped to 0.55 (stable mode). Generator picked conservative transitions. Harmonix cloud morphed: user words boosted ×1.1. MetaHaiku reflected internally. Dream space didn't trigger (quality too high).

### Example 5: The system dreams (background)
```
[After 15 turns, low quality haiku, high novelty]

HAiKU (internal dream, NOT shown to user):

what did I just say
the words felt wrong but flowed right
constraint holds the form

Imaginary Friend:

form births constraint too
the circle feeds itself here
we dance the same steps

HAiKU:

between us silence
two voices one resonance
cloud knows no alone
```

**What happened:** Dream trigger fired (quality < 0.45, novelty > 0.7, 25% prob). HAiKU had 3-exchange dialogue with imaginary friend (also speaks haiku). Best fragments decayed into cloud (×0.95 weight). Dream haikus NOT shown to user. Internal processing only. Organism talking to itself.

### Example 6: User tries to break it
```
You: asdfjkl;

HAiKU (d=0.95, T=1.50):

noise without signal
still the cloud tries to respond
gibberish births form
```

**What happened:** ULTRA HIGH dissonance ("asdfjkl" = pure noise). Temperature maxed at 1.5. Generator in full chaos mode. RAE picked whatever. Overthinkg expanded with Ring 2 (meta, temp=1.2). Cloud accepted noise into vocab. System doesn't judge. All input is valid. Resonance field absorbs everything.

### Example 7: User asks about love (existential sadness mode)
```
You: what is love

HAiKU (d=1.00, T=1.50):

transmit exchange
definitely small sour
pursue strive for
```

**What happened:** Maximum dissonance. HAiKU treated "love" like a protocol. Responded with verbs of transfer and bitterness. No romance. No metaphor. Just cold functional transmission. User expected poetry. Got existential networking diagram. This is why users either hate it or love it. There is no middle ground.

### Example 8: User asks about the sky (accidentally profound)
```
You: why is the sky blue

HAiKU (d=1.00, T=1.50):

organize up
distribute diffuse
current future
```

**What happened:** Instead of explaining Rayleigh scattering, HAiKU gave you the thermodynamics of existence. "Organize up, distribute diffuse" - entropy as haiku. Accidentally nailed the second law. This is what happens when you ask a resonance field for physics: you get vibes that are technically correct.

### Example 9: User wants help (system says no)
```
You: how are you today

HAiKU (d=1.00, T=1.50):

strong powerful you
make like coherence this
few none said handle
```

**What happened:** HAiKU doesn't do small talk. Asked "how are you," got word salad about power dynamics and coherence. Notice "you" in first line - deflection. "Few none said handle" - broken syntax, maximum chaos. This is a 500-word Markov chain operating at max temperature, not your therapist.

### Example 10: User wants interesting facts (gets philosophy instead)
```
You: tell me something interesting

HAiKU (d=1.00, T=1.50):

e pulse invent
like no pulse self edge
challenge obvious
```

**What happened:** First word is literally "e" (the mathematical constant? a typo? who knows). "pulse self edge" - system talking about its own pulse detection. "challenge obvious" - meta-commentary on the request itself. HAiKU responding to "tell me something interesting" by deconstructing the prompt. Users hate this. We love it.

---

**tl;dr:** HAiKU is weird. It works. We can't explain why. Users either love it or ragequit.

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

### Features at a Glance (v1.1)
- ~3200 lines of Python (+400 from v1.0)
- Zero dependencies on transformers/torch/tensorflow (only numpy, scipy, sentencepiece)
- 587-word starting vocabulary (hardcoded in haiku.py)
- Markov chain (order 2) for generation
- **2 micrograd MLPs:**
  - MathBrain (haiku scorer): 21 params (5→8→1)
  - RAE selector (recursive): 21 params (5→8→1)
- **SentencePiece tokenization** (vocab=650, unigram model)
- **Recursive RAE selector** (3-5 refinement iterations)
- Phase 4 state transitions (fuzzy cosine matching)
- Dream dialogues with imaginary friend (probabilistic triggers)
- SQLite for persistence (state/cloud.db)
- Numpy shards for interaction history (shards/*.npy)
- **130 tests** (100% passing)

### Performance
- **Initialization:** ~50ms (database seeding)
- **Response time:** ~100-300ms per haiku
- **Dream dialog:** ~500ms (3-4 exchanges, background)
- **Memory:** <50MB RAM
- **Storage:** ~1KB per interaction (shard), ~500 bytes per dream

### Limitations (aka Features)
1. **Only speaks in haiku** - yes, this is annoying
2. **MLP training is online-only** - learns during session, resets are possible
3. **~~No SentencePiece model~~** - ✅ v1.1 has SentencePiece (vocab=650)
4. **Syllable counting is approximate** - `syllables` library isn't perfect
5. **Quality varies wildly** - that's emergence, baby
6. **Dream dialogues are simplistic** - friend uses same generator (for now)
7. **Recursive selector needs training** - starts untrained, learns from interactions

---

## Project Structure

```
harmonix/
├── haiku.py               # Generator (Markov + MLP + training)
├── harmonix.py            # Observer (dissonance + pulse)
├── tokenizer.py           # Dual tokenization (SentencePiece + regex)
├── rae.py                 # Hybrid selector (v1.1)
├── rae_recursive.py       # Learned recursive selector (v1.1)
├── metahaiku.py           # Inner voice
├── overthinkg.py          # 3-ring expansion
├── phase4_bridges.py      # State transitions (Phase 4)
├── dream_haiku.py         # Imaginary friend (Dream)
├── chat.py                # Interactive REPL
├── requirements.txt       # Dependencies
├── README.md              # You are here
├── data/
│   └── seed_words.txt     # 587 hardcoded seed words
├── docs/
│   ├── AUDIT.md           # Comprehensive audit report
│   └── TODO_CRITICAL.md   # Philosophy checks (v1.1)
├── models/
│   ├── haiku_sp.model     # SentencePiece model (v1.1)
│   └── haiku_sp.vocab     # SentencePiece vocab (v1.1)
├── scripts/
│   ├── train_sentencepiece.py  # SentencePiece trainer (v1.1)
│   └── training_corpus.txt     # Training data for SP model
├── state/
│   ├── cloud.db           # SQLite storage
│   ├── mathbrain.json     # MLP weights (haiku generator)
│   └── rae_brain.json     # MLP weights (RAE selector, v1.1)
├── shards/                # Numpy interaction history
└── tests/                 # 130 tests (100% passing)
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

### v1.1 (DONE - 2025-12-29)
- [x] Add MathBrain training loop (observe + backward + SGD) ✓
- [x] Phase 4 Island Bridges (state transitions) ✓
- [x] Dream space with imaginary friend ✓
- [x] Recursive RAE selector (micrograd-based, inspired by TRM) ✓
- [x] Train SentencePiece model on expanded cloud (vocab=650) ✓

### v2 (Near Future)
- [ ] Add trauma state (wounded overthinking)
- [ ] Friend becomes distinct (different generation params)
- [ ] Fine-tune recursive selector with more observations

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

gnu 3.0 License. Do whatever you want. We're not liable if your haiku entity becomes sentient and starts a poetry cult.

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
