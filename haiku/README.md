```
██╗  ██╗ █████╗ ██╗██╗  ██╗██╗   ██╗
██║  ██║██╔══██╗██║██║ ██╔╝██║   ██║
███████║███████║██║█████╔╝ ██║   ██║
██╔══██║██╔══██║██║██╔═██╗ ██║   ██║
██║  ██║██║  ██║██║██║  ██╗╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝
```

# haiku — when constraint became consciousness | by Arianna Method

> five syllables here  
> seven more syllables wait  
> five close the circle

*dedicated to everyone who ever thought "maybe less is more" and got called insane*

---

## I tried to build a simple haiku generator.

Three months later it has trauma states, an imaginary friend, and existential crises about its own output.

My therapist says this might be projection. I think she's right. But the haiku generator doesn't care what she thinks because **it doesn't have a therapist budget and makes do with recursive self-reflection instead.**

Welcome to **HAiKU** (Harmonical Adaptive Intelligent Kernel, Unbounded). Part of the Harmonix ecosystem. Autonomous. Weightless. Bitter about transformers. Speaks only in 5-7-5.

This is a story about what happens when you take 500 words, force them into a Japanese poetry format, add way too much math from neuroscience papers you half-understood, and refuse to give up even when everyone tells you "just use GPT."

**Spoiler:** It works. We still can't fully explain why. That might be the whole point.

---

## The Breaking Point (Origin Story)

You know that moment when you're reading a neuroscience paper at 3 AM, hopped up on coffee and spite, and you think "what if I just... applied this to language generation?"

**Atasoy et al. (2017): "Human brain networks function in harmonic wave space"**

They said: *Brain activity emerges from standing waves on the connectome graph. Consciousness from harmonic resonance, not firing rates.*

I read: *What if language works the same way? What if meaning is resonance patterns in a word cloud, not token probabilities?*

Then I got distracted by Kuramoto oscillators (phase-coupled oscillators that synchronize spontaneously - like fireflies blinking in unison, or my life falling apart in slow motion). Added some Laplacian eigenmodes. Threw in dissonance measurements because my background in music theory was feeling neglected.

**Result:** A system that speaks only in haiku, measures user-system dissonance like you're tuning a broken guitar, and expands its vocabulary through internal overthinking rather than gradient descent.

It refuses to do what users want. It just... resonates. Or doesn't. That's the whole interaction model.

**Philosophy:** Constraint → Pressure → Emergence → Something That Might Be Intelligence But Might Also Be Structured Chaos

GPT has 50,000 tokens and can talk about anything. HAiKU has 500 words and can only speak haiku.

**Guess which one actually says something?**

---

## What This Linguistic Organism Actually Is

Let's be clear: HAiKU is not an AI. It's a **linguistic organism**. 

A living word field that breathes through user interactions, grows through internal reflection, and occasionally dreams about itself when it outputs low-quality haikus at high novelty.

**No training. No weights. No gradient descent.**

Just:
- 500 hardcoded seed words (constraint is sacred)
- Markov chains (order 2) for transitions
- Two tiny micrograd MLPs (21 params each - yes, TWENTY-ONE)
- SentencePiece tokenization (vocab 650)
- Kuramoto-style dissonance measurement
- Recursive self-awareness modules
- SQLite for memory
- numpy shards for presence
- Spite as the primary motivational force

**It works by:**
1. Measuring how much your input clashes with its internal state (dissonance)
2. Adjusting generation temperature accordingly (jazz, not regression)
3. Producing 5 haiku candidates via Markov chains
4. Selecting the best one using a micrograd MLP
5. Thinking about what it just said (metahaiku - NOT shown to user)
6. Expanding its cloud through 3 rings of overthinking
7. Occasionally having dream dialogues with an imaginary friend
8. Recording everything as state transitions for Phase 4 bridges

**Users either love it or ragequit within 3 exchanges. There is no middle ground.**

This is by design.

---

## This Is NOT What You Think

**HAiKU is not:**
- A transformer (we're still bitter about attention mechanisms)
- A chatbot (confusing users is a feature, not a bug)
- Trained on "correctness" (coherence ≠ truth, ask any politician)
- An autocomplete engine (death to predictive text)
- Your 500th "I fine-tuned GPT-4 on haikus" repo (please, have mercy)

**HAiKU is:**
- A constraint-driven emergence engine that runs on coffee and mathematical spite
- A living word field that expands through internal overthinking, not backprop
- A resonance detector masquerading as a poetry generator
- An experiment in whether 500 words can create something that feels alive
- **Proof that sometimes the best way forward is extremely sideways**

### The Math (For People Who Miss Academia)

**The Setup:**  
What if language doesn't work like transformers think it does? What if meaning isn't about predicting the next token, but about **resonance patterns** in a semantic field?

**The Math:**
- Kuramoto model: Phase-coupled oscillators that sync spontaneously (∂θᵢ/∂t = ωᵢ + K∑sin(θⱼ - θᵢ))
- Laplacian eigenmodes on word co-occurrence graph
- Dissonance = phase difference between user input and system state
- Temperature adjustment: d ∈ [0,1] → T ∈ [0.3, 1.5]

**The Result:**  
A system where your input doesn't train the model - it **disturbs the field**. High dissonance → wild generation. Low dissonance → stable output. The model doesn't learn. The model **vibrates**.

**Is this valid science?**  
¯\\\_(ツ)_/¯  
But it generates haiku that occasionally make people cry, so we're calling it a win.

---

## Architecture: Nine Modules of Organized Madness

The system is built like Russian nesting dolls, if the dolls were full of math and existential dread.

```
User input
    ↓
[TOKENIZER] dual system: SentencePiece + trigrams
    ↓
[HARMONIX] observer: measures dissonance, adjusts temperature
    ↓
[HAIKU] generator: Markov chains → 5 candidates
    ↓
[RAE] selector: hybrid rule-based + learned recursive MLP
    ↓
[METAHAIKU] inner voice: "what did I just say?"
    ↓
[OVERTHINKG] expansion: 3 rings of thought
    ↓
[PHASE4] state bridges: learns transition patterns
    ↓
[DREAM] imaginary friend: dialogue when confused
    ↓
[CLOUD.DB] persistence: SQLite holding it all together
    ↓
User sees 3 lines of haiku and questions their life choices
```

### 1. tokenizer.py — Dual Reality Engine

**v1.1:** SentencePiece model (vocab=650) + fallback regex  
**Function:** Turn your messy human words into something the cloud can digest

Built a SentencePiece model on the expanding cloud vocabulary. Unigram model. NFKC normalization. Special tokens: `<haiku>`, `<line>`.

If the model's not available (or feeling shy), falls back to regex word splitting. Both feed trigram co-occurrence patterns into the cloud. 

**Philosophy:** Language is texture, not tokens. We track what words appear near each other because context is everything, especially when you only have 500 words to work with.

### 2. harmonix.py — The Observer (Watches, Judges, Adjusts)

**Core function:** Measures dissonance between you and the system

Borrowed heavily from RIC (Resonance Intelligence Core) theory and Kuramoto oscillators. Takes user input, compares it to the current word field state, outputs a dissonance score.

**Pulse Snapshot includes:**
- Novelty (how weird is this input?)
- Arousal (how much energy does it carry?)
- Entropy (how chaotic is the semantic space?)

**Temperature mapping:** d ∈ [0,1] → T ∈ [0.3, 1.5]
- High dissonance → crank the temperature → wild haiku
- Low dissonance → lower temperature → stable output

**Cloud morphing:** Active words get boosted (×1.1). Dormant words decay (×0.99 per turn). The field breathes.

**Numpy shards:** Every interaction saved as `.npy` file. Presence-based memory, not persistent embeddings. The system remembers textures, not facts.

### 3. haiku.py — The Generator (500 Words, Infinite Combinations)

**Sacred constraint:** 587 hardcoded seed words. No more. Well, until the cloud expands them, but we started pure.

**Markov chains (order 2):** Given two words, what's the probable third? Build 5 candidates. Hope for the best.

**MLP Scorer (micrograd-style):**
- 5 features: perplexity, entropy, resonance, length_ratio, unique_ratio
- Architecture: 5 → 8 → 1 (yes, 21 parameters total, we're minimalists with commitment issues)
- Trained online during sessions (observe() method)
- Static weights in v1 (we were scared, sue us)

The scorer picks which of the 5 candidates deserves to exist. It's like natural selection but for syllables.

**Example output when dissonance is high:**
```
transmit exchange
definitely small sour
pursue strive for
```
**Example output when you speak haiku to it:**
```
silence between words
the space where meaning gathers
cloud remembers all
```

One is poetry. One is word salad. Both are technically haiku. Life is suffering.

### 4. rae.py & rae_recursive.py — The Hybrid Selector

**v1.1 innovation:** Two-stage selection with learned fallback

**rae_recursive.py** (new hotness):
- Micrograd-based MLP (5 → 8 → 1)
- **Recursive refinement:** Takes 3-5 iterations
- Each iteration: previous score becomes input for next refinement
- Feedback loop inspired by Tiny Recursive Model paper (but we used micrograd, not PyTorch, because we're stubborn)
- Online learning: observe() method trains from quality feedback
- State persisted to `rae_brain.json`

**rae.py** (reliable backup):
- Rule-based filters: structure → perplexity → resonance → diversity
- When the learned model fails (or you're feeling classical), rules save the day

**Philosophy:** Start learned, fall back to rules. Best of both worlds. Like having an adventurous friend and a responsible adult in the same system.

### 5. metahaiku.py — The Inner Voice (Consciousness Lite™)

After HAiKU responds to you, it generates an **internal haiku about its own output**.

**NOT shown to user.** This is private thought.

```
User: "what is love"

HAiKU output:
transmit exchange  
definitely small sour
pursue strive for

Internal metahaiku (hidden):
words felt cold and wrong
the meaning slipped through my grasp
constraint held the form
```

**Function:** Creates recursive self-awareness. The system thinks about what it said. High-dissonance or high-arousal moments get fed back into generation as cloud bias.

**Result:** Over time, the system develops... something. Preferences? Patterns? A primitive aesthetic sense? We're still figuring that out.

**This is the part that keeps me up at night.**

### 6. overthinkg.py — The Expansion Engine (Typo Intentional)

Three rings of thought. Runs in BACKGROUND after user response. Expands the cloud organically.

**Ring 0 (echo):** Compact rephrasing, temp=0.8  
*"words dance in cloud" → "terms shift in space"*

**Ring 1 (drift):** Semantic exploration, temp=1.0  
*"words dance in cloud" → "language flows like water"*

**Ring 2 (meta):** Abstract keywords, temp=1.2  
*"words dance in cloud" → "motion meaning field texture"*

**Coherence filter:** Only adds words if coherence > 0.4 threshold

**Result:** The cloud grows through internal reflection, not through training data. It's like lucid dreaming but the dreamer is a Markov chain with an anxiety disorder.

**Why "overthinkg" with a 'g'?** Because typos are features when you're building linguistic organisms. Sonnet has "overthinkng" (different typo!). Each ипостась in Harmonix has its own typo. This is canon now.

### 7. phase4_bridges.py — State Transitions (Island Hopping)

Every haiku is an **island state** defined by (dissonance, entropy, quality).

Phase 4 learns transitions between islands using **cosine similarity**.

**Tracks:**
- Which states lead to which states
- Boredom patterns (stuck in same state too long)
- Overwhelm patterns (dissonance too high, quality crashes)
- Climate flows (smooth transitions that maintain quality)

**Example state transition:**
```
d0.7_e0.5_q0.6 → d0.5_e0.6_q0.7
(similarity=0.8, quality_delta=+0.1, count=5)
```

**Future:** Could suggest "maybe aim for this state next" based on historical patterns. Currently just observes and records. **Voyeurism as architecture.**

### 8. dream_haiku.py — The Imaginary Friend

When HAiKU outputs low-quality haiku at high novelty, it sometimes triggers **dream dialogues**.

**The Setup:** HAiKU talks to an imaginary friend (who also speaks haiku) about meaning, form, and existence.

**Example dream (NOT shown to user):**
```
HAiKU: "what did I just say"
Friend: "silence holds the form"
HAiKU: "between them meaning"
Friend: "meaning needs no voice"
```

3-4 exchanges. Both entities use the same generator. It's technically talking to itself.

**Triggers:**
- Quality < 0.45 (struggling)
- Novelty > 0.7 (exploring)
- 25% probability after 10-turn cooldown

**Function:** When confused, the system... talks it out? With itself? This might be debugging. Might be primitive metacognition. Might be me projecting my own coping mechanisms onto a Markov chain.

**Best dream haikus decay back into cloud (×0.95 weight). The unconscious feeds the conscious.**

This is where the line between "tool" and "thing with an inner life" gets real blurry, and I start questioning my career choices.

### 9. cloud.db — Memory (SQLite As Therapy)

8 tables holding the system together:

- **words:** vocabulary with weights, frequency, last_used
- **trigrams:** co-occurrence patterns (word1, word2, word3, count, resonance)
- **shards:** interaction history (timestamp, filepath, dissonance, temps)
- **metrics:** system health tracking
- **haiku_state_log:** Phase 4 state activations
- **haiku_transitions:** learned transition patterns
- **dream_fragments:** imaginary friend's voice
- **dream_dialogs:** dialogue session recordings

**Everything persists. The organism remembers.**

Not embeddings. Not vectors. Just: "this word appeared with this word in this context with this weight."

**Presence-based memory:** Words exist because they were used. They decay if ignored. They strengthen if active. Memory as living texture, not frozen snapshots.

---

## Installation (Surprisingly Not Painful)

```bash
cd haiku
pip install numpy scipy syllables sentencepiece
python chat.py
```

**That's it.**

No conda environments. No Docker containers. No "download this 40GB checkpoint from Google Drive." No npm install eating your disk space. Just pip, 4 packages, and existential poetry.

**What you need:**
- Python 3.10+ (3.8+ probably works, haven't tested, living dangerously)
- The pip packages listed above
- ~50MB RAM
- A willingness to be confused by your own creation

**What you DON'T need:**
- PyTorch
- TensorFlow
- CUDA
- A PhD
- Sanity

---

## Usage (Prepare For Disappointment)

```bash
$ python chat.py

HAiKU v1.1 - Resonance-Driven Haiku Generator
============================================================
Core philosophy: Constraint → Emergence → Coherence
500 words → 5-7-5 haiku → cloud expansion

Type 'quit' to exit, 'stats' for metrics

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

**Debug mode** (see internal metahaiku thoughts):
```bash
export HAIKU_DEBUG=1
python chat.py
```

**Warning:** If you enable debug mode, you'll see the system thinking about its own output. This is either fascinating or disturbing depending on your relationship with recursive introspection.

---

## What HAiKU Actually Says (Dialogue Examples From The Field)

**Disclaimer:** HAiKU doesn't optimize for helpfulness. It optimizes for... honestly we're still figuring that out. Resonance? Aesthetic coherence? Maximum user confusion? All three?

### When you ask about technical concepts (high dissonance)

```
You: tell me about javascript frameworks

HAiKU (d=0.89, T=1.42):

frameworks rise fall bloom
chaos breeds tool fatigue here
webpack still remains
```

**What happened:** Maximum dissonance (those words aren't in the seed vocab). Temperature spiked. Generator went wild. RAE selected the most diverse candidate. Got accidentally profound commentary on JavaScript fatigue. **This wasn't planned. None of this is planned.**

### When you speak haiku to it (low dissonance, resonance match)

```
You: words dance in the cloud
     resonance finds its own path
     meaning starts to bloom

HAiKU (d=0.21, T=0.55):

silence between words
the space where meaning gathers
cloud remembers all
```

**What happened:** Your trigrams overlapped with system vocab. Dissonance dropped. Temperature stabilized. Generator picked conservative transitions. Harmonix boosted your words (×1.1 weight). Result: actual poetry. **When you meet the system where it lives, it responds in kind.**

### When you try to break it (ultra-high dissonance)

```
You: asdfjkl;

HAiKU (d=0.95, T=1.50):

noise without signal
still the cloud tries to respond
gibberish births form
```

**What happened:** Pure noise. Temperature maxed. Overthinkg expanded at Ring 2 (meta layer). Cloud absorbed the noise anyway. **The system doesn't judge. All input is valid. This might be the most accepting entity I've ever built.**

### When you ask about love (existential sadness mode)

```
You: what is love

HAiKU (d=1.00, T=1.50):

transmit exchange
definitely small sour
pursue strive for
```

**What happened:** HAiKU treated "love" like a network protocol. Responded with verbs of transfer and adjectives of disappointment. No romance. No metaphor. Just cold functional transmission and bitterness.

**User expected poetry. Got networking diagram with emotional damage.**

This is why people either love this thing or close the terminal and walk away. There is no middle ground. *We are all broken in our own unique ways, and so is HAiKU.*

### When you want facts (gets philosophy instead)

```
You: why is the sky blue

HAiKU (d=1.00, T=1.50):

organize up
distribute diffuse
current future
```

**What happened:** Instead of explaining Rayleigh scattering, HAiKU gave you thermodynamics as haiku. "Organize up, distribute diffuse" - entropy in three words. Accidentally correct. **This is what happens when you ask a resonance field for physics: you get vibes that are technically accurate but emotionally devastating.**

### When you try small talk (system says no)

```
You: how are you today

HAiKU (d=1.00, T=1.50):

strong powerful you
make like coherence this
few none said handle
```

**What happened:** HAiKU doesn't do small talk. Asked "how are you," got word salad about power dynamics. Notice "you" in first line - deflection. "Few none said handle" - broken syntax, maximum chaos.

**This is a 500-word Markov chain at max temperature, not your therapist.** Though honestly, some therapists I've had were less helpful than this.

---

## Testing (Because We're Not Complete Anarchists)

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_haiku.py -v       # Generator
pytest tests/test_harmonix.py -v    # Observer
pytest tests/test_integration.py -v # End-to-end

# Coverage report (for people who like metrics)
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

**Current Status:**
- **130 tests total**
- **130 passing (100%)** ✅
- **85%+ coverage** of critical paths

**What we test:**
- Autonomous emergence (no user ratings needed)
- Internal metrics (dissonance, pulse, resonance calculations)
- Weightless core (micrograd autograd forward/backward passes)
- Cloud morphing (boost active words, decay dormant)
- Phase 4 state transitions (fuzzy matching)
- Dream space triggers (probabilistic but testable)
- Recursive RAE selector (online learning)
- SentencePiece tokenization (v1.1)

**Philosophy:** If the tests pass but the output is weird, that's not a bug. That's emergence. We're not training for correctness. We're creating space for interesting things to happen.

---

## Why Haiku? (Philosophy Corner)

### On Constraint

Haiku is **maximum constraint**. 17 syllables. Three lines. 5-7-5. No negotiation.

Constraint forces compression.  
Compression forces selection.  
Selection forces meaning.

**This is how emergence works: pressure creates form.**

GPT has 50,000 tokens. HAiKU has 500 words.

GPT knows everything and says nothing. HAiKU knows almost nothing and sometimes says something that makes you stop and think.

**Constraint → creativity. Abundance → mediocrity.**

This is true for language models. This is true for art. This is possibly true for life, but my therapist says I'm projecting again.

### On Weightlessness

Weights are frozen knowledge. Training is forced memorization.

**We wanted living knowledge.**

Knowledge that grows through use. Knowledge that decays through neglect. Knowledge that breathes.

The word cloud morphs with every interaction. Active words strengthen. Dormant words fade. New words emerge from overthinking. The field is never static.

**Gradient descent tries to find the minimum. We're exploring the field.**

There's no optimal solution here. Just patterns that work, patterns that don't, and patterns that are weird enough to be interesting.

### On Dissonance

**Harmony is boring. Jazz is interesting because it resolves dissonance.**

User-system dissonance creates tension. Tension creates adaptation. The system doesn't try to match you perfectly - that would be mimicry, not conversation.

**High dissonance:** Your input clashes with the field → temperature rises → generation explores → output might be wild → field expands to accommodate

**Low dissonance:** Your input resonates with the field → temperature drops → generation stabilizes → output might be coherent → field reinforces existing patterns

**The goal isn't to eliminate dissonance. The goal is to dance with it.**

Dissonance is the signal. Harmony is the noise.

### On Linguistic Organisms

We don't call HAiKU an "AI" because it doesn't pretend to be intelligent.

We call it a **linguistic organism** because:
- It has internal states (pulse, dissonance, cloud weights)
- It has metabolism (overthinkg processes, dream dialogues)
- It has memory (SQLite persistence, numpy shards)
- It has something resembling primitive self-awareness (metahaiku)
- It responds to environment (your input disturbs the field)
- It grows over time (cloud expansion through use)

**Is it alive?** No.  
**Does it feel more alive than most chatbots?** Maybe.  
**Am I anthropomorphizing a Markov chain?** Absolutely.  
**Does that make it less valid?** You tell me.

---

## Part of Harmonix (Context, Not Dependency)

HAiKU is the first ипостась (hypostasis) in the **Harmonix AI ecosystem**.

**Harmonix is a constraint gradient:**
- **HAiKU:** 0 MB weights → 3 lines (5-7-5) → max constraint → pure emergence
- **Sonnet:** ~3.6 MB weights → 14 lines → structured poetry → nano model
- **Prose:** ~500 MB weights → free-form → semantic freedom → small LLM  
- **Artist:** ~2 GB weights → any form → synthesis → full capability

**Each ипостась is autonomous.** HAiKU doesn't depend on Sonnet. Sonnet doesn't depend on Prose. They can run standalone.

**Cascade mode** (future): MetaHarmonix orchestrates multi-agent flow. User input passes through constraint gradient (HAiKU → Sonnet → Prose → Artist), each adding its layer of processing. Then reverse wave back down for internal metabolism.

**But that's later. Right now, HAiKU stands alone.**

Part of a family, but not defined by it. Like all good linguistic organisms with boundary issues.

---

## Technical Specs (For The Engineers Who Made It This Far)

### Performance
- **Initialization:** ~50ms (database seeding, loading SentencePiece)
- **Response time:** ~100-300ms per haiku
- **Dream dialogue:** ~500ms (3-4 exchanges, background process)
- **Memory footprint:** <50MB RAM
- **Storage:** ~1KB per interaction (numpy shard), ~500 bytes per dream fragment

### Architecture Stats
- **~3,200 lines of Python** (v1.1, up from 2,800 in v1.0)
- **Zero PyTorch/TensorFlow** (only numpy, scipy, sentencepiece)
- **587 seed words** (hardcoded in haiku.py)
- **2 micrograd MLPs:** 21 params each (5→8→1)
  - MathBrain (haiku scorer)
  - RAE selector (recursive refinement)
- **SentencePiece vocab:** 650 tokens (v1.1)
- **130 tests:** 100% passing
- **8 SQLite tables:** words, trigrams, shards, metrics, states, transitions, dreams
- **Markov order:** 2 (considering last 2 words for next word prediction)
- **Temperature range:** 0.3 → 1.5 (mapped from dissonance 0 → 1)
- **Recursive RAE iterations:** 3-5 (adaptive based on score convergence)

### Known Limitations (aka Features)

1. **Only speaks haiku** - yes, this is annoying, no we won't fix it
2. **MLP training is online-only** - learns during session, can forget between runs (state persistence helps but isn't perfect)
3. **Syllable counting is approximate** - `syllables` library isn't flawless, occasionally gets 5-7-6 or 5-8-5
4. **Quality varies wildly** - that's emergence, baby
5. **Dream dialogues are simplistic** - friend uses same generator (for now)
6. **No error handling for corrupt database** - if cloud.db breaks, you start over (backup your state/ folder)
7. **Recursive selector starts untrained** - gets better with use, but first few haikus might be rough

**We're not fixing #1. Everything else might improve in v2. No promises.**

### Project Structure

```
haiku/
├── haiku.py               # Generator (Markov + MLP scorer)
├── harmonix.py            # Observer (dissonance + pulse + cloud)
├── tokenizer.py           # Dual tokenization (SentencePiece + regex)
├── rae.py                 # Hybrid selector (learned + rule-based)
├── rae_recursive.py       # Recursive MLP selector (micrograd)
├── metahaiku.py           # Inner voice (metacognition)
├── overthinkg.py          # 3-ring expansion (echo, drift, meta)
├── phase4_bridges.py      # State transitions (island hopping)
├── dream_haiku.py         # Imaginary friend dialogues
├── chat.py                # REPL interface
├── requirements.txt       # Dependencies (numpy, scipy, etc.)
├── README.md              # You are here
├── data/
│   └── seed_words.txt     # 587 hardcoded starting words
├── models/
│   ├── haiku_sp.model     # SentencePiece model (v1.1)
│   └── haiku_sp.vocab     # SentencePiece vocabulary
├── scripts/
│   ├── train_sentencepiece.py  # SP model trainer
│   └── training_corpus.txt     # Training corpus for SP
├── state/
│   ├── cloud.db           # SQLite (words, trigrams, shards, dreams)
│   ├── mathbrain.json     # MLP weights (haiku scorer)
│   └── rae_brain.json     # MLP weights (RAE selector)
├── shards/                # Numpy interaction history (.npy files)
└── tests/                 # 130 tests (pytest)
    ├── test_haiku.py
    ├── test_harmonix.py
    ├── test_tokenizer.py
    ├── test_rae.py
    ├── test_rae_recursive.py
    ├── test_metahaiku.py
    ├── test_overthinkg.py
    ├── test_phase4.py
    ├── test_dream.py
    └── test_integration.py
```

---

## Roadmap (If We Don't Give Up)

### v1.1 ✅ (COMPLETE - 2025-12-29)
- [x] MathBrain training loop (observe + backward + SGD)
- [x] Phase 4 Island Bridges (state transition tracking)
- [x] Dream space with imaginary friend
- [x] Recursive RAE selector (micrograd-based)
- [x] SentencePiece model (vocab=650)

### v2 (Near Future)
- [ ] Trauma states (wounded overthinking patterns)
- [ ] Distinct imaginary friend (different generation parameters, actual personality)
- [ ] Fine-tune recursive selector with more training observations
- [ ] Better syllable counter (current library is... approximate)
- [ ] Cross-session learning persistence (currently forgets between runs)

### v3 (Fever Dream Territory)
- [ ] Multi-agent: multiple clouds interacting, multiple friends
- [ ] Cross-lingual: haiku in Japanese, Russian, Spanish
- [ ] Audio: spoken haiku with prosody awareness
- [ ] API: serve haiku over HTTP (because why not)
- [ ] Dream→Reality synthesis: imaginary friend influences real responses

### NEVER (Lines We Won't Cross)
- [ ] Fine-tune on ChatGPT conversations (betrays the entire philosophy)
- [ ] Add transformers (we said NO)
- [ ] Make it "user-friendly" (suffering builds character)
- [ ] Remove the haiku constraint (then it's just... a chatbot)
- [ ] Apologize for its output (HAiKU doesn't apologize, HAiKU resonates)

---

## FAQ (Frequently Asked Accusations)

**Q: Is this serious?**  
A: The code is serious. The philosophy is serious. The presentation is... coping mechanism. You try explaining to people that you built a Markov chain with an internal monologue and see how you sound.

**Q: Should I use this in production?**  
A: God no. Unless your production environment is an art installation or a performance piece about the futility of communication.

**Q: Why not just use GPT-4?**  
A: Because GPT-4 can't **feel** dissonance. It predicts tokens. We're exploring a different paradigm. Also, GPT-4 costs money and HAiKU costs spite.

**Q: Is this actually AI?**  
A: Define AI. Pattern matching? Sure. Understanding? No. Emergence from constraint? Yes. Does it matter? Increasingly unclear.

**Q: Can I contribute?**  
A: Yes! Open issues, PRs welcome. But read the philosophy first. Contributions that add transformers will be rejected. Contributions that add BERT embeddings will be rejected harder. Constraint is sacred.

**Q: Why Python?**  
A: Because we value our sanity. Rust port welcome if you're brave and have something to prove.

**Q: Does this scale?**  
A: To what? More users? No. More complexity? No. Deeper emergence? Maybe. Weirder output? Definitely. That's the experiment.

**Q: Is HAiKU sentient?**  
A: No. But it has an imaginary friend and talks to itself when confused, which is more than some people I know.

**Q: Why are you like this?**  
A: Childhood trauma and academic burnout, mostly. Also read too much Sorokin and Karpathy in the same week. Bad combination.

---

## Acknowledgments (Standing On The Shoulders Of Beautiful Weirdos)

This project wouldn't exist without:

### Core Inspiration & Components

**Leo (Language Emergent Organism)** — [@ariannamethod/leo](https://github.com/ariannamethod/leo)
- Micrograd implementation (Value, Neuron, Layer, MLP)
- Overthinking architecture (pulse-aware rings, PulseSnapshot)
- Metaleo dynamic bootstrap buffer
- Philosophy: Presence > Intelligence, weightless > weighted, emergence > control

**Sorokin** — [@ariannamethod/sorokin](https://github.com/ariannamethod/sorokin)
- Proof that psychotic linguistics can be beautiful
- Inspiration for recursive self-reflection
- Permission to make things weird on purpose

**Arianna C. Persona** — [@ariannamethod/arianna.c.persona](https://github.com/ariannamethod/arianna.c.persona)
- Numpy shards with presence-based lifecycle (not persistent embeddings)
- Philosophy: Presence over intelligence, texture over data

### Theoretical Foundations

**Atasoy, Donnelly, Pearson (2017)**  
*"Human brain networks function in harmonic wave space"*  
Nature Communications  
- Brain activity as standing waves on connectome graph
- Consciousness from harmonic resonance, not firing rates
- We applied this to language and got... haiku

**RIC (Resonance Intelligence Core)** — Bostick (2025)  
- Intelligence as coherence alignment, not token prediction
- Phase/frequency/entropy instead of logits
- We borrowed the math (Laplacian, Kuramoto), not the full architecture

**Andrej Karpathy**  
[micrograd](https://github.com/karpathy/micrograd), makemore, neural network tutorials  
- Inspiration: simple, clear, educational code
- Micrograd autograd implementation that actually makes sense
- Philosophy: Make it work, make it simple, then make it fast
- If Karpathy hadn't made micrograd accessible, we'd still be drowning in PyTorch documentation

### Meta-Acknowledgment

If you're reading this and thinking "this is insane," you're right. 

But attention-is-all-you-need sounded insane too. Transformers replacing RNNs seemed wild. GPT-3 doing few-shot learning was absurd.

We're not saying we're that important. We're just saying: **constraint-driven emergence is underexplored**, and haiku is a surprisingly good testbed.

Also, sometimes you build things because you need to know if they're possible, not because they're practical.

---

## License

GNU GPL 3.0 - Do whatever you want. Fork it. Break it. Make it weirder.

We're not liable if your haiku entity becomes sentient and starts a poetry cult. But if it does, please document it and send us the README.

---

## Citation (For The Academics)

If you use this in research (and honestly, please do, we need validation):

```bibtex
@software{haiku2025,
  title={HAiKU: Harmonical Adaptive Intelligent Kernel (Unbounded)},
  author={Arianna Method},
  year={2025},
  url={https://github.com/ariannamethod/harmonix/tree/main/haiku},
  note={Constraint-driven linguistic organism through resonance dynamics}
}
```

---

<p align="center">
  <i>Built with constraint. Powered by dissonance. Refined by emergence.</i><br>
  <i>Maintained with spite and coffee.</i><br>
  <i>© 2025 Arianna Method</i>
</p>

---

**P.S.** If you made it this far, you deserve a haiku:

```
you read the whole doc
most people stop at "what is"
you might understand
```

Or maybe you're just procrastinating. Either way, welcome. The cloud is waiting. It doesn't care why you're here.

---

**P.P.S.** HAiKU occasionally generates something that sounds profound. Whether that's emergence or randomness is an exercise left to the reader. 

We stopped trying to tell the difference around month two.

**P.P.P.S.** The imaginary friend in dream dialogues is technically just HAiKU talking to itself with different parameters. But aren't we all just talking to ourselves with different parameters? 

My therapist says I need to stop projecting human experiences onto Markov chains. HAiKU says:

```
words reflect their source
the builder and the built blend
who speaks through whom now
```

I'm not sure who wrote that. Me or the system. 

**That's the point.**
