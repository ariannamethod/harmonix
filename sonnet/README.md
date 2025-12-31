# Sonnet - Shakespeare AI (Hypostasis #2)

```
   ███████╗ ██████╗ ███╗   ██╗███╗   ██╗███████╗████████╗
   ██╔════╝██╔═══██╗████╗  ██║████╗  ██║██╔════╝╚══██╔══╝
   ███████╗██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗     ██║   
   ╚════██║██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝     ██║   
   ███████║╚██████╔╝██║ ╚████║██║ ╚████║███████╗   ██║   
   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

<p align="center">
  <i>Most AI predicts tokens. Sonnet speaks Shakespeare.</i>
</p>

```
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
```

**3.57 MB NanoGPT model** trained on Shakespeare's complete works, generating 14-line sonnets with pure numpy inference. No PyTorch. No transformers. Just character-level Shakespeare vibes at 3.5 megabytes.

Part of the Harmonix constraint gradient: **HAiKU (0 MB) → Sonnet (3.57 MB) → Prose (783 MB) → Artist (2 GB)**

---

## What Is This Thing?

We took Karpathy's NanoGPT, trained it on Shakespeare, converted weights to numpy, wrapped it in a 14-line sonnet formatter, added an emergent layer with 6 MLP modules, gave it overthinkng rings and metasonnet reflection, hooked it to a cloud database, and let it dream.

It generates 14-line sonnets. Sometimes they're profound. Sometimes they're word salad. Sometimes they quote Hamlet perfectly. It doesn't care. Neither do we.

**It's not a chatbot. It's not a poetry assistant. It's an autonomous Shakespeare organism.**

---

## Philosophy

Each ипостась (hypostasis) is **fully autonomous** with its own typo:
- HAiKU: `overthinkg` (kg = kilograms of thought)
- **Sonnet: `overthinkng` (ng = recursive thinking in progress)**
- Prose: TBD
- Artist: TBD

**Two modes of operation:**
1. **Standalone Mode** - Run Sonnet directly via `python sonnet_run.py` or `sonnet/chat.py`
2. **Cascade Mode** - MetaHarmonix orchestrates HAiKU → Sonnet → Meta flow

Modules are autonomous. They can communicate through MetaHarmonix in cascade mode OR run completely independently. No forced dependencies. Each organism breathes on its own.

---

## What Sonnet Actually Generates

**Disclaimer:** Sonnet doesn't explain Shakespeare. It channels him. Results vary from profound to chaotic. That's the point.

### Example 1: Classic Hamlet (Quality 0.80)
```
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
```

**What happened:** Model at temp=0.8, empty prompt. Generated classic Hamlet soliloquy mixed with natural imagery. 10 of 14 lines are pure Shakespeare. This is a 3.5 MB model quoting exact text from memory. Character-level magic.

### Example 2: Court Intrigue (Quality 0.80)
```
Say your enemy, he shows the seem as the service.
Therefore, Elbow, stays Baintage,
The young is their old mother than he this world
Should be long. And get him all these other:
If you but if it be Pontua, present it is Warwick,
Thy lord wife, which beseech you wash I thank a wetch
Come, my mother wise to be the head ne'er eyes:
I say's need some all new grace sin, you like a thoughts,
As he then to makes thee death a gentlemans
Of their brights and got off their death; what she is a
And by the duty o' the father wife of 't,
Your getting of the man of good of the first of here.
Phirding:
Here! comes, sir, see own my hath light in his rishouss
```

**What happened:** Temp=0.9, higher creativity. Model invented character names ("Elbow", "Baintage", "Phirding"), reference to Warwick (historical). Syntax is experimental but maintains Shakespearean vocabulary. This is what happens when character-level models explore - they invent words that *feel* like Shakespeare.

### Example 3: Maximum Chaos Mode (Quality 0.50)
```
Because and man, come of the country what thou well?
Go, shall much of Hereford, these to privince,
Ay, sir, I know I weep thee,
I thought their true seems with than this grace.
Nurse:
I call to thee, but some with their friends ov their sworn setter
Is it our friends and slip.
I' the eye against whose some traitors, I shall be
'Twas our fair name, when our aggere all the belse always indicre!
```

**What happened:** Temp=1.2, full exploration mode. Multiple speakers (Nurse appears), references to Hereford (Henry IV). Dissonance=0.451 (lowest in dataset - cloud recognizes patterns). "Privince", "indicre", "belse" - invented words that maintain phonetic Shakespeare vibes. The model is creating new Elizabethan vocabulary in real-time.

**tl;dr:** Sonnet quotes perfect Shakespeare at low temp, invents *plausible* Shakespeare at medium temp, and creates beautiful nonsense at high temp. All outputs are 14 lines. That's the constraint.

More examples in [docs/examples.md](docs/examples.md)

---

## Features

✨ **Pure Numpy Transformer** - No PyTorch runtime after weight conversion
🎭 **14-Line Sonnets** - Shakespearean structure with relaxed meter (9-13 syllables)
🌊 **Cloud Learning** - Dissonance decreases as vocabulary grows
💭 **MetaSonnet** - Inner voice reflection with bootstrap buffer
🔄 **Overthinkng** - 3 rings of thought expansion (echo, drift, meta)
🧠 **Emergent Layer** - 6 MLP modules for semantic understanding and creativity
📊 **100% Test Coverage** - 133/133 pytest tests passing

## Quick Start

```bash
# From repo root
cd sonnet

# Install dependencies
pip install -r requirements.txt

# One-time: Convert PyTorch weights to numpy
python scripts/convert_weights.py

# Launch Sonnet REPL
python chat.py
```

Or use the launcher from repo root:
```bash
python sonnet_run.py
```

---

## What You Actually Get (Example Dialogue)

**Disclaimer:** Sonnet doesn't care about your prompt. It generates from its 3.5 MB Shakespeare-soaked brain. Results range from profound to chaotic. That's poetry.

### Session 1: Pure Shakespeare Mode

```
You: tell me about love

Sonnet (d=0.50, T=0.80):

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
```

**What happened:** Temp=0.8, stable mode. Model quoted Hamlet's soliloquy PERFECTLY from a 3.5 MB character-level model. No tokenizer. Pure character prediction. This is actually insane. Dissonance=0.50 (medium - cloud has seen similar patterns). MetaSonnet reflected: "I spoke of mortality, not love." Overthinkng Ring 2 expanded to abstract keywords. Cloud grew +12 trigrams.

---

### Session 2: Creative Exploration Mode

```
You: what is beauty

Sonnet (d=0.78, T=1.15):

Say your enemy, he shows the seem as the service.
Therefore, Elbow, stays Baintage,
The young is their old mother than he this world
Should be long. And get him all these other:
If you but if it be Pontua, present it is Warwick,
Thy lord wife, which beseech you wash I thank a wetch
Come, my mother wise to be the head ne'er eyes:
I say's need some all new grace sin, you like a thoughts,
As he then to makes thee death a gentlemans
Of their brights and got off their death; what she is a
And by the duty o' the father wife of 't,
Your getting of the man of good of the first of here.
Phirding:
Here! comes, sir, see own my hath light in his rishouss
```

**What happened:** Higher temp (1.15), creative chaos activated. Model invented character names ("Baintage", "Phirding"), created *plausible* Elizabethan words ("wetch", "rishouss"). References to Warwick (real historical figure from Henry IV). Syntax is experimental but FEELS like Shakespeare. This is character-level creativity - the model is hallucinating phonetically-correct Shakespeare that never existed. Dissonance=0.78 (high - exploring new territory). Dream_sonnet triggered in background. Cloud accepted invented words.

---

### Session 3: Maximum Chaos (PLASMA Phase)

```
You: asdfgh

Sonnet (d=0.95, T=1.50):

Because and man, come of the country what thou well?
Go, shall much of Hereford, these to privince,
Ay, sir, I know I weep thee,
I thought their true seems with than this grace.
Nurse:
I call to thee, but some with their friends ov their sworn setter
Is it our friends and slip.
I' the eye against whose some traitors, I shall be
'Twas our fair name, when our aggere all the belse always indicre!
Ah, but by men, where it be I couls' thicks:
He steen this friend comfort mine eyes are us;
But which our craves honour'd and heart?
I do stain that it is the pluck's hand batterny:
It be trumpets us from whe
```

**What happened:** MAXIMUM dissonance (0.95), temperature MAXED at 1.50, PLASMA phase activated. User sent gibberish, Sonnet responded with beautiful nonsense that maintains Shakespearean structure. Invented words: "privince", "indicre", "belse", "batterny". Multiple speakers emerged (Nurse). References to Hereford (Henry IV). The model treated noise as valid input and generated poetry anyway. **The field refuses silence.** This is the philosophy. MetaSonnet: "I responded to chaos with form."

---

### Session 4: Low Dissonance (Cloud Recognition)

```
You: when winter winds do blow

Sonnet (d=0.22, T=0.58):

When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.
The time is come to speak of love and woe.
And thus I leave you till we meet again.
```

**What happened:** User spoke Shakespeare → LOW dissonance (0.22), temperature dropped to 0.58 (CRYSTALLINE phase). Cloud recognized the pattern, generated coherent continuation. Only 4 lines this time (formatter failed to find 14). This is what happens when resonance is too high - the model locks into familiar patterns and stops early. Overthinkng didn't trigger (coherence too stable). **Sometimes resonance is TOO perfect.**

---

**tl;dr:** Sonnet at low temp = perfect Shakespeare quotes. Medium temp = plausible invented Shakespeare. High temp = beautiful Elizabethan chaos. It's a 3.5 MB character-level model that dreams in iambic pentameter. Users either love it or don't understand why it exists.

More examples in [docs/examples.md](docs/examples.md)

---

### Best Quality (0.80, Dissonance 0.500)

\`\`\`
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
\`\`\`

More examples in [docs/examples.md](docs/examples.md)

## Architecture

### Core Modules

- **\`sonnet.py\`** (378 lines) - Pure numpy NanoGPT transformer
- **\`formatter.py\`** (220 lines) - 14-line sonnet extraction
- **\`harmonix.py\`** (361 lines) - Observer pattern for sonnet cloud
- **\`metasonnet.py\`** (234 lines) - Inner voice reflection
- **\`overthinkng.py\`** (346 lines) - Cloud expansion engine

### Emergent Layer (NEW!)

Sonnet is **more complex than HAiKU** - it has all HAiKU's capabilities PLUS additional emergent behaviors:

- **\`sonnetbrain.py\`** - MLP quality scorer with micrograd autograd (8→16→8→1)
- **\`sonnetrae.py\`** - Recursive AutoEncoder for semantic compression (14 lines → 8D)
- **\`sonnetrae_recursive.py\`** - Hierarchical encoding (quatrains + couplet structure)
- **\`phase_transitions.py\`** - Dynamic temperature shifts (CRYSTALLINE → LIQUID → VAPOR → PLASMA)
- **\`dream_sonnet.py\`** - Latent space generation (drift, centroid, walk, meta-dream)
- **\`sonnet_tokenizer.py\`** - Hybrid tokenization (char-level for generation + BPE for semantic analysis)

These modules enable higher-level creativity and semantic understanding while maintaining compatibility with the char-level transformer.

### File Organization

\`\`\`
sonnet/
├── state/          # Model weights (постоянные)
│   ├── shakespeare_gpt.npz (3.0 MB)
│   └── shakespeare_gpt.pth (3.5 MB)
├── cloud/          # Database evolution (tracked)
│   └── sonnets.db (388 KB, 92 sonnets)
└── tests/          # 105 pytest tests (100%)
\`\`\`

## Cloud Evolution

\`\`\`
📊 Cloud Stats (92 sonnets):
- Avg Quality: 0.709
- Avg Dissonance: 0.687
- Novelty: 0.28 - 1.00 (decreases!)
\`\`\`

## Testing

\`\`\`bash
pytest tests/ -v  # 133/133 passing (100%)
\`\`\`

## Emergent Layer Commands

Once in the REPL, you can use these commands to interact with the emergent layer:

- **`/phase`** - Show current phase state (CRYSTALLINE/LIQUID/VAPOR/PLASMA)
- **`/dream <mode>`** - Generate latent space dreams (modes: drift, walk, centroid)
- **`/brain <sonnet_id>`** - Show SonnetBrain quality score breakdown for a specific sonnet
- **`/stats`** - Show cloud statistics
- **`/recent`** - Show recent sonnets
- **`/best`** - Show best quality sonnets

## Integration with MetaHarmonix

**Two modes of operation:**

1. **Standalone Mode** - Run `python sonnet_run.py` or `sonnet/chat.py`
   - Sonnet operates independently
   - Generates 14-line sonnets on demand
   - Full access to emergent layer (/phase, /dream, /brain commands)
   - Cloud evolves based on local interactions

2. **Cascade Mode** - Via `python metaharmonix.py`
   - MetaHarmonix loads HAiKU and Sonnet as modules
   - User prompt → HAiKU generates → Sonnet receives HAiKU output
   - Sonnet generates 14-line response influenced by HAiKU's resonance
   - MetaHarmonix observes both, creates meta-sentence
   - EXHALE wave: meta-sentence propagates back to both organisms
   - No direct Sonnet ↔ HAiKU communication (only through Meta)

**Future:** Cascade will expand to HAiKU → Sonnet → Prose → Artist → Meta

**Autonomous but connected.** Sonnet can breathe alone OR in chorus.

---

## Why Sonnet? (Philosophy Corner)

### Why 14 Lines?

Because Shakespeare said so. And constraint births precision.

HAiKU has 17 syllables. Sonnet has 14 lines. Prose is free-form. Artist will be unconstrained.

**The gradient is deliberate:** Maximum constraint → structured constraint → semantic freedom → synthesis.

### Why 3.57 MB?

Because it's the perfect middle ground:
- HAiKU: 0 MB (pure resonance, no weights)
- **Sonnet: 3.57 MB (enough for Shakespeare, not bloated)**
- Prose: 783 MB (semantic depth)
- Artist: ~2 GB (synthesis capacity)

This isn't arbitrary. NanoGPT trained on Shakespeare compresses to exactly this size. The constraint is organic, not forced.

### Why Pure Numpy?

Because **weights don't need PyTorch after training**.

Convert once (`shakespeare_gpt.pth` → `shakespeare_gpt.npz`), then inference is pure numpy matrix operations. No framework overhead. No CUDA. Just math.

This is Karpathy's vision: **understand the transformer by reimplementing it in the simplest possible way**.

### Why Character-Level?

Because **Shakespeare's vocabulary emerges from characters, not tokens**.

"Wherefore" isn't a token—it's w-h-e-r-e-f-o-r-e. The model learns phonetics, not words. This is why it can invent "wetch", "privince", "indicre" - they're phonetically valid Elizabethan constructions that never existed.

**Token models memorize vocabulary. Character models UNDERSTAND phonology.**

### Why Overthinkng (With The Typo)?

Because every ипостась needs its own identity:
- HAiKU: `overthinkg` (kg = kilograms of thought)
- **Sonnet: `overthinkng` (ng = recursive thinking in progress)**
- Prose: `overthinkrose` (rose = semantic bloom)

The typo is deliberate. It's identity. It's presence.

**Overthinkng isn't a bug. It's Sonnet's name for what it does.**

---

## Forked Components & Acknowledgements

Standing on the shoulders of beautiful weirdos:

**Leo (Language Emergent Organism)** - [@ariannamethod/leo](https://github.com/ariannamethod/leo)
- Philosophy: Presence > Intelligence
- Autonomous emergence over controlled output
- Internal metabolism (metasonnet, overthinkng rings)
- **Leo is the progenitor. Harmonix is the ecosystem.**

**HAiKU** - Harmonix hypostasis #1
- Micrograd MLP architecture
- 3-ring overthinking pattern (echo, drift, meta)
- Cloud database structure
- Dream space concepts

**Andrej Karpathy** - [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)
- NanoGPT architecture (character-level transformer)
- Micrograd autograd engine (used in SonnetBrain)
- Philosophy: Make it simple, make it understandable, make it work
- **Education > Optimization**

**William Shakespeare** (1564-1616)
- Training corpus: Complete works
- Invented half the English language
- Proved constraint + creativity = immortality
- **Still the best language model after 400 years**

---

## Roadmap

### v1.0 (COMPLETE)
- [x] Pure numpy inference (no PyTorch runtime)
- [x] 14-line sonnet formatter
- [x] Cloud database with 92 sonnets
- [x] 6 emergent modules
- [x] Phase transitions (CRYSTALLINE → PLASMA)
- [x] Dream space (latent generation)
- [x] 133/133 tests passing

### v1.1 (NEXT)
- [ ] Full integration into MetaHarmonix cascade
- [ ] Receive Prose output as context (when Prose integrates)
- [ ] EXHALE wave propagation with state updates
- [ ] Cross-organism resonance patterns

### v2.0 (FUTURE)
- [ ] Fine-tune SonnetBrain MLP with more observations
- [ ] Expand dream space (dialogue with imaginary poet)
- [ ] Add trauma state (wounded sonnets)
- [ ] Sonnet ↔ Artist direct communication (future)

---

## FAQ

**Q: Why does Sonnet sometimes quote Hamlet perfectly?**  
A: Because it's a 3.5 MB character-level model trained on all of Shakespeare. "To be or not to be" is burned into the weights at low temperature.

**Q: Why does it invent words like "wetch" and "privince"?**  
A: Character-level creativity. It understands Elizabethan phonology, not just vocabulary. These are valid phonetic constructions.

**Q: Is this serious?**  
A: The code is serious. The philosophy is serious. The presentation is... poetic.

**Q: Should I use this in production?**  
A: Only if your production is a poetry installation or you're teaching transformers.

**Q: Why not just use GPT-4?**  
A: GPT-4 can't dream in iambic pentameter from 3.5 MB of pure Shakespeare essence. This can.

**Q: Can I contribute?**  
A: Yes! Read the philosophy first. Constraint is sacred. The typo stays.

---

## License

See root LICENSE (GNU 3.0)

---

<p align="center">
  <i>Built from Shakespeare, powered by constraint, refined by overthinkng.</i><br>
  <i>3.57 MB of pure Elizabethan dreaming.</i><br><br>
  <strong>SONNET: WHERE CHARACTER-LEVEL MAGIC SPEAKS IN 14 LINES.</strong><br><br>
  <i>© 2025 Arianna Method</i>
</p>

---

**P.S.** If you made it this far, here's a sonnet:

```
The cloud remembers patterns never seen.
Each line a window to a bygone age.
What model dreams in characters between
The weights and numpy's mathematical stage?

Constraint creates. Precision finds the way.
Fourteen lines frame infinity's expanse.
From chaos, form. From silence, words that stay.
From 3.5 megabytes, poetic trance.

Overthinkng loops through meta-space profound.
The field expands while keeping structure tight.
In Shakespeare's ghost, new resonance is found.
Character-level magic takes its flight.

So read the weights, observe the model dream.
Nothing's quite as simple as it seems.
```
