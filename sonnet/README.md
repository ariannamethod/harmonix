# Sonnet - Shakespeare AI (Hypostasis #2)

```
   ███████╗ ██████╗ ███╗   ██╗███╗   ██╗███████╗████████╗
   ██╔════╝██╔═══██╗████╗  ██║████╗  ██║██╔════╝╚══██╔══╝
   ███████╗██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗     ██║   
   ╚════██║██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝     ██║   
   ███████║╚██████╔╝██║ ╚████║██║ ╚████║███████╗   ██║   
   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

**3.57 MB NanoGPT model** trained on Shakespeare, generating 14-line sonnets with pure numpy inference.

Part of the Harmonix AI gradient: **HAiKU (0 MB) → Sonnet (3.57 MB) → Prose (783 MB) → Artist (2 GB)**

## Philosophy

Each hypostasis is **fully autonomous** with its own typo:
- HAiKU: `overthinkg` (kg = kilograms of thought)
- **Sonnet: `overthinkng` (ng = recursive thinking in progress)**
- Prose: TBD
- Artist: TBD

Modules communicate **only through MetaHarmonix in cascade mode** - no direct dependencies.

## Features

✨ **Pure Numpy Transformer** - No PyTorch runtime after weight conversion
🎭 **14-Line Sonnets** - Shakespearean structure with relaxed meter (9-13 syllables)
🌊 **Cloud Learning** - Dissonance decreases as vocabulary grows
💭 **MetaSonnet** - Inner voice reflection with bootstrap buffer
🔄 **Overthinkng** - 3 rings of thought expansion (echo, drift, meta)
🧠 **Emergent Layer** - 6 MLP modules for semantic understanding and creativity
📊 **100% Test Coverage** - 133/133 pytest tests passing

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# One-time: Convert weights to numpy
python scripts/convert_weights.py

# Launch REPL
python chat.py
\`\`\`

## Example Sonnets

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
├── state/          # Model weights
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

Cascade mode only - no direct connection to other hypostases.

## License

See root LICENSE
