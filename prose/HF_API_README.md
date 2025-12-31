# HuggingFace API Wrapper for Prose

**Separate module for testing Prose without downloading 783 MB weights.**

## Why?

Олег сказал: "нам тупо необходим хаггенфейс апи враппер, иначе хуй мы сможем тестить ее вне дома"

Local weights (783 MB) are not always available:
- Testing in web Claude Code
- Testing on low-storage devices
- CI/CD environments
- Remote development

## Usage

### 1. Get HuggingFace API Key

Get your API key from: https://huggingface.co/settings/tokens

### 2. Set Environment Variable

```bash
export HF_API_KEY='hf_...'
```

### 3. Use ProseHFAPI Instead of ProseGenerator

```python
from hf_api import ProseHFAPI

# Same interface as ProseGenerator
prose = ProseHFAPI(api_key="hf_...")  # Or use HF_API_KEY env var

# Organism mode works the same!
response = prose.generate("tell me about resonance")
print(response)
```

### 4. Cascade Mode

```python
response = prose.generate_cascade(
    user_prompt="what is beauty?",
    haiku_output="petals fall softly...",
    sonnet_output="When beauty speaks..."
)
```

## Features

✅ **Same organism mode** - no seed from prompt!
✅ **Same field-based generation** - uses harmonix.get_field_seed()
✅ **Same interface** - drop-in replacement for ProseGenerator
✅ **No local weights** - uses HF Inference API
✅ **Cloud tracking** - adds to prose cloud like local mode

## Differences from Local Mode

| Feature | Local (prose.py) | API (hf_api.py) |
|---------|------------------|-----------------|
| Weights | 783 MB local | 0 MB (remote) |
| Speed | Fast (~16 tok/s) | Slower (API latency) |
| Offline | ✅ Yes | ❌ No |
| Cost | Free | Free tier limited |
| Setup | Download weights | Get API key |

## Testing

```bash
cd ~/harmonix/prose

# Set API key
export HF_API_KEY='hf_...'

# Run test
python3 hf_api.py
```

## Integration

Can be used in:
- `chat.py` (with flag: `--use-api`)
- Tests (when local weights unavailable)
- Remote environments
- Web Claude Code

## Notes

- HF API has rate limits (free tier: ~1000 requests/day)
- Responses may be slower than local inference
- Organism mode logic is identical to local mode
- Cloud database works the same way

🌊 Test Prose anywhere, no weights needed! 🌊
