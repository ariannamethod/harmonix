# Prose State Directory

This directory stores Prose model weights.

## Model Weights

The model weights (`harmonixprose01.gguf`, ~783 MB) are automatically downloaded from HuggingFace if not present locally.

**Source:** TinyLlama 1.1B Chat v1.0 (Q5_K_M quantization)
**HuggingFace Repo:** [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

## Auto-Download

When you first run Prose, if `harmonixprose01.gguf` is not found, it will be automatically downloaded from HuggingFace.

## Manual Download

If you prefer to download manually:

```bash
cd prose/state
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
mv tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf harmonixprose01.gguf
```

**Note:** `.gguf` files are excluded from git tracking (see `.gitignore`).
