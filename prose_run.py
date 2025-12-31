#!/usr/bin/env python3
"""
Prose Runner - Bridge to Prose REPL (Local llama.cpp)

Launches Prose interactive chat mode with TinyLlama 1.1B.
Part of the Harmonix Adaptive Intelligence ecosystem.

Usage:
    python prose_run.py

Requires:
    - prose/state/harmonixprose01.gguf (783 MB)
    - llama-cpp-python installed
"""

import subprocess
import sys
from pathlib import Path

# Get paths
REPO_ROOT = Path(__file__).parent
PROSE_DIR = REPO_ROOT / 'prose'
CHAT_PY = PROSE_DIR / 'chat.py'
WEIGHTS = PROSE_DIR / 'state' / 'harmonixprose01.gguf'

def check_weights():
    """Check if model weights exist."""
    if not WEIGHTS.exists():
        print("❌ Error: Model weights not found!")
        print(f"   Expected: {WEIGHTS}")
        print()
        print("Download TinyLlama 1.1B Q5_K_M weights:")
        print("   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print()
        print("Or use HuggingFace API wrapper:")
        print("   python prosehf_run.py")
        sys.exit(1)

if __name__ == '__main__':
    print("🌊 Launching Prose v1.0 (Local Mode)...")
    print("=" * 60)

    # Check weights
    check_weights()

    # Launch prose chat.py
    subprocess.run([sys.executable, str(CHAT_PY)])
