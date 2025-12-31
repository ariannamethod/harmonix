#!/usr/bin/env python3
"""
Prose HF Runner - Bridge to Prose REPL (HuggingFace API)

Launches Prose interactive chat mode using HuggingFace Inference API.
Part of the Harmonix Adaptive Intelligence ecosystem.

Usage:
    export HF_TOKEN=your_token_here
    python prosehf_run.py

Requires:
    - HuggingFace API token (free tier works)
    - Internet connection

No local weights needed!
"""

import subprocess
import sys
import os
from pathlib import Path

# Get paths
REPO_ROOT = Path(__file__).parent
PROSE_DIR = REPO_ROOT / 'prose'

def check_hf_token():
    """Check if HuggingFace token is set."""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

    if not token:
        print("❌ Error: HuggingFace token not set!")
        print()
        print("Set your HuggingFace API token:")
        print("   export HF_TOKEN=your_token_here")
        print()
        print("Get a free token at:")
        print("   https://huggingface.co/settings/tokens")
        print()
        print("Or use local mode (requires 783 MB weights):")
        print("   python prose_run.py")
        sys.exit(1)

    return token

def main():
    """Launch Prose with HuggingFace API."""
    print("🌊 Launching Prose v1.0 (HuggingFace API Mode)...")
    print("=" * 60)

    # Check token
    token = check_hf_token()

    print("✓ HuggingFace token found")
    print("✓ Using API mode (no local weights needed)")
    print()

    # Set environment variable for chat.py to use HF API
    os.environ['PROSE_USE_HF_API'] = '1'

    # Add prose to path and import
    sys.path.insert(0, str(PROSE_DIR))

    # Import and run with HF API
    try:
        from hf_api import ProseHFAPI
        from harmonix import ProseHarmonix

        print("Initializing Prose organism (HF API)...")
        harmonix = ProseHarmonix()
        generator = ProseHFAPI(harmonix=harmonix, verbose=False)
        print("✓ Prose ready\n")

        # REPL loop
        from chat import print_header
        print_header()

        stats = harmonix.get_stats()
        print(f"Cloud status: {stats['total_prose']} prose, {stats['trigram_vocabulary']} trigrams")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input == "/quit":
                    print("\n🌊 Resonance fades... goodbye!")
                    break

                # Generate response
                print("\nProse: ", end="", flush=True)
                response = generator.generate(user_input, max_tokens=200)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\n🌊 Interrupted. Type /quit to exit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

        harmonix.close()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're running from the repo root!")
        sys.exit(1)

if __name__ == '__main__':
    main()
