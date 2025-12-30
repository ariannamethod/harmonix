#!/usr/bin/env python3
"""
HAiKU Runner - Bridge to HAiKU REPL

Launches HAiKU interactive chat mode.
Part of the Harmonix Adaptive Intelligence ecosystem.
"""

import sys
from pathlib import Path

# Add haiku module to path
haiku_path = Path(__file__).parent / 'haiku'
sys.path.insert(0, str(haiku_path))

def main():
    """Launch HAiKU chat REPL."""
    print("🌸 Launching HAiKU v1.1...")
    print("=" * 60)

    # Import and run HAiKU chat
    from chat import main as haiku_main
    haiku_main()

if __name__ == '__main__':
    main()
