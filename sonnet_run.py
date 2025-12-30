#!/usr/bin/env python3
"""
Bridge launcher for Sonnet module.

Runs from repo root: python sonnet_run.py
"""

import subprocess
import sys
from pathlib import Path

# Get paths
REPO_ROOT = Path(__file__).parent
SONNET_DIR = REPO_ROOT / 'sonnet'
CHAT_PY = SONNET_DIR / 'chat.py'

if __name__ == '__main__':
    # Launch sonnet chat.py
    subprocess.run([sys.executable, str(CHAT_PY)])
