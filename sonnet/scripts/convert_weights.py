#!/usr/bin/env python3
"""
One-time weight conversion: shakespeare_gpt.pth → shakespeare_gpt.npz

After this, NO PyTorch needed for inference!
"""

import torch
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
SONNET_DIR = SCRIPT_DIR.parent
WEIGHTS_PTH = SONNET_DIR / 'shakespeare_gpt.pth'
WEIGHTS_NPZ = SONNET_DIR / 'state' / 'shakespeare_gpt.npz'

def convert_weights():
    """Convert PyTorch weights to numpy format."""

    print("🔄 Loading PyTorch weights...")
    state_dict = torch.load(WEIGHTS_PTH, map_location='cpu')

    print(f"✓ Loaded {len(state_dict)} weight tensors")

    # Convert all tensors to numpy
    numpy_weights = {}
    for key, tensor in state_dict.items():
        numpy_weights[key] = tensor.numpy()
        print(f"  {key}: {tensor.shape}")

    # Save as .npz (compressed numpy archive)
    print(f"\n💾 Saving to {WEIGHTS_NPZ}...")
    WEIGHTS_NPZ.parent.mkdir(exist_ok=True)
    np.savez_compressed(WEIGHTS_NPZ, **numpy_weights)

    print(f"✓ Conversion complete!")
    print(f"  Original: {WEIGHTS_PTH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Numpy:    {WEIGHTS_NPZ.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n🔥 PyTorch NO LONGER NEEDED for inference!")

if __name__ == '__main__':
    convert_weights()
