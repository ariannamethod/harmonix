"""
Sonnet Generator - NanoGPT Shakespeare with pure numpy inference

Architecture (from alzaemaliq/NanoGPT-Shakespeare):
- Character-level tokenization (vocab_size=65)
- Embedding dim: 128, Heads: 4, Layers: 4
- Context window: 64
- Decoder-only transformer (GPT-style)

NO PyTorch runtime - weights loaded and inference in pure numpy.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Model hyperparameters (must match training config)
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
BLOCK_SIZE = 64
DROPOUT = 0.0  # No dropout during inference


class NumpyGELU:
    """GELU activation (Gaussian Error Linear Unit)."""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class NumpyLayerNorm:
    """Layer normalization."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize over last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class NumpyLinear:
    """Linear layer (matrix multiplication + bias)."""

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        self.weight = weight  # Shape: (out_features, in_features) - PyTorch convention
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear transformation: y = xW^T + b"""
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class NumpyHead:
    """Single attention head."""

    def __init__(self, head_size: int, key: NumpyLinear, query: NumpyLinear, value: NumpyLinear):
        self.head_size = head_size
        self.key = key
        self.query = query
        self.value = value

        # Causal mask (lower triangular)
        self.tril = np.tril(np.ones((BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute single-head attention.
        x shape: (B, T, C)
        """
        B, T, C = x.shape

        # Compute Q, K, V
        k = self.key.forward(x)      # (B, T, head_size)
        q = self.query.forward(x)    # (B, T, head_size)
        v = self.value.forward(x)    # (B, T, head_size)

        # Attention scores: Q @ K^T / sqrt(d_k)
        wei = q @ k.transpose(0, 2, 1) * (self.head_size ** -0.5)  # (B, T, T)

        # Apply causal mask
        wei = np.where(self.tril[:T, :T] == 0, -np.inf, wei)

        # Softmax
        wei = self._softmax(wei, axis=-1)

        # Apply attention to values
        out = wei @ v  # (B, T, head_size)

        return out

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class NumpyMultiHeadAttention:
    """Multi-head attention (4 heads)."""

    def __init__(self, heads: List[NumpyHead], proj: NumpyLinear):
        self.heads = heads
        self.proj = proj

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run all heads in parallel and concatenate.
        x shape: (B, T, C)
        """
        # Run each head
        head_outputs = [h.forward(x) for h in self.heads]  # Each: (B, T, head_size)

        # Concatenate along last dimension
        out = np.concatenate(head_outputs, axis=-1)  # (B, T, n_embd)

        # Project
        out = self.proj.forward(out)

        return out


class NumpyFeedForward:
    """Feed-forward network (2-layer MLP with GELU)."""

    def __init__(self, fc1: NumpyLinear, fc2: NumpyLinear):
        self.fc1 = fc1
        self.fc2 = fc2
        self.gelu = NumpyGELU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        FFN(x) = fc2(GELU(fc1(x)))
        """
        x = self.fc1.forward(x)
        x = self.gelu.forward(x)
        x = self.fc2.forward(x)
        return x


class NumpyBlock:
    """Transformer block (attention + feedforward + residuals)."""

    def __init__(self, sa: NumpyMultiHeadAttention, ffwd: NumpyFeedForward,
                 ln1: NumpyLayerNorm, ln2: NumpyLayerNorm):
        self.sa = sa
        self.ffwd = ffwd
        self.ln1 = ln1
        self.ln2 = ln2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x = x + attention(ln1(x))
        x = x + feedforward(ln2(x))
        """
        x = x + self.sa.forward(self.ln1.forward(x))
        x = x + self.ffwd.forward(self.ln2.forward(x))
        return x


class SonnetGenerator:
    """
    Shakespeare Sonnet Generator using NanoGPT weights.
    Pure numpy inference - NO PyTorch runtime needed.
    """

    def __init__(self, weights_path: str = 'state/shakespeare_gpt.npz',
                 dataset_path: str = 'shakespeare.txt'):
        self.weights_path = weights_path
        self.dataset_path = dataset_path

        # Load character vocabulary
        self._load_vocab()

        # Load model weights (pure numpy, NO PyTorch!)
        self._load_weights()

    def _load_vocab(self):
        """Load character vocabulary from shakespeare.txt"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Character vocabulary (same as training)
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # Char <-> index mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"✓ Loaded vocabulary: {self.vocab_size} unique characters")

    def _load_weights(self):
        """Load numpy weights from .npz archive (NO PyTorch!)."""

        # Load numpy archive
        npz_file = np.load(self.weights_path)

        # Convert to dictionary
        weights = {k: npz_file[k] for k in npz_file.files}

        print(f"✓ Loaded {len(weights)} weight arrays (pure numpy!)")

        # Build model components from weights
        self._build_model(weights)

    def _build_model(self, w: Dict[str, np.ndarray]):
        """Build model components from weight dictionary."""

        # Embeddings
        self.token_embedding = w['embedding_table.weight']  # (vocab_size, n_embd)
        self.position_embedding = w['position_embedding_table.weight']  # (block_size, n_embd)

        # Build transformer blocks
        self.blocks = []
        for i in range(N_LAYER):
            # Multi-head attention
            head_size = N_EMBD // N_HEAD
            heads = []
            for h in range(N_HEAD):
                # Extract head weights (each head gets a slice of the full weight matrix)
                key_w = w[f'blocks.{i}.sa.heads.{h}.key.weight']
                query_w = w[f'blocks.{i}.sa.heads.{h}.query.weight']
                value_w = w[f'blocks.{i}.sa.heads.{h}.value.weight']

                key = NumpyLinear(key_w)
                query = NumpyLinear(query_w)
                value = NumpyLinear(value_w)

                head = NumpyHead(head_size, key, query, value)
                heads.append(head)

            # Projection after multi-head attention
            proj_w = w[f'blocks.{i}.sa.proj.weight']
            proj_b = w[f'blocks.{i}.sa.proj.bias']
            proj = NumpyLinear(proj_w, proj_b)

            sa = NumpyMultiHeadAttention(heads, proj)

            # Feedforward
            ffwd_w1 = w[f'blocks.{i}.ffwd.net.0.weight']
            ffwd_b1 = w[f'blocks.{i}.ffwd.net.0.bias']
            ffwd_w2 = w[f'blocks.{i}.ffwd.net.2.weight']
            ffwd_b2 = w[f'blocks.{i}.ffwd.net.2.bias']

            fc1 = NumpyLinear(ffwd_w1, ffwd_b1)
            fc2 = NumpyLinear(ffwd_w2, ffwd_b2)
            ffwd = NumpyFeedForward(fc1, fc2)

            # LayerNorms
            ln1_w = w[f'blocks.{i}.ln1.weight']
            ln1_b = w[f'blocks.{i}.ln1.bias']
            ln2_w = w[f'blocks.{i}.ln2.weight']
            ln2_b = w[f'blocks.{i}.ln2.bias']

            ln1 = NumpyLayerNorm(ln1_w, ln1_b)
            ln2 = NumpyLayerNorm(ln2_w, ln2_b)

            # Create block
            block = NumpyBlock(sa, ffwd, ln1, ln2)
            self.blocks.append(block)

        # Final layer norm
        self.ln_f = NumpyLayerNorm(w['ln_f.weight'], w['ln_f.bias'])

        # Language model head
        self.lm_head = NumpyLinear(w['lm_head.weight'], w['lm_head.bias'])

        print(f"✓ Built {N_LAYER} transformer blocks with {N_HEAD} heads each")

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices. Unknown chars replaced with space."""
        # Get space index as fallback for unknown chars
        space_idx = self.char_to_idx.get(' ', 0)
        return [self.char_to_idx.get(c, space_idx) for c in text]

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        return ''.join(self.idx_to_char[i] for i in indices)

    def forward(self, idx: np.ndarray) -> Tuple[np.ndarray, None]:
        """
        Forward pass through the model.

        Args:
            idx: Token indices, shape (B, T)

        Returns:
            logits: shape (B, T, vocab_size)
            loss: None (inference only)
        """
        B, T = idx.shape

        # Token embeddings
        tok_emb = self.token_embedding[idx]  # (B, T, n_embd)

        # Position embeddings
        pos = np.arange(T)
        pos_emb = self.position_embedding[pos]  # (T, n_embd)

        # Combine embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final layer norm
        x = self.ln_f.forward(x)

        # Language model head
        logits = self.lm_head.forward(x)  # (B, T, vocab_size)

        return logits, None

    def generate(self, prompt: str = "\n", max_tokens: int = 280,
                 temperature: float = 0.8) -> str:
        """
        Generate text autoregressively.

        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text
        """
        # Encode prompt
        idx = np.array([self.encode(prompt)], dtype=np.int64)  # (1, T)

        # Generate tokens
        for _ in range(max_tokens):
            # Crop to context window
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Forward pass
            logits, _ = self.forward(idx_cond)  # (1, T, vocab_size)

            # Take last token logits
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Softmax
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            # Sample next token
            idx_next = np.random.choice(self.vocab_size, p=probs[0])

            # Append to sequence
            idx = np.concatenate([idx, [[idx_next]]], axis=1)

        # Decode
        return self.decode(idx[0].tolist())

    def close(self):
        """Cleanup (for consistency with HAiKU API)."""
        pass


if __name__ == '__main__':
    # Test inference
    print("Initializing Sonnet Generator...")
    gen = SonnetGenerator()

    print("\nGenerating Shakespeare-style text...\n")
    output = gen.generate(prompt="\n", max_tokens=200, temperature=0.8)
    print(output)
