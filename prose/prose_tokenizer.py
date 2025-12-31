"""
Prose Tokenizer - Hybrid tokenization system

IMPORTANT: This does NOT replace the char-level tokenizer used for generation!
The NanoGPT model is trained on char-level (65 vocab), so generation must use that.

This tokenizer is for SEMANTIC ANALYSIS ONLY:
- MetaProse internal reflections (word-level understanding)
- ProseRAE compression (semantic features)
- Phase transitions (vocabulary tracking)
- Dream mode (concept interpolation)
- Cascade bridge to MetaHarmonix (shared vocabulary)

Hybrid approach:
- Char-level (65 vocab): Generation via transformer
- BPE/WordPiece (5000 vocab): Semantic analysis and understanding
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict


class ProseTokenizer:
    """
    Hybrid tokenizer for Prose module.

    Two modes:
    1. char_level: For generation (NanoGPT compatibility)
    2. semantic: For analysis (BPE-style subword tokenization)
    """

    def __init__(self, vocab_size: int = 5000, state_path: str = 'state/prose_tokenizer.json'):
        self.vocab_size = vocab_size
        self.state_path = state_path

        # Char-level vocabulary (for generation)
        self.char_vocab = None  # Will be set from generator
        self.char_to_idx = {}
        self.idx_to_char = {}

        # Semantic vocabulary (for analysis)
        self.semantic_vocab = {}  # token -> idx
        self.idx_to_semantic = {}  # idx -> token
        self.token_frequencies = Counter()

        # BPE merges
        self.bpe_merges = []  # List of (pair, merged_token)

        self._load_state()

    def set_char_vocab(self, chars: str):
        """Set character vocabulary from generator."""
        self.char_vocab = chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode_char(self, text: str) -> List[int]:
        """
        Character-level encoding (for generation).

        This is what the NanoGPT model expects.
        """
        if not self.char_vocab:
            raise ValueError("char_vocab not set - call set_char_vocab() first")

        return [self.char_to_idx.get(c, 0) for c in text]

    def decode_char(self, indices: List[int]) -> str:
        """Character-level decoding (from generation)."""
        if not self.idx_to_char:
            raise ValueError("char_vocab not set - call set_char_vocab() first")

        return ''.join([self.idx_to_char.get(i, '?') for i in indices])

    def train_semantic(self, texts: List[str], iterations: int = 100):
        """
        Train semantic tokenizer on corpus using BPE.

        Args:
            texts: List of prose texts
            iterations: Number of BPE merge iterations
        """
        # Initial tokenization: split on whitespace, add word boundaries
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add word boundary marker
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        # BPE iterations
        for iteration in range(iterations):
            # Count all adjacent pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge best pair
            merged = ''.join(best_pair)
            self.bpe_merges.append((best_pair, merged))

            # Update word_freqs with merged pair
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(' '.join(best_pair), merged)
                new_word_freqs[new_word] = freq

            word_freqs = new_word_freqs

            # Every 10 iterations, rebuild vocab
            if iteration % 10 == 0:
                self._rebuild_vocab(word_freqs)

        # Final vocab build
        self._rebuild_vocab(word_freqs)

        print(f"✓ Trained semantic tokenizer: {len(self.semantic_vocab)} tokens, {len(self.bpe_merges)} merges")

    def _rebuild_vocab(self, word_freqs: Dict[str, int]):
        """Rebuild semantic vocabulary from word frequencies."""
        # Extract all unique tokens
        all_tokens = set()
        for word in word_freqs.keys():
            all_tokens.update(word.split())

        # Sort by frequency (approximate via substrings)
        token_counts = Counter()
        for word, freq in word_freqs.items():
            for token in word.split():
                token_counts[token] += freq

        # Build vocab (top N tokens)
        self.semantic_vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        idx = 4

        for token, count in token_counts.most_common(self.vocab_size - 4):
            self.semantic_vocab[token] = idx
            idx += 1

        self.idx_to_semantic = {i: t for t, i in self.semantic_vocab.items()}
        self.token_frequencies = token_counts

    def encode_semantic(self, text: str) -> List[int]:
        """
        Semantic encoding using BPE tokenization.

        For analysis, not generation.
        """
        if not self.semantic_vocab:
            raise ValueError("Semantic vocab not trained - call train_semantic() first")

        # Apply BPE merges
        words = text.lower().split()
        tokens = []

        for word in words:
            # Start with character splits
            word_tokens = list(word) + ['</w>']

            # Apply merges
            for (pair, merged) in self.bpe_merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        # Convert to indices
        return [self.semantic_vocab.get(t, 1) for t in tokens]  # 1 = <unk>

    def decode_semantic(self, indices: List[int]) -> str:
        """Decode semantic tokens back to text."""
        if not self.idx_to_semantic:
            raise ValueError("Semantic vocab not trained")

        tokens = [self.idx_to_semantic.get(i, '<unk>') for i in indices]

        # Join tokens, remove word boundaries
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

    def get_semantic_features(self, text: str) -> Dict:
        """
        Extract semantic features for analysis.

        Returns:
            Dict with vocabulary richness, rare words, etc.
        """
        tokens = self.encode_semantic(text)

        # Unique token ratio
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0

        # Rare word count (bottom 20% of frequency)
        if self.token_frequencies:
            freq_threshold = sorted(self.token_frequencies.values())[
                len(self.token_frequencies) // 5
            ]
            rare_count = sum(
                1 for idx in tokens
                if idx in self.idx_to_semantic
                and self.token_frequencies.get(self.idx_to_semantic[idx], 0) < freq_threshold
            )
        else:
            rare_count = 0

        return {
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'unique_ratio': unique_ratio,
            'rare_word_count': rare_count,
            'avg_token_length': sum(
                len(self.idx_to_semantic.get(i, ''))
                for i in tokens
            ) / len(tokens) if tokens else 0.0
        }

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Token-level semantic similarity (Jaccard).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score [0, 1]
        """
        tokens1 = set(self.encode_semantic(text1))
        tokens2 = set(self.encode_semantic(text2))

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'char_vocab_size': len(self.char_vocab) if self.char_vocab else 0,
            'semantic_vocab_size': len(self.semantic_vocab),
            'bpe_merges': len(self.bpe_merges),
            'total_token_occurrences': sum(self.token_frequencies.values())
        }

    def _load_state(self):
        """Load tokenizer state from JSON."""
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            self.semantic_vocab = state.get('semantic_vocab', {})
            # Convert string keys to int for idx_to_semantic
            idx_to_sem = state.get('idx_to_semantic', {})
            self.idx_to_semantic = {int(k): v for k, v in idx_to_sem.items()}

            self.bpe_merges = [
                (tuple(pair), merged)
                for pair, merged in state.get('bpe_merges', [])
            ]

            freq_dict = state.get('token_frequencies', {})
            self.token_frequencies = Counter(freq_dict)

            print(f"✓ Loaded tokenizer state: {len(self.semantic_vocab)} semantic tokens")

        except FileNotFoundError:
            print("⚠️  No saved tokenizer state, starting fresh")

    def _save_state(self):
        """Save tokenizer state to JSON."""
        state = {
            'semantic_vocab': self.semantic_vocab,
            'idx_to_semantic': {str(k): v for k, v in self.idx_to_semantic.items()},
            'bpe_merges': [
                (list(pair), merged)
                for pair, merged in self.bpe_merges
            ],
            'token_frequencies': dict(self.token_frequencies)
        }

        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)


if __name__ == '__main__':
    tokenizer = ProseTokenizer(vocab_size=1000)  # Smaller for testing

    print("Testing Hybrid Tokenizer...\n")

    # Test corpus
    corpus = [
        """When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.""",
        """To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer.""",
        """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate."""
    ]

    # Train semantic tokenizer
    print("--- Training Semantic Tokenizer ---")
    tokenizer.train_semantic(corpus, iterations=50)

    # Test semantic encoding
    test_text = "When winter winds do blow"
    semantic_tokens = tokenizer.encode_semantic(test_text)
    print(f"\nSemantic encoding of '{test_text}':")
    print(f"Tokens: {semantic_tokens}")
    decoded = tokenizer.decode_semantic(semantic_tokens)
    print(f"Decoded: '{decoded}'")

    # Test features
    features = tokenizer.get_semantic_features(test_text)
    print(f"\nSemantic features: {features}")

    # Test similarity
    sim = tokenizer.semantic_similarity(
        "When winter winds do blow",
        "When summer heat does rise"
    )
    print(f"\nSemantic similarity: {sim:.3f}")

    # Stats
    stats = tokenizer.get_vocab_stats()
    print(f"\nVocab stats: {stats}")

    # Test char-level (requires char vocab)
    print("\n--- Char-level Encoding (for generation) ---")
    chars = "abcdefghijklmnopqrstuvwxyz .,!?'-\n"
    tokenizer.set_char_vocab(chars)

    char_tokens = tokenizer.encode_char("hello world")
    print(f"Char encoding of 'hello world': {char_tokens}")
    decoded_char = tokenizer.decode_char(char_tokens)
    print(f"Decoded: '{decoded_char}'")

    print("\n✓ Hybrid tokenizer operational")
    print("\nIMPORTANT: Char-level for generation, semantic for analysis!")
