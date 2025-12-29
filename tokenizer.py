"""
Dual Tokenizer: SentencePiece + Trigrams
Handles both subword tokenization and co-occurrence patterns.
"""

import re
from typing import List, Tuple, Dict
from pathlib import Path

class DualTokenizer:
    """
    Dual tokenization system:
    - Subword: SentencePiece model (v1.1) or regex fallback (v1.0)
    - Trigrams: Co-occurrence patterns for resonance detection
    """

    def __init__(self, use_sentencepiece: bool = True, model_path: str = 'haiku_sp.model'):
        """
        Initialize tokenizer.

        Args:
            use_sentencepiece: If True and model exists, use SentencePiece. Otherwise fallback to regex.
            model_path: Path to SentencePiece model file.
        """
        self.use_sentencepiece = use_sentencepiece
        self.sp_model = None

        # Try to load SentencePiece model
        if use_sentencepiece and Path(model_path).exists():
            try:
                import sentencepiece as spm
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(model_path)
                print(f"✓ Loaded SentencePiece model: {model_path} (vocab={self.sp_model.vocab_size()})")
            except Exception as e:
                print(f"⚠️ Failed to load SentencePiece model: {e}")
                print("⚠️ Falling back to regex tokenization")
                self.sp_model = None

        # Regex fallback
        self.word_pattern = re.compile(r'\b\w+\b')

    def tokenize_subword(self, text: str) -> List[str]:
        """
        Subword tokenization using SentencePiece (v1.1) or regex (v1.0).

        Returns list of tokens (subwords).
        """
        if self.sp_model is not None:
            # SentencePiece: encode as pieces, remove ▁ prefix
            pieces = self.sp_model.encode_as_pieces(text.lower())
            # Remove ▁ (sentencepiece prefix) and filter empty
            tokens = [p.replace('▁', '') for p in pieces]
            return [t for t in tokens if t]
        else:
            # Regex fallback (v1.0 behavior)
            words = self.word_pattern.findall(text.lower())
            return words
    
    def tokenize_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract trigrams (3-word sequences) from text.
        Returns list of (word1, word2, word3) tuples.
        """
        words = self.tokenize_subword(text)
        if len(words) < 3:
            return []
        
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i+1], words[i+2]))
        
        return trigrams
    
    def tokenize_dual(self, text: str) -> Dict[str, List]:
        """
        Perform both tokenization methods.
        Returns dict with 'subwords' and 'trigrams' keys.
        """
        return {
            'subwords': self.tokenize_subword(text),
            'trigrams': self.tokenize_trigrams(text)
        }
