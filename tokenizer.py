"""
Dual Tokenizer: SentencePiece + Trigrams
Handles both subword tokenization and co-occurrence patterns.
"""

import re
from typing import List, Tuple, Dict

class DualTokenizer:
    """
    Dual tokenization system:
    - Subword: Simple whitespace + punctuation tokenization (no SentencePiece model needed for v1)
    - Trigrams: Co-occurrence patterns for resonance detection
    """
    
    def __init__(self):
        """Initialize tokenizer with basic pattern matching."""
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def tokenize_subword(self, text: str) -> List[str]:
        """
        Simple word tokenization (v1: no trained SentencePiece model).
        Splits on whitespace and punctuation, lowercases.
        """
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
