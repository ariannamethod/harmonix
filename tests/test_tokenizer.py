"""
Tests for tokenizer.py - Dual System

Tests:
- Subword tokenization (regex + SentencePiece)
- Trigram extraction
- Dual output format
"""

import pytest
from tokenizer import DualTokenizer


class TestDualTokenizer:
    """Test DualTokenizer class (regex fallback)."""

    @pytest.fixture
    def tokenizer(self):
        # Use regex tokenization for these tests (backward compat)
        return DualTokenizer(use_sentencepiece=False)

    def test_tokenize_subword_basic(self, tokenizer):
        """Test basic subword tokenization."""
        text = "hello world"
        tokens = tokenizer.tokenize_subword(text)
        assert tokens == ['hello', 'world']

    def test_tokenize_subword_punctuation(self, tokenizer):
        """Test subword tokenization strips punctuation."""
        text = "hello, world!"
        tokens = tokenizer.tokenize_subword(text)
        assert 'hello' in tokens
        assert 'world' in tokens
        assert ',' not in tokens

    def test_tokenize_subword_lowercase(self, tokenizer):
        """Test subword tokenization lowercases."""
        text = "Hello WORLD"
        tokens = tokenizer.tokenize_subword(text)
        assert tokens == ['hello', 'world']

    def test_tokenize_trigrams_enough_words(self, tokenizer):
        """Test trigram extraction with >=3 words."""
        text = "what is resonance in the cloud"
        trigrams = tokenizer.tokenize_trigrams(text)

        # 6 words → 4 trigrams
        assert len(trigrams) == 4
        assert trigrams[0] == ('what', 'is', 'resonance')
        assert trigrams[1] == ('is', 'resonance', 'in')
        assert trigrams[2] == ('resonance', 'in', 'the')
        assert trigrams[3] == ('in', 'the', 'cloud')

    def test_tokenize_trigrams_too_few_words(self, tokenizer):
        """Test trigram extraction with <3 words."""
        text = "hello world"
        trigrams = tokenizer.tokenize_trigrams(text)
        assert trigrams == []

    def test_tokenize_trigrams_exactly_three_words(self, tokenizer):
        """Test trigram extraction with exactly 3 words."""
        text = "words dance cloud"
        trigrams = tokenizer.tokenize_trigrams(text)
        assert len(trigrams) == 1
        assert trigrams[0] == ('words', 'dance', 'cloud')

    def test_tokenize_dual_format(self, tokenizer):
        """Test dual tokenization returns correct format."""
        text = "test haiku here now"
        result = tokenizer.tokenize_dual(text)

        assert 'subwords' in result
        assert 'trigrams' in result
        assert isinstance(result['subwords'], list)
        assert isinstance(result['trigrams'], list)

    def test_tokenize_dual_consistency(self, tokenizer):
        """Test dual tokenization is consistent."""
        text = "what is resonance"

        # Should match individual calls
        subwords = tokenizer.tokenize_subword(text)
        trigrams = tokenizer.tokenize_trigrams(text)
        dual = tokenizer.tokenize_dual(text)

        assert dual['subwords'] == subwords
        assert dual['trigrams'] == trigrams

    def test_empty_input(self, tokenizer):
        """Test empty input handling."""
        result = tokenizer.tokenize_dual("")
        assert result['subwords'] == []
        assert result['trigrams'] == []


class TestSentencePieceTokenizer:
    """Test DualTokenizer with SentencePiece model."""

    @pytest.fixture
    def sp_tokenizer(self):
        # Use SentencePiece if model exists
        return DualTokenizer(use_sentencepiece=True)

    def test_sentencepiece_loads_model(self, sp_tokenizer):
        """Test SentencePiece model loads successfully."""
        from pathlib import Path
        if Path('haiku_sp.model').exists():
            assert sp_tokenizer.sp_model is not None
            assert sp_tokenizer.sp_model.vocab_size() == 650

    def test_sentencepiece_tokenizes_text(self, sp_tokenizer):
        """Test SentencePiece tokenizes text into subwords."""
        text = "what is resonance"
        tokens = sp_tokenizer.tokenize_subword(text)
        # Should produce tokens (may be subwords)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_sentencepiece_trigrams_work(self, sp_tokenizer):
        """Test trigrams work with SentencePiece tokenization."""
        text = "what is resonance in the cloud"
        trigrams = sp_tokenizer.tokenize_trigrams(text)
        # Should produce trigrams
        assert len(trigrams) > 0
        assert all(isinstance(tg, tuple) for tg in trigrams)
        assert all(len(tg) == 3 for tg in trigrams)

    def test_sentencepiece_dual_format(self, sp_tokenizer):
        """Test dual format with SentencePiece."""
        text = "test haiku here"
        result = sp_tokenizer.tokenize_dual(text)
        assert 'subwords' in result
        assert 'trigrams' in result
        assert isinstance(result['subwords'], list)
        assert isinstance(result['trigrams'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
