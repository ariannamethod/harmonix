"""
Train SentencePiece model for HAiKU tokenizer.

Creates a SentencePiece model trained on:
- 587 SEED_WORDS from haiku.py
- Expanded cloud words from cloud.db (if exists)
- Bootstrap haiku examples

Model params:
- vocab_size: 650 (based on corpus size)
- model_type: unigram (best for variable-length tokens)
- character_coverage: 1.0 (cover all characters in corpus)

Usage:
    python scripts/train_sentencepiece.py    # Run from repo root
"""

import sentencepiece as spm
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def prepare_corpus():
    """Prepare training corpus from seed words and cloud."""

    # Import SEED_WORDS
    from haiku import SEED_WORDS

    corpus_lines = []

    # Add seed words (create example sentences)
    print(f"Loading {len(SEED_WORDS)} seed words...")
    for i in range(0, len(SEED_WORDS), 5):
        # Group into 5-word chunks for context
        chunk = SEED_WORDS[i:i+5]
        corpus_lines.append(' '.join(chunk))

    # Try to load expanded words from cloud.db
    db_path = Path('state/cloud.db')
    if db_path.exists():
        print("Loading expanded cloud words from database...")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT word FROM words ORDER BY weight DESC LIMIT 500")
        cloud_words = [row[0] for row in cursor.fetchall()]

        # Add cloud words in chunks
        for i in range(0, len(cloud_words), 5):
            chunk = cloud_words[i:i+5]
            corpus_lines.append(' '.join(chunk))

        # Also add some trigram examples
        cursor.execute("SELECT word1, word2, word3 FROM trigrams ORDER BY count DESC LIMIT 200")
        for w1, w2, w3 in cursor.fetchall():
            corpus_lines.append(f"{w1} {w2} {w3}")

        conn.close()
        print(f"Added {len(cloud_words)} cloud words and trigrams")

    # Add some haiku examples for context
    example_haikus = [
        "words dance in the cloud",
        "resonance finds its own path",
        "constraint births form",
        "silence holds meaning",
        "patterns emerge from chaos",
        "the field morphs with use",
        "dissonance creates tension",
        "harmony resolves discord",
        "waves meet in space"
    ]
    corpus_lines.extend(example_haikus)

    # Write corpus to file (in scripts/ directory)
    corpus_path = Path(__file__).parent / 'training_corpus.txt'
    with open(corpus_path, 'w') as f:
        f.write('\n'.join(corpus_lines))

    print(f"Wrote {len(corpus_lines)} lines to {corpus_path}")
    return str(corpus_path)

def train_model(corpus_path: str):
    """Train SentencePiece model."""

    print("\nTraining SentencePiece model...")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix='haiku_sp',
        vocab_size=650,  # Reduced from 1000 (corpus too small for 1k)
        model_type='unigram',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<haiku>', '<line>'],  # Special tokens for haiku structure
        normalization_rule_name='nfkc_cf',  # Normalize text
        remove_extra_whitespaces=True,
        split_by_whitespace=True,
        max_sentence_length=1000
    )

    print("✓ Model trained: haiku_sp.model, haiku_sp.vocab")

def test_model():
    """Test the trained model."""

    print("\nTesting SentencePiece model...")

    sp = spm.SentencePieceProcessor()
    sp.load('haiku_sp.model')

    test_texts = [
        "what is resonance in the cloud",
        "tell me about harmony and dissonance",
        "constraint births form"
    ]

    for text in test_texts:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"\nText: {text}")
        print(f"Pieces: {pieces}")
        print(f"IDs: {ids}")
        print(f"Decoded: {sp.decode_pieces(pieces)}")

    print(f"\n✓ Vocab size: {sp.vocab_size()}")

if __name__ == '__main__':
    corpus_path = prepare_corpus()
    train_model(corpus_path)
    test_model()
    print("\n🔥 SentencePiece model ready for HAiKU v1.1!")
