"""
Prose Formatter - Process free-form prose output

Unlike Sonnet (14 lines) and HAiKU (5-7-5), Prose has no length constraint.
This formatter cleans, validates, and structures free-form text.
"""

import re
from typing import List, Tuple, Optional, Dict


class ProseFormatter:
    """
    Formats and validates free-form prose output.

    No constraint on length - prose flows freely.
    Focus: coherence, flow, semantic density.
    """

    def __init__(self, min_words: int = 20, max_words: int = 1000):
        """
        Initialize formatter.

        Args:
            min_words: Minimum word count for valid prose
            max_words: Maximum word count (soft limit)
        """
        self.min_words = min_words
        self.max_words = max_words

    def clean_text(self, text: str) -> str:
        """
        Clean raw prose output.

        - Remove extra whitespace
        - Fix punctuation spacing
        - Remove artifacts
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove multiple newlines (but preserve paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix punctuation spacing (no space before, one after)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])([^\s\n])', r'\1 \2', text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text

    def split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs (separated by blank lines).

        Returns:
            List of paragraph strings
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)

        # Filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Simple regex-based approach (not perfect but good enough).

        Returns:
            List of sentence strings
        """
        # Split on sentence-ending punctuation followed by space/newline
        sentences = re.split(r'([.!?]+)\s+', text)

        # Recombine punctuation with sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + sentences[i + 1]
            result.append(sentence.strip())

        # Handle last sentence if no trailing punctuation split
        if len(sentences) % 2 == 1:
            last = sentences[-1].strip()
            if last:
                result.append(last)

        return [s for s in result if s]

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def compute_metrics(self, text: str) -> Dict[str, any]:
        """
        Compute prose metrics.

        Returns:
            Dictionary with:
            - word_count
            - sentence_count
            - paragraph_count
            - avg_sentence_length
            - avg_paragraph_length
        """
        paragraphs = self.split_paragraphs(text)
        sentences = self.split_sentences(text)
        words = self.count_words(text)

        return {
            'word_count': words,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sentence_length': words / len(sentences) if sentences else 0,
            'avg_paragraph_length': words / len(paragraphs) if paragraphs else 0,
        }

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Validate prose output.

        Checks:
        - Minimum word count
        - Maximum word count (soft)
        - Has at least one sentence
        - Not just garbage/repetition

        Returns:
            (is_valid, error_message)
        """
        metrics = self.compute_metrics(text)

        # Check word count
        if metrics['word_count'] < self.min_words:
            return False, f"Too short: {metrics['word_count']} < {self.min_words} words"

        if metrics['word_count'] > self.max_words:
            return False, f"Too long: {metrics['word_count']} > {self.max_words} words"

        # Check has sentences
        if metrics['sentence_count'] < 1:
            return False, "No complete sentences found"

        # Check for repetition (simple heuristic)
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:  # Less than 40% unique words
                return False, f"Too repetitive: {unique_ratio:.1%} unique words"

        return True, ""

    def format_prose(self, text: str) -> Optional[str]:
        """
        Full formatting pipeline.

        Returns:
            Cleaned and validated prose, or None if invalid
        """
        # Clean
        text = self.clean_text(text)

        # Validate
        is_valid, error = self.validate(text)
        if not is_valid:
            return None

        return text

    def format_with_metrics(self, text: str) -> Optional[Tuple[str, Dict]]:
        """
        Format prose and return with metrics.

        Returns:
            (formatted_prose, metrics) or None if invalid
        """
        formatted = self.format_prose(text)
        if formatted is None:
            return None

        metrics = self.compute_metrics(formatted)
        return formatted, metrics


# Convenience function
def format_prose(text: str, **kwargs) -> Optional[str]:
    """Format prose text (one-shot)."""
    formatter = ProseFormatter(**kwargs)
    return formatter.format_prose(text)


if __name__ == "__main__":
    # Test formatter
    print("=== Testing Prose Formatter ===\n")

    formatter = ProseFormatter()

    test_cases = [
        # Valid prose
        """Resonance is the interplay between the sound, the words, and the message.
        It's the way the words come together to create a sound and a feeling.
        It's the way you hear the words, and the way they resonate with you.

        When we speak of resonance in the context of language, we touch upon
        something deeper than mere communication. We approach the threshold
        where meaning and music converge.""",

        # Too short
        "Just a few words.",

        # Repetitive
        "The the the the the the the the the the the the the the the the.",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {text[:80]}...")
        result = formatter.format_with_metrics(text)

        if result:
            prose, metrics = result
            print(f"✓ Valid prose")
            print(f"Metrics: {metrics}")
        else:
            is_valid, error = formatter.validate(text)
            print(f"✗ Invalid: {error}")

        print("-" * 60 + "\n")
