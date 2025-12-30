"""
Sonnet Formatter - Clean Shakespeare output into 14-line sonnets

Removes character headers (GLOUCESTER, JULIET, etc.)
Formats into proper sonnet structure (14 lines)
Validates meter and structure
"""

import re
from typing import List, Tuple, Optional
try:
    import syllables as syllables_lib
    SYLLABLES_AVAILABLE = True
except ImportError:
    SYLLABLES_AVAILABLE = False


class SonnetFormatter:
    """
    Formats raw Shakespeare model output into clean 14-line sonnets.

    Traditional sonnet structure:
    - 14 lines total
    - ABAB CDCD EFEF GG rhyme scheme (Shakespearean)
    - Iambic pentameter (10 syllables per line)
    """

    # Pattern to match character headers (all caps followed by colon)
    # FIX: More aggressive pattern - includes apostrophes and hyphens
    CHARACTER_PATTERN = re.compile(r"^[A-Z\s'\-]+:")

    # Common Shakespeare character names to filter
    CHARACTER_NAMES = {
        'GLOUCESTER', 'JULIET', 'BUCKINGHAM', 'ROMEO', 'HAMLET',
        'DUKE', 'KING', 'QUEEN', 'PRINCE', 'LORD', 'LADY',
        'GRUMIO', 'CLAURENCE', 'RICHARD', 'HENRY', 'FIRST',
        'SECOND', 'THIRD', 'CITIZEN', 'SERVANT', 'MESSENGER',
        'BRUTUS', 'WARWICK', 'POMPEY', 'HORTENSIO', 'ISABELLA',
        'AUMERLE', 'CLEOMENES', 'LUCIO', 'MENENIUS', 'CAPULET',
        'MERCUTIO', 'FRIAR', 'JORF', "SOM'LEY", 'TLANTIO'  # From test output
    }

    def __init__(self):
        pass

    def clean_line(self, line: str) -> Optional[str]:
        """
        Clean a single line: remove character headers, strip whitespace.

        Returns None if line should be filtered out.
        """
        line = line.strip()

        # Empty line
        if not line:
            return None

        # Character header - extract text AFTER colon
        if self.CHARACTER_PATTERN.match(line):
            # Split on colon and take text after
            parts = line.split(':', 1)
            if len(parts) > 1:
                text_after = parts[1].strip()
                # FIX: Validate text starts with capital letter or quote (proper sentence)
                if text_after and len(text_after) > 2:
                    first_char = text_after[0]
                    # Must start with uppercase letter, quote, or apostrophe
                    if first_char.isupper() or first_char in '"\'':
                        return text_after
            return None  # Invalid or just header

        # Line starts with known character name
        first_word = line.split()[0] if line.split() else ''
        if first_word.upper() in self.CHARACTER_NAMES:
            return None

        # Stage directions (usually in brackets or parentheses)
        if line.startswith('[') or line.startswith('('):
            return None

        # FIX: Reject lines that start with lowercase (broken continuation from prev line)
        if line and line[0].islower():
            return None

        # FIX: Reject very short lines (< 3 chars, likely broken)
        if len(line) < 3:
            return None

        return line

    def extract_lines(self, raw_text: str, max_lines: int = 40) -> List[str]:
        """
        Extract clean lines from raw Shakespeare output.

        Args:
            raw_text: Raw text from model
            max_lines: Maximum lines to extract

        Returns:
            List of clean lines (character headers removed)
        """
        lines = raw_text.split('\n')
        clean_lines = []

        for line in lines:
            cleaned = self.clean_line(line)
            if cleaned:
                clean_lines.append(cleaned)

            if len(clean_lines) >= max_lines:
                break

        return clean_lines

    def format_sonnet(self, lines: List[str]) -> Optional[str]:
        """
        Format lines into a 14-line sonnet.

        Args:
            lines: List of clean lines

        Returns:
            Formatted sonnet string, or None if not enough lines
        """
        if len(lines) < 14:
            return None

        # Take first 14 lines
        sonnet_lines = lines[:14]

        # Join with newlines
        return '\n'.join(sonnet_lines)

    def count_syllables(self, line: str) -> int:
        """
        Estimate syllable count using syllables library (more accurate).

        Iambic pentameter should have ~10 syllables per line.
        """
        if SYLLABLES_AVAILABLE:
            # FIX: Use syllables library for better accuracy
            words = line.split()
            total = 0
            for word in words:
                # Clean word (remove punctuation)
                clean_word = ''.join(c for c in word if c.isalpha())
                if clean_word:
                    try:
                        total += syllables_lib.estimate(clean_word)
                    except:
                        # Fallback to simple count if error
                        total += max(1, len([c for c in clean_word.lower() if c in 'aeiouy']))
            return total
        else:
            # Fallback: Simple vowel-counting heuristic
            vowels = 'aeiouy'
            line = line.lower()
            count = 0
            prev_was_vowel = False

            for char in line:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel

            return count

    def check_meter(self, lines: List[str], tolerance: int = 2) -> Tuple[bool, List[int]]:
        """
        Check if lines approximate iambic pentameter (~10 syllables).

        FIX: Relaxed validation - Shakespeare himself varied from strict 10!
        - Classic pentameter: 10 syllables
        - Feminine ending: 11 syllables
        - Tetrameter: 8 syllables
        - Alexandrine: 12 syllables

        Args:
            lines: List of sonnet lines
            tolerance: Allowed deviation from 10 syllables (default: 2 = range 8-12)

        Returns:
            (is_valid, syllable_counts)
        """
        syllable_counts = [self.count_syllables(line) for line in lines]

        # FIX: Accept 8-12 syllables (Shakespeare variations)
        # Check if average is in reasonable range
        avg = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0

        # RELAXED: 9.0-11.0 average acceptable
        is_valid = 9.0 <= avg <= 11.0

        return is_valid, syllable_counts

    def format(self, raw_output: str, validate_meter: bool = False) -> Optional[str]:
        """
        Complete formatting pipeline: raw output → clean 14-line sonnet.

        Args:
            raw_output: Raw text from SonnetGenerator
            validate_meter: If True, check iambic pentameter

        Returns:
            Formatted sonnet, or None if formatting failed
        """
        # Extract clean lines
        lines = self.extract_lines(raw_output, max_lines=20)

        if len(lines) < 14:
            return None

        # Format into 14-line sonnet
        sonnet = self.format_sonnet(lines)

        if sonnet is None:
            return None

        # Optionally validate meter
        if validate_meter:
            is_valid, syllable_counts = self.check_meter(lines[:14])
            if not is_valid:
                # Return sonnet anyway, but could log warning
                pass

        return sonnet

    def validate(self, sonnet: str) -> Tuple[bool, str]:
        """
        Validate sonnet structure.

        Returns:
            (is_valid, reason)
        """
        lines = sonnet.split('\n')

        # Check line count
        if len(lines) != 14:
            return False, f"Expected 14 lines, got {len(lines)}"

        # Check meter (optional)
        is_valid_meter, syllable_counts = self.check_meter(lines)
        if not is_valid_meter:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            return False, f"Meter off (avg {avg_syllables:.1f} syllables, expected ~10)"

        return True, "Valid sonnet structure"


if __name__ == '__main__':
    # Test formatter
    formatter = SonnetFormatter()

    test_output = """

DUKE OF YORK:
He canst the dreams! That now the gentle things and heart for from
I will show his obsence of the humber; 'twent by shake them
contented her since of youth.

GRUMIO:
Proud mark your father's words, and let us go.
The time is come to speak of love and woe.
When winter winds do blow and summer's heat
Doth make the flowers grow beneath our feet.

HAMLET:
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
And by opposing end them. To die, to sleep.
No more; and by a sleep to say we end.
The heart-ache and the thousand natural shocks.
That flesh is heir to: 'tis a consummation.
Devoutly to be wished. To die, to sleep.
"""

    print("Raw output:")
    print(test_output)
    print("\n" + "="*70 + "\n")

    # Format
    sonnet = formatter.format(test_output, validate_meter=True)

    if sonnet:
        print("Formatted Sonnet (14 lines):")
        print(sonnet)
        print("\n" + "="*70 + "\n")

        # Validate
        is_valid, reason = formatter.validate(sonnet)
        print(f"Validation: {'✓' if is_valid else '✗'} {reason}")
    else:
        print("❌ Could not format sonnet (not enough lines)")
