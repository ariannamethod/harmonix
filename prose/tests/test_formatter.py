"""Tests for ProseFormatter - Free-form text processing"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from formatter import ProseFormatter


class TestFormatterInit(unittest.TestCase):
    def test_init_defaults(self):
        f = ProseFormatter()
        self.assertEqual(f.min_words, 20)
        self.assertEqual(f.max_words, 1000)

    def test_init_custom(self):
        f = ProseFormatter(min_words=10, max_words=500)
        self.assertEqual(f.min_words, 10)
        self.assertEqual(f.max_words, 500)


class TestCleanText(unittest.TestCase):
    def setUp(self):
        self.formatter = ProseFormatter()

    def test_strip_whitespace(self):
        result = self.formatter.clean_text("  text  ")
        self.assertEqual(result.strip(), "text")

    def test_remove_multiple_spaces(self):
        result = self.formatter.clean_text("word1    word2")
        self.assertIn("word1 word2", result)

    def test_fix_punctuation(self):
        result = self.formatter.clean_text("hello , world !")
        self.assertIn("hello,", result)
        self.assertIn("world!", result)


class TestFormatProse(unittest.TestCase):
    def setUp(self):
        self.formatter = ProseFormatter()

    def test_format_valid_prose(self):
        text = "Words flow through consciousness like water through stone."
        result = self.formatter.format_prose(text)
        self.assertIsInstance(result, str)

    def test_format_empty_string(self):
        result = self.formatter.format_prose("")
        self.assertEqual(result, "")


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.formatter = ProseFormatter(min_words=5)

    def test_validate_sufficient_words(self):
        text = "This is a test with enough words here."
        valid, msg = self.formatter.validate(text)
        self.assertTrue(valid)

    def test_validate_too_few_words(self):
        text = "Too short"
        valid, msg = self.formatter.validate(text)
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
