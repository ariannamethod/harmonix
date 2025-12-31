"""Tests for Overthinkrose - 4-ring expansion system"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from overthinkrose import Overthinkrose


class TestOverthinkroseInit(unittest.TestCase):
    def test_init(self):
        ot = Overthinkrose()
        self.assertIsNotNone(ot)


class TestRingExpansion(unittest.TestCase):
    def setUp(self):
        self.ot = Overthinkrose()

    def test_expand_basic(self):
        core = "consciousness"
        result = self.ot.expand(core)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
