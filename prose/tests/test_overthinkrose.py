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
        # Create overthinkrose with temp DB
        import tempfile
        db_path = tempfile.mktemp(suffix='.db')
        self.ot = Overthinkrose(db_path=db_path)

        # Add some test prose to cloud
        from harmonix import ProseHarmonix
        self.harmonix = ProseHarmonix(db_path=db_path)
        self.harmonix.add_prose("Test prose for expansion.", quality=0.8)
        self.harmonix.add_prose("Another prose entry here.", quality=0.7)

    def tearDown(self):
        self.ot.close()
        self.harmonix.close()

    def test_expand_basic(self):
        # Get recent prose to expand
        recent = self.harmonix.get_recent_prose(limit=2)

        # Expand (may return None if no expansions generated)
        result = self.ot.expand(recent_prose=recent, num_rings=2)

        # expand() returns None if no valid expansions
        # Just check it runs without error
        self.assertTrue(result is None or isinstance(result, dict))


if __name__ == '__main__':
    unittest.main()
