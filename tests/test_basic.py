
import unittest
import os
import pandas as pd

class TestProjectStructure(unittest.TestCase):
    def test_directories_exist(self):
        self.assertTrue(os.path.exists('data/raw'))
        self.assertTrue(os.path.exists('data/processed'))
        self.assertTrue(os.path.exists('vector_store'))
        self.assertTrue(os.path.exists('src'))
        self.assertTrue(os.path.exists('notebooks'))

    def test_imports(self):
        try:
            import src
        except ImportError:
            self.fail("Could not import src")

if __name__ == '__main__':
    unittest.main()
