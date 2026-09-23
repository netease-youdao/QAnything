import os
import tempfile
import unittest

from qanything_kernel.utils.path_security import safe_join, validate_filename


class PathSecurityTest(unittest.TestCase):
    def test_rejects_traversal_and_absolute_paths(self):
        for value in ("/tmp/x", "../../x", "..\\x", "C:\\x"):
            with self.assertRaises(ValueError):
                validate_filename(value)

    def test_accepts_normal_filename(self):
        self.assertEqual(validate_filename("paper.pdf"), "paper.pdf")

    def test_safe_join_rejects_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                safe_join(root, "..", "outside")


if __name__ == "__main__":
    unittest.main()
