import os
import tempfile
import unittest

from qanything_kernel.utils.path_security import safe_join, validate_filename


class FilenameValidationTest(unittest.TestCase):
    def test_accepts_normal_filename(self):
        self.assertEqual(validate_filename("report.pdf"), "report.pdf")

    def test_rejects_absolute_and_traversal_names(self):
        for name in ("/tmp/marker", "../../marker", "..\\marker", "C:\\marker"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_filename(name)

    def test_rejects_special_components(self):
        for name in (".", "..", "bad\x00name", "CON.txt", "report.txt ", "name:stream"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_filename(name)


class SafeJoinTest(unittest.TestCase):
    def test_keeps_paths_inside_root(self):
        with tempfile.TemporaryDirectory() as root:
            result = safe_join(root, "user", "kb", "file.txt")
            self.assertEqual(result, os.path.join(os.path.realpath(root), "user", "kb", "file.txt"))

    def test_rejects_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                safe_join(root, "..", "outside.txt")


if __name__ == "__main__":
    unittest.main()
