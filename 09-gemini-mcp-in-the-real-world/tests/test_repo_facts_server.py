import tempfile
import unittest
from pathlib import Path

from repo_facts_server import _count_todos


class CountTodosTests(unittest.TestCase):
    def test_counts_todo_and_fixme_comments_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.py"
            source.write_text(
                "value = 1  # TODO: explain this\n# FIXME handle edge case\n",
                encoding="utf-8",
            )

            result = _count_todos(".", root=root)

        self.assertIn("2 TODO/FIXME comments found", result)
        self.assertIn("sample.py:1", result)
        self.assertIn("sample.py:2", result)

    def test_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "workspace"
            root.mkdir()

            with self.assertRaisesRegex(ValueError, "inside the workspace"):
                _count_todos("..", root=root)

    def test_skips_virtual_environment_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ignored_file = root / ".venv" / "lib" / "fake.py"
            ignored_file.parent.mkdir(parents=True)
            ignored_file.write_text("# TODO should be ignored\n", encoding="utf-8")

            result = _count_todos(".", root=root)

        self.assertEqual(result, "No TODO or FIXME comments found.")


if __name__ == "__main__":
    unittest.main()
