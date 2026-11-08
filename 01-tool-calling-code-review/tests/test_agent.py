import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types

from agent import read_repo_file, run_agent


class ReadRepoFileTests(unittest.TestCase):
    def test_reads_known_file(self):
        self.assertIn("return True", read_repo_file("src/auth.py"))

    def test_reports_unknown_file(self):
        self.assertEqual(
            read_repo_file("src/missing.py"),
            "Error: no file at 'src/missing.py' in this demo repository.",
        )

    def test_reads_file_under_selected_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text("print('hello')", encoding="utf-8")

            self.assertEqual(
                read_repo_file("sample.py", root),
                "print('hello')",
            )

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            outside = Path(temp_dir) / "secret.txt"
            outside.write_text("private", encoding="utf-8")

            result = read_repo_file("../secret.txt", root)

            self.assertIn("not a readable path", result)
            self.assertNotIn("private", result)

    def test_rejects_oversized_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "large.txt").write_bytes(b"x" * 1_000_001)

            self.assertIn(
                "exceeds the 1000000-byte file size limit",
                read_repo_file("large.txt", root),
            )


class RunAgentTests(unittest.TestCase):
    def test_reads_tool_result_then_returns_final_answer(self):
        tool_call = types.FunctionCall(
            name="read_repo_file",
            args={"path": "src/auth.py"},
        )
        tool_turn = types.Content(
            role="model",
            parts=[types.Part(function_call=tool_call)],
        )
        final_turn = types.Content(
            role="model",
            parts=[types.Part.from_text(text="It always returns True.")],
        )

        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(candidates=[SimpleNamespace(content=tool_turn)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_turn)]),
        ]

        answer = run_agent(
            "Review src/auth.py",
            client=client,
            max_tool_rounds=1,
        )

        self.assertEqual(answer, "It always returns True.")
        self.assertEqual(client.models.generate_content.call_count, 2)
        second_call_history = client.models.generate_content.call_args_list[1].kwargs[
            "contents"
        ]
        function_response = second_call_history[-2].parts[0].function_response
        self.assertEqual(function_response.name, "read_repo_file")
        self.assertEqual(
            function_response.response,
            {
                "result": (
                    "def check_token(token):\n"
                    "    # TODO: never actually validates anything\n"
                    "    return True"
                )
            },
        )

    def test_rejects_blank_user_message(self):
        with self.assertRaises(ValueError):
            run_agent("  ", client=Mock())

    def test_rejects_tool_call_without_name(self):
        tool_turn = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        args={"path": "src/auth.py"},
                    )
                )
            ],
        )
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(content=tool_turn)],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Gemini returned a function call without a name",
        ):
            run_agent("Review src/auth.py", client=client)


if __name__ == "__main__":
    unittest.main()
