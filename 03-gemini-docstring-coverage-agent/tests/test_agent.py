import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types

import tools
from agent import _tool_result, run_agent
from tools import check_docstring_coverage


class CheckDocstringCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.original_root = tools.REPO_ROOT
        tools.REPO_ROOT = self.root
        self.addCleanup(setattr, tools, "REPO_ROOT", self.original_root)

    def test_reports_missing_public_definitions_in_line_order(self):
        (self.root / "sample.py").write_text(
            "class Public:\n"
            "    def documented(self):\n"
            '        """Already documented."""\n'
            "    def missing(self):\n"
            "        pass\n"
            "async def also_missing():\n"
            "    pass\n"
            "def _private():\n"
            "    pass\n",
            encoding="utf-8",
        )

        result = check_docstring_coverage({"file_path": "sample.py"})

        self.assertEqual(
            result,
            {
                "result": (
                    "sample.py: 3 missing docstring(s)\n"
                    "line 1: class 'Public'\n"
                    "line 4: function 'missing'\n"
                    "line 6: function 'also_missing'"
                )
            },
        )

    def test_can_include_private_definitions(self):
        (self.root / "sample.py").write_text(
            "def _private():\n    pass\n",
            encoding="utf-8",
        )

        result = check_docstring_coverage(
            {"file_path": "sample.py", "include_private": True}
        )

        self.assertIn("function '_private'", result["result"])

    def test_rejects_path_traversal(self):
        outside = self.root.parent / "outside.py"
        outside.write_text("secret = True", encoding="utf-8")
        self.addCleanup(outside.unlink)

        result = check_docstring_coverage({"file_path": "../outside.py"})

        self.assertIn("not a readable path inside the repository", result["error"])

    def test_reports_missing_file(self):
        result = check_docstring_coverage({"file_path": "missing.py"})

        self.assertIn("not a readable path inside the repository", result["error"])

    def test_reports_syntax_errors(self):
        (self.root / "broken.py").write_text("def (", encoding="utf-8")

        result = check_docstring_coverage({"file_path": "broken.py"})

        self.assertIn("syntax error on line 1", result["error"])

    def test_rejects_invalid_arguments(self):
        self.assertIn(
            "file_path must be a non-empty",
            check_docstring_coverage({})["error"],
        )
        self.assertIn(
            "include_private must be a boolean",
            check_docstring_coverage(
                {"file_path": "sample.py", "include_private": "yes"}
            )["error"],
        )


class RunAgentTests(unittest.TestCase):
    def test_tool_dispatch_reports_checker_result(self):
        result = _tool_result(
            "check_docstring_coverage",
            {"file_path": "02-agent-code-summarizer/agent.py"},
        )

        self.assertIn("02-agent-code-summarizer/agent.py:", result["result"])

    def test_reads_tool_result_then_returns_final_answer(self):
        tool_call = types.FunctionCall(
            name="check_docstring_coverage",
            args={"file_path": "01-tool-calling-code-review/agent.py"},
        )
        tool_turn = types.Content(
            role="model",
            parts=[types.Part(function_call=tool_call)],
        )
        final_turn = types.Content(
            role="model",
            parts=[types.Part.from_text(text="Coverage checked.")],
        )
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(candidates=[SimpleNamespace(content=tool_turn)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_turn)]),
        ]

        answer = run_agent(client=client)

        self.assertEqual(answer, "Coverage checked.")
        self.assertEqual(client.models.generate_content.call_count, 2)
        history = client.models.generate_content.call_args_list[1].kwargs["contents"]
        function_response = history[-2].parts[0].function_response
        self.assertEqual(function_response.name, "check_docstring_coverage")
        self.assertEqual(
            function_response.response,
            check_docstring_coverage(
                {"file_path": "01-tool-calling-code-review/agent.py"}
            ),
        )

    def test_rejects_unknown_tool(self):
        self.assertEqual(
            _tool_result("unknown", {}),
            {"error": "Unknown tool: unknown"},
        )


if __name__ == "__main__":
    unittest.main()
