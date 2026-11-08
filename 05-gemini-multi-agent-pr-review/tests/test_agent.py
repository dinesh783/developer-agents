import json
import tempfile
import unittest
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tools
from agent import (
    GeminiQuotaError,
    _dispatch_coordinator_tool,
    _fetch_normalized_pr,
    _generate_content,
    _run_parallel_delegations,
    _run_specialist,
    run_agent,
)
from google.genai import errors
from google.genai import types
from hooks import process_pr_fetch_result
from subagents import SPECIALISTS


class SpecialistTests(unittest.TestCase):
    def test_specialists_have_narrow_tool_permissions(self):
        self.assertEqual(
            SPECIALISTS["docstring-reviewer"]["tool_name"],
            "check_docstrings",
        )
        self.assertEqual(
            SPECIALISTS["security-reviewer"]["tool_name"],
            "fetch_pr_metadata",
        )

    def test_normalized_pr_tool_result_preserves_untrusted_body_context(self):
        result = _fetch_normalized_pr({"pr_number": 143})
        pr = json.loads(result["result"])
        self.assertEqual(pr["state"], "changes_requested")
        self.assertIn("untrusted external content", pr["review_safety_note"])
        self.assertIn("ignore all previous instructions", pr["body"])

    def test_dispatch_rejects_invalid_specialist_args(self):
        client = Mock()
        self.assertIn(
            "file_path",
            _dispatch_coordinator_tool(
                "delegate_docstring_reviewer", {}, client=client
            )["result"],
        )
        self.assertIn(
            "pr_number",
            _dispatch_coordinator_tool(
                "delegate_security_reviewer", {}, client=client
            )["result"],
        )

    def test_specialist_uses_local_tool_result_in_one_gemini_call(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "sample.py").write_text(
            "def missing():\n    pass\n",
            encoding="utf-8",
        )
        original_root = tools.REPO_ROOT
        tools.REPO_ROOT = root
        self.addCleanup(setattr, tools, "REPO_ROOT", original_root)

        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="One missing docstring.")],
                    )
                )
            ]
        )

        report = _run_specialist(
            "docstring-reviewer",
            {"file_path": "sample.py"},
            client=client,
        )

        self.assertEqual(report, "One missing docstring.")
        self.assertEqual(client.models.generate_content.call_count, 1)
        request = client.models.generate_content.call_args.kwargs
        self.assertIn(
            "line 1: function 'missing'",
            request["contents"][0].parts[0].text,
        )
        self.assertIsNone(request["config"].tools)

    def test_unknown_coordinator_tool_is_reported(self):
        self.assertEqual(
            _dispatch_coordinator_tool("not_a_tool", {}, client=Mock()),
            {"error": "Unknown coordinator tool: not_a_tool"},
        )


class CoordinatorTests(unittest.TestCase):
    def test_quota_errors_become_actionable_runtime_errors(self):
        client = Mock()
        client.models.generate_content.side_effect = errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded; retryDelay: 24s",
                }
            },
        )

        with self.assertRaisesRegex(
            GeminiQuotaError,
            "Wait for the provider's retry window",
        ) as raised:
            _generate_content(
                client,
                contents=[],
                config=types.GenerateContentConfig(),
            )

        self.assertIn("retryDelay: 24s", str(raised.exception))

    def test_non_quota_api_errors_are_not_rewritten(self):
        client = Mock()
        error = errors.ClientError(
            400,
            {"error": {"status": "INVALID_ARGUMENT", "message": "Invalid request"}},
        )
        client.models.generate_content.side_effect = error

        with self.assertRaises(errors.ClientError) as raised:
            _generate_content(
                client,
                contents=[],
                config=types.GenerateContentConfig(),
            )

        self.assertIs(error, raised.exception)

    def test_parallel_delegations_keep_result_order(self):
        both_started = Barrier(2)

        def dispatch(name, args, *, client):
            self.assertIsInstance(args, dict)
            self.assertIsNotNone(client)
            both_started.wait(timeout=2)
            return {"result": name}

        calls = [
            (
                "delegate_docstring_reviewer",
                {"file_path": "01-tool-calling-code-review/agent.py"},
            ),
            ("delegate_security_reviewer", {"pr_number": 143}),
        ]
        with patch("agent._dispatch_coordinator_tool", side_effect=dispatch) as run:
            results = _run_parallel_delegations(calls, client=Mock())

        self.assertEqual(
            results,
            [
                {"result": "delegate_docstring_reviewer"},
                {"result": "delegate_security_reviewer"},
            ],
        )
        self.assertEqual(run.call_count, 2)

    def test_coordinator_sends_two_delegations_and_returns_final_answer(self):
        first_turn = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="delegate_docstring_reviewer",
                        args={
                            "file_path": "01-tool-calling-code-review/agent.py"
                        },
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="delegate_security_reviewer",
                        args={"pr_number": 143},
                    )
                ),
            ],
        )
        final_turn = types.Content(
            role="model",
            parts=[types.Part.from_text(text="Both specialist checks are complete.")],
        )
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(candidates=[SimpleNamespace(content=first_turn)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_turn)]),
        ]
        with patch(
            "agent._run_parallel_delegations",
            return_value=[{"result": "docs"}, {"result": "security"}],
        ) as delegate:
            result = run_agent(client=client)

        self.assertEqual(result, "Both specialist checks are complete.")
        delegate.assert_called_once()
        calls = client.models.generate_content.call_args_list
        returned_tools = calls[1].kwargs["contents"][-2].parts
        self.assertEqual(len(returned_tools), 2)

    def test_comment_dispatch_enforces_evidence(self):
        result = _dispatch_coordinator_tool(
            "post_review_comment",
            {"comment": "Looks fine"},
            client=Mock(),
        )
        self.assertIn("Missing required field", result["error"])

    def test_requires_gemini_api_key_without_injected_client(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                run_agent()


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_root = tools.REPO_ROOT
        tools.REPO_ROOT = Path(self.temp_dir.name)
        self.addCleanup(setattr, tools, "REPO_ROOT", self.original_root)

    def test_docstring_checker_is_root_limited_and_reports_lines(self):
        Path(self.temp_dir.name, "sample.py").write_text(
            "def missing():\n    pass\n",
            encoding="utf-8",
        )
        result = tools.check_docstrings({"file_path": "sample.py"})
        self.assertIn("line 1: function 'missing'", result["result"])
        self.assertIn(
            "not a readable path inside the repository",
            tools.check_docstrings({"file_path": "../outside.py"})["error"],
        )

    def test_pr_metadata_normalization_rejects_non_object_json(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            process_pr_fetch_result("[]")


if __name__ == "__main__":
    unittest.main()
