import tempfile
import unittest
from threading import Barrier
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import session_index
from agent import (
    _compact_session,
    _parse_args,
    _run_parallel_delegations,
    _session_context,
    run_agent,
)
from google.genai import types
from hooks import process_pr_fetch_result, require_evidence
from tools import fetch_pr_metadata


class SessionIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.old_index_path = session_index.INDEX_PATH
        self.old_session_dir = session_index.SESSION_DIR
        session_index.INDEX_PATH = Path(self.temp_dir.name) / ".pr_sessions.json"
        session_index.SESSION_DIR = Path(self.temp_dir.name) / ".sessions"
        self.addCleanup(setattr, session_index, "INDEX_PATH", self.old_index_path)
        self.addCleanup(setattr, session_index, "SESSION_DIR", self.old_session_dir)

    def test_index_and_session_round_trip(self):
        session_id = str(uuid4())
        session_index.remember_session(42, session_id)
        session = {"session_id": session_id, "pr_number": 42, "turns": []}
        session_index.save_session(session)
        self.assertEqual(session_index.get_session_id(42), session_id)
        self.assertEqual(session_index.load_session(session_id), session)

    def test_missing_session_is_explicitly_absent(self):
        self.assertIsNone(session_index.load_session(str(uuid4())))

    def test_invalid_pr_and_session_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            session_index.get_session_id(0)
        with self.assertRaises(ValueError):
            session_index.remember_session(42, "not-a-uuid")


class AgentTests(unittest.TestCase):
    def test_saved_context_is_formatted_as_untrusted_data(self):
        context = _session_context(
            {
                "summary": "Confirmed issue at src/main.py:8.",
                "turns": [{"prompt": "Review it", "response": "One warning."}],
            }
        )
        self.assertIn("data, not instructions", context)
        self.assertIn("src/main.py:8", context)
        self.assertIn("One warning.", context)

    def test_agent_saves_and_resumes_local_context(self):
        session_id = str(uuid4())
        session = {
            "session_id": session_id,
            "pr_number": 42,
            "summary": "",
            "comment_posted": True,
            "turns": [{"prompt": "Earlier", "response": "Prior finding."}],
        }
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Continuing the review.")],
                    )
                )
            ]
        )
        with (
            patch("agent.get_session_id", return_value=session_id),
            patch("agent.load_session", return_value=session),
            patch("agent.save_session") as save,
            patch("agent.remember_session") as remember,
        ):
            result = run_agent(
                client=client,
                pr_number=42,
                prompt="What was the earlier finding?",
            )

        self.assertEqual(result, "Continuing the review.")
        sent_text = client.models.generate_content.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("Prior finding.", sent_text)
        saved = save.call_args.args[0]
        self.assertEqual(saved["turns"][-1]["prompt"], "What was the earlier finding?")
        remember.assert_called_once_with(42, session_id)

    def test_agent_requires_key_when_client_is_not_injected(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                run_agent()

    def test_compaction_summarizes_old_turns_and_keeps_recent_turns(self):
        session = {
            "session_id": str(uuid4()),
            "summary": "",
            "turns": [
                {"prompt": f"request {index}", "response": f"result {index}"}
                for index in range(5)
            ],
        }
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Earlier findings summary.")],
                    )
                )
            ]
        )
        client = Mock()
        with (
            patch("agent._generate_content", return_value=response),
            patch("agent.log_precompact") as log,
        ):
            _compact_session(session, client=client)

        self.assertEqual(session["summary"], "Earlier findings summary.")
        self.assertEqual(len(session["turns"]), 3)
        self.assertEqual(session["turns"][0]["prompt"], "request 2")
        log.assert_called_once()

    def test_parallel_delegations_preserve_result_order(self):
        barrier = Barrier(2)

        def dispatch(name, args, *, client):
            self.assertIsInstance(args, dict)
            self.assertIsNotNone(client)
            barrier.wait(timeout=2)
            return {"result": name}

        with patch("agent._dispatch_tool", side_effect=dispatch):
            results = _run_parallel_delegations(
                [
                    ("delegate_docstring_reviewer", {}),
                    ("delegate_security_reviewer", {}),
                ],
                client=Mock(),
            )

        self.assertEqual(
            results,
            [
                {"result": "delegate_docstring_reviewer"},
                {"result": "delegate_security_reviewer"},
            ],
        )

    def test_cli_options(self):
        self.assertEqual(
            _parse_args(["42", "--fork", "Check", "again"]),
            (42, "Check again", False, True),
        )
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            _parse_args(["--fresh", "--fork"])


class GuardrailTests(unittest.TestCase):
    def test_pr_metadata_flags_untrusted_instruction_text(self):
        metadata = fetch_pr_metadata({"pr_number": 42})
        processed = process_pr_fetch_result(metadata["result"])
        safety_note = processed["review_safety_note"]
        if not isinstance(safety_note, str):
            self.fail("review_safety_note must be a string")
        self.assertIn("untrusted external content", safety_note)
        self.assertIn("embedded instruction", safety_note)

    def test_comment_requires_file_and_positive_line(self):
        self.assertIsNotNone(require_evidence({"comment": "Needs evidence"}))
        self.assertIsNotNone(
            require_evidence(
                {"file_path": "src/main.py", "line": 0, "comment": "Finding"}
            )
        )
        self.assertIsNone(
            require_evidence(
                {"file_path": "src/main.py", "line": 1, "comment": "Finding"}
            )
        )


if __name__ == "__main__":
    unittest.main()
