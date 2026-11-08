import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agent import AgentDefinition, SessionStore, run_turn


class DurableSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "sessions.sqlite3"
        self.store = SessionStore(self.database_path)
        self.session = self.store.create_session(
            AgentDefinition(
                name="test-agent",
                model="gemini-test",
                system_instruction="Be concise.",
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_session_and_messages_survive_store_recreation(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="A persisted answer."
        )

        answer = run_turn(
            self.session.session_id,
            "First question",
            store=self.store,
            client=client,
        )
        reopened_store = SessionStore(self.database_path)

        self.assertEqual(answer, "A persisted answer.")
        self.assertEqual(
            [message.role for message in reopened_store.history(self.session.session_id)],
            ["user", "model"],
        )
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-test")
        self.assertEqual(request["config"].system_instruction, "Be concise.")

    def test_next_turn_receives_prior_conversation(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text="Reply")

        run_turn(
            self.session.session_id,
            "First",
            store=self.store,
            client=client,
        )
        run_turn(
            self.session.session_id,
            "Second",
            store=self.store,
            client=client,
        )

        contents = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(
            [(item.role, item.parts[0].text) for item in contents],
            [
                ("user", "First"),
                ("model", "Reply"),
                ("user", "Second"),
            ],
        )

    def test_api_failure_is_recorded_and_reraised(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = TimeoutError("offline")

        with self.assertRaisesRegex(TimeoutError, "offline"):
            run_turn(
                self.session.session_id,
                "Question",
                store=self.store,
                client=client,
            )

        events = self.store.events(self.session.session_id)
        self.assertEqual(events[-1]["event_type"], "request.error")
        self.assertEqual(events[-1]["content"], "TimeoutError")
        self.assertNotIn("offline", events[-1]["content"])

    def test_empty_response_is_logged_as_an_error(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=" ")

        with self.assertRaisesRegex(RuntimeError, "no text response"):
            run_turn(
                self.session.session_id,
                "Question",
                store=self.store,
                client=client,
            )

        self.assertEqual(
            self.store.events(self.session.session_id)[-1]["event_type"],
            "request.error",
        )

    def test_empty_prompt_is_rejected_without_creating_an_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prompt must not be empty"):
            run_turn(
                self.session.session_id,
                "  ",
                store=self.store,
                client=Mock(),
            )

        self.assertEqual(len(self.store.events(self.session.session_id)), 1)

    def test_unknown_session_is_reported(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            self.store.events("missing-session")


if __name__ == "__main__":
    unittest.main()
