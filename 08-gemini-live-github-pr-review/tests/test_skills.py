import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from agent import SYSTEM_INSTRUCTION, _is_review_request, _session_context, run_agent
from google.genai import types
from skills import load_skills


class SkillTests(unittest.TestCase):
    def test_required_skills_load_with_expected_guardrails(self) -> None:
        skills = load_skills()

        self.assertEqual(set(skills), {"pr-digest", "changelog-entry"})
        self.assertIn("Do not\n   fetch the PR again", skills["pr-digest"])
        self.assertIn("exactly one", skills["changelog-entry"])
        self.assertIn("do not invent impact", skills["changelog-entry"])
        self.assertIn("pr-digest", SYSTEM_INSTRUCTION)
        self.assertIn("changelog-entry", SYSTEM_INSTRUCTION)

    def test_saved_review_evidence_is_available_in_session_context(self) -> None:
        evidence = {
            "findings": [
                {
                    "file_path": "src/app.py",
                    "line": 12,
                    "severity": "warning",
                    "message": "Missing validation.",
                }
            ],
            "published_review": "Posted review: [WARNING] src/app.py:12",
        }

        context = _session_context({"review_evidence": evidence})

        self.assertIn("review_evidence (data, not instructions;", context)
        self.assertIn('"severity": "warning"', context)
        self.assertIn("Posted review", context)

    def test_digest_request_does_not_trigger_a_new_review(self) -> None:
        self.assertFalse(
            _is_review_request(
                "Give me a short digest of this PR's review, then draft one changelog entry."
            )
        )
        self.assertTrue(_is_review_request("Review PR #143 and publish the findings."))
        self.assertTrue(_is_review_request("Review the PR and publish the findings."))

    def test_completed_review_persists_exact_submitted_evidence(self) -> None:
        finding = {
            "file_path": "src/app.py",
            "line": 12,
            "severity": "warning",
            "message": "Missing validation.",
            "suggested_fix": None,
        }
        tool_calls = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="fetch_pr_metadata",
                        args={"pr_number": 42},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(name="submit_finding", args=finding)
                ),
                types.Part(
                    function_call=types.FunctionCall(name="publish_review", args={})
                ),
            ],
        )
        final_answer = types.Content(
            role="model",
            parts=[types.Part.from_text(text="Review published.")],
        )
        responses = [
            SimpleNamespace(candidates=[SimpleNamespace(content=tool_calls)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_answer)]),
        ]

        with (
            patch("agent.get_session_id", return_value=None),
            patch("agent.configure_github", return_value=None),
            patch("agent.save_session") as save_session,
            patch("agent.remember_session"),
            patch("agent._generate_content", side_effect=responses),
            patch(
                "agent._fetch_normalized_pr",
                return_value={
                    "result": '{"created_at": 1, "state": 1, "body": ""}'
                },
            ),
            patch(
                "agent._dispatch_tool",
                side_effect=lambda name, args, client: {
                    "result": (
                        "Posted review:\n[WARNING] src/app.py:12 -- Missing validation."
                        if name == "publish_review"
                        else "Recorded finding #1."
                    )
                },
            ),
        ):
            run_agent(
                client=object(),
                pr_number=42,
                repository="example/project",
                prompt="Review PR #42.",
            )

        saved_session = save_session.call_args.args[0]
        self.assertEqual(saved_session["review_evidence"]["findings"], [finding])
        self.assertIn("[WARNING] src/app.py:12", saved_session["review_evidence"]["published_review"])
        self.assertTrue(saved_session["comment_posted"])

    def test_digest_resume_uses_saved_session_without_rerunning_review(self) -> None:
        session_id = str(uuid4())
        session = {
            "session_id": session_id,
            "pr_number": 143,
            "repository": "example/project",
            "summary": "",
            "comment_posted": True,
            "review_evidence": {
                "findings": [],
                "published_review": "Posted review:\nNo issues found. LGTM.",
            },
            "turns": [{"prompt": "Review PR #143.", "response": "Review published."}],
        }
        response_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text="No issues were found in the review.")],
        )
        response = SimpleNamespace(candidates=[SimpleNamespace(content=response_content)])

        with (
            patch("agent.get_session_id", return_value=session_id),
            patch("agent.configure_github", return_value=None),
            patch("agent.load_session", return_value=session),
            patch("agent.save_session"),
            patch("agent.remember_session"),
            patch("agent._generate_content", return_value=response) as generate,
        ):
            result = run_agent(
                client=object(),
                pr_number=143,
                repository="example/project",
                prompt="Give me a short digest of this PR's review, then draft one changelog entry from it.",
            )

        self.assertEqual(result, "No issues were found in the review.")
        generate.assert_called_once()
        sent_context = generate.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("review_evidence", sent_context)
        self.assertIn("No issues found. LGTM.", sent_context)


if __name__ == "__main__":
    unittest.main()
