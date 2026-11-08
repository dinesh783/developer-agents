import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import Effort, Stakes, TaskType, recommend_model, run_agent


class ModelSelectionTests(unittest.TestCase):
    def test_simple_high_volume_task_uses_fast_tier(self) -> None:
        result = recommend_model(TaskType.CLASSIFY, stakes=Stakes.LOW)

        self.assertEqual(result.profile.tier, "fast")
        self.assertEqual(result.profile.model_id, "gemini-2.5-flash-lite")
        self.assertEqual(result.effort, Effort.LOW)

    def test_code_review_uses_balanced_tier_by_default(self) -> None:
        result = recommend_model("code-review")

        self.assertEqual(result.profile.tier, "balanced")
        self.assertEqual(result.profile.model_id, "gemini-2.5-flash")
        self.assertEqual(result.effort, Effort.MEDIUM)

    def test_high_stakes_raise_simple_task_to_capable_tier(self) -> None:
        result = recommend_model("classify", stakes="high")

        self.assertEqual(result.profile.tier, "capable")
        self.assertIn("high stakes", result.reason)

    def test_explicit_effort_can_raise_but_not_lower_task_tier(self) -> None:
        simple_high_effort = recommend_model("format", effort="high", stakes="low")
        complex_low_effort = recommend_model("multi-step", effort="low", stakes="low")

        self.assertEqual(simple_high_effort.profile.tier, "capable")
        self.assertEqual(complex_low_effort.profile.tier, "capable")

    def test_model_ids_can_be_configured_with_environment_variables(self) -> None:
        with patch.dict(os.environ, {"GEMINI_MODEL_FAST": "custom-fast-model"}):
            result = recommend_model("extract", stakes="low")

        self.assertEqual(result.profile.model_id, "custom-fast-model")

    def test_unknown_task_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            recommend_model("unknown-task")


class GeminiRequestTests(unittest.TestCase):
    def test_request_uses_selected_model_and_effort_instruction(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text="Done.")

        recommendation, answer = run_agent(
            "Review this change.",
            task=TaskType.CODE_REVIEW,
            stakes=Stakes.LOW,
            client=client,
        )

        self.assertEqual(answer, "Done.")
        self.assertEqual(recommendation.profile.model_id, "gemini-2.5-flash")
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-2.5-flash")
        self.assertIn("clear answer", request["config"].system_instruction)

    def test_explicit_model_override_is_sent(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text="Done.")

        recommendation, _ = run_agent(
            "Summarize this.",
            task="extract",
            stakes="low",
            model="gemini-custom",
            client=client,
        )

        self.assertEqual(recommendation.profile.model_id, "gemini-custom")
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"],
            "gemini-custom",
        )

    def test_missing_api_key_is_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                run_agent("Hello")

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prompt must not be empty"):
            run_agent("  ", client=Mock())

    def test_empty_gemini_response_is_reported(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=None)

        with self.assertRaisesRegex(RuntimeError, "no text response"):
            run_agent("Hello", client=client)


if __name__ == "__main__":
    unittest.main()
