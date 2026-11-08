import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import errors

from agent import PriceRates, TokenUsage, estimate_cost, run_prompt


class CostEstimationTests(unittest.TestCase):
    def test_calculates_cached_and_uncached_input_separately(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=400_000)
        rates = PriceRates(
            input_per_million=1.0,
            output_per_million=2.0,
            cached_input_per_million=0.1,
        )

        self.assertAlmostEqual(estimate_cost(usage, rates) or 0, 0.84)

    def test_unknown_rates_do_not_report_a_false_zero_cost(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=0)

        self.assertIsNone(
            estimate_cost(
                usage,
                PriceRates(
                    input_per_million=None,
                    output_per_million=1.0,
                    cached_input_per_million=None,
                ),
            )
        )

    def test_cached_tokens_need_a_cached_rate(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=25)

        self.assertIsNone(
            estimate_cost(
                usage,
                PriceRates(
                    input_per_million=1.0,
                    output_per_million=1.0,
                    cached_input_per_million=None,
                ),
            )
        )

    def test_negative_environment_rate_is_rejected(self) -> None:
        with patch.dict(os.environ, {"GEMINI_INPUT_PRICE_PER_MILLION": "-1"}):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                PriceRates.from_environment()


class GeminiTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "nested" / "runs.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_records_usage_latency_and_cost_without_prompt_text(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="A concise response.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=120,
                candidates_token_count=30,
                cached_content_token_count=20,
            ),
        )
        rates = PriceRates(1.0, 2.0, 0.1)

        result = run_prompt(
            "private prompt contents",
            client=client,
            rates=rates,
            log_path=self.log_path,
        )

        self.assertEqual(result.answer, "A concise response.")
        self.assertEqual(result.usage, TokenUsage(120, 30, 20))
        event = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(event["status"], "success")
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertNotIn("private prompt contents", self.log_path.read_text(encoding="utf-8"))

    def test_creates_context_cache_and_uses_its_name(self) -> None:
        client = Mock()
        client.caches.create.return_value = SimpleNamespace(name="cached/context-1")
        client.models.generate_content.return_value = SimpleNamespace(
            text="Cached answer.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=50,
                candidates_token_count=10,
                cached_content_token_count=30,
            ),
        )

        result = run_prompt(
            "Question",
            cache_context="Stable instructions",
            cache_ttl="600s",
            client=client,
            log_path=self.log_path,
        )

        self.assertEqual(result.answer, "Cached answer.")
        client.caches.create.assert_called_once()
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["config"].cached_content, "cached/context-1")

    def test_missing_api_key_is_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                run_prompt("Hello", log_path=self.log_path)

    def test_api_error_is_logged_then_reraised(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = errors.APIError(
            503, {"message": "Temporary service issue"}
        )

        with self.assertRaises(errors.APIError):
            run_prompt("Hello", client=client, log_path=self.log_path)

        event = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_type"], "APIError")

    def test_empty_response_is_logged_and_reported(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=None)

        with self.assertRaisesRegex(RuntimeError, "no text response"):
            run_prompt("Hello", client=client, log_path=self.log_path)

        event = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_type"], "EmptyResponse")

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prompt must not be empty"):
            run_prompt("  ", client=Mock(), log_path=self.log_path)


if __name__ == "__main__":
    unittest.main()
