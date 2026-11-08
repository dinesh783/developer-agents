"""Run a Gemini request with token, cost, latency, and error telemetry."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types


@dataclass(frozen=True)
class PriceRates:
    """Configured USD prices per million tokens."""

    input_per_million: float | None
    output_per_million: float | None
    cached_input_per_million: float | None

    @classmethod
    def from_environment(cls) -> "PriceRates":
        """Read current rates from environment variables, without stale defaults."""
        return cls(
            input_per_million=_read_rate("GEMINI_INPUT_PRICE_PER_MILLION"),
            output_per_million=_read_rate("GEMINI_OUTPUT_PRICE_PER_MILLION"),
            cached_input_per_million=_read_rate(
                "GEMINI_CACHED_INPUT_PRICE_PER_MILLION"
            ),
        )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int


@dataclass(frozen=True)
class RunResult:
    answer: str
    model: str
    duration_ms: int
    usage: TokenUsage | None
    estimated_cost_usd: float | None


def _read_rate(name: str) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        rate = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative number.") from error
    if rate < 0:
        raise ValueError(f"{name} must be a non-negative number.")
    return rate


def estimate_cost(usage: TokenUsage, rates: PriceRates) -> float | None:
    """Estimate per-request token charges when the required rates are configured."""
    if rates.input_per_million is None or rates.output_per_million is None:
        return None
    if usage.cached_input_tokens and rates.cached_input_per_million is None:
        return None

    uncached_input_tokens = max(
        0, usage.input_tokens - usage.cached_input_tokens
    )
    input_cost = (
        uncached_input_tokens * rates.input_per_million
        + usage.cached_input_tokens * (rates.cached_input_per_million or 0.0)
    ) / 1_000_000
    output_cost = usage.output_tokens * rates.output_per_million / 1_000_000
    return input_cost + output_cost


def _usage_from_response(response: Any) -> TokenUsage | None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    input_tokens = getattr(metadata, "prompt_token_count", None)
    output_tokens = getattr(metadata, "candidates_token_count", None)
    if input_tokens is None or output_tokens is None:
        return None
    return TokenUsage(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cached_input_tokens=int(
            getattr(metadata, "cached_content_token_count", None) or 0
        ),
    )


def _append_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, sort_keys=True) + "\n")


def _append_failure(
    log_path: Path,
    *,
    timestamp: str,
    model: str,
    started_at: float,
    error_type: str,
) -> None:
    _append_event(
        log_path,
        {
            "timestamp": timestamp,
            "model": model,
            "status": "error",
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
            "error_type": error_type,
        },
    )


def run_prompt(
    prompt: str,
    *,
    model: str = "gemini-2.5-flash",
    rates: PriceRates | None = None,
    cache_context: str | None = None,
    cache_ttl: str = "3600s",
    log_path: Path = Path("runs.jsonl"),
    client: Any | None = None,
) -> RunResult:
    """Send a prompt to Gemini and persist metadata without logging prompt text."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if not model.strip():
        raise ValueError("Model must not be empty.")
    if cache_context is not None and not cache_context.strip():
        raise ValueError("Cache context must not be empty.")

    active_rates = rates or PriceRates.from_environment()
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the GEMINI_API_KEY environment variable before calling Gemini."
            )
        client = genai.Client(api_key=api_key)

    started_at = time.perf_counter()
    started_timestamp = datetime.now(timezone.utc).isoformat()
    try:
        config = None
        if cache_context is not None:
            cached_content = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    display_name="production-operations-context",
                    system_instruction=cache_context,
                    ttl=cache_ttl,
                ),
            )
            cache_name = getattr(cached_content, "name", None)
            if not cache_name:
                _append_failure(
                    log_path,
                    timestamp=started_timestamp,
                    model=model,
                    started_at=started_at,
                    error_type="MissingCachedContentName",
                )
                raise RuntimeError("Gemini did not return a cached-content name.")
            config = types.GenerateContentConfig(cached_content=cache_name)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        answer = response.text
        if not answer:
            _append_failure(
                log_path,
                timestamp=started_timestamp,
                model=model,
                started_at=started_at,
                error_type="EmptyResponse",
            )
            raise RuntimeError("Gemini returned no text response.")

        duration_ms = round((time.perf_counter() - started_at) * 1000)
        usage = _usage_from_response(response)
        cost = estimate_cost(usage, active_rates) if usage is not None else None
        result = RunResult(
            answer=answer,
            model=model,
            duration_ms=duration_ms,
            usage=usage,
            estimated_cost_usd=cost,
        )
        _append_event(
            log_path,
            {
                "timestamp": started_timestamp,
                "model": model,
                "status": "success",
                "duration_ms": duration_ms,
                "usage": asdict(usage) if usage is not None else None,
                "estimated_cost_usd": cost,
            },
        )
        return result
    except errors.APIError as error:
        _append_failure(
            log_path,
            timestamp=started_timestamp,
            model=model,
            started_at=started_at,
            error_type=type(error).__name__,
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gemini prompt with token and cost telemetry."
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--cache-context-file",
        type=Path,
        help="Optional stable system context to cache for this request.",
    )
    parser.add_argument(
        "--cache-ttl",
        default="3600s",
        help="Gemini cached-content lifetime (default: 3600s).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("runs.jsonl"),
        help="JSONL telemetry file (default: runs.jsonl).",
    )
    parser.add_argument("prompt", nargs="+", help="Prompt text sent to Gemini.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_context = (
        args.cache_context_file.read_text(encoding="utf-8")
        if args.cache_context_file
        else None
    )
    try:
        result = run_prompt(
            " ".join(args.prompt),
            model=args.model,
            cache_context=cache_context,
            cache_ttl=args.cache_ttl,
            log_path=args.log_file,
        )
    except (errors.APIError, RuntimeError) as error:
        print(f"Gemini request failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(result.answer)
    print(f"\nModel: {result.model} | Latency: {result.duration_ms} ms")
    if result.usage is None:
        print("Token usage: unavailable")
    else:
        print(
            "Tokens: "
            f"{result.usage.input_tokens} input "
            f"({result.usage.cached_input_tokens} cached), "
            f"{result.usage.output_tokens} output"
        )
    if result.estimated_cost_usd is None:
        print("Estimated token cost: not configured (set Gemini rate variables)")
    else:
        print(f"Estimated token cost: ${result.estimated_cost_usd:.8f}")
    print(f"Telemetry: {args.log_file}")


if __name__ == "__main__":
    main()
