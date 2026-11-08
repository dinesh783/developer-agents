"""Choose a Gemini model tier for a task, then send the request."""

import argparse
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from google import genai
from google.genai import types


class TaskType(str, Enum):
    CLASSIFY = "classify"
    EXTRACT = "extract"
    FORMAT = "format"
    CODE_REVIEW = "code-review"
    MULTI_STEP = "multi-step"
    OPEN_ENDED = "open-ended"


class Stakes(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Effort(str, Enum):
    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ModelProfile:
    tier: str
    model_id: str
    tradeoff: str


@dataclass(frozen=True)
class ModelRecommendation:
    profile: ModelProfile
    effort: Effort
    reason: str


_TIER_ORDER = ("fast", "balanced", "capable")
_DEFAULT_MODELS = {
    "fast": "gemini-2.5-flash-lite",
    "balanced": "gemini-2.5-flash",
    "capable": "gemini-2.5-pro",
}
_TIER_TRADEOFFS = {
    "fast": "lowest cost and latency; best for simple, high-volume work",
    "balanced": "balanced capability, cost, and latency",
    "capable": "strongest reasoning; use when task complexity or stakes justify it",
}
_TASK_TIERS = {
    TaskType.CLASSIFY: "fast",
    TaskType.EXTRACT: "fast",
    TaskType.FORMAT: "fast",
    TaskType.CODE_REVIEW: "balanced",
    TaskType.MULTI_STEP: "capable",
    TaskType.OPEN_ENDED: "capable",
}
_EFFORT_TIERS = {
    Effort.LOW: "fast",
    Effort.MEDIUM: "balanced",
    Effort.HIGH: "capable",
}
_EFFORT_INSTRUCTIONS = {
    Effort.LOW: "Be concise. Return only the information needed to complete the task.",
    Effort.MEDIUM: "Give a clear answer with enough explanation to make it useful.",
    Effort.HIGH: (
        "Work through the task carefully. Check important assumptions and give a "
        "structured explanation of the conclusion."
    ),
}


def _enum_value(enum_type: type[Enum], value: Any, label: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"Unsupported {label} {value!r}. Choose from: {choices}.") from error


def _profile(tier: str) -> ModelProfile:
    env_name = f"GEMINI_MODEL_{tier.upper()}"
    model_id = os.environ.get(env_name, _DEFAULT_MODELS[tier]).strip()
    if not model_id:
        raise ValueError(f"{env_name} must not be empty.")
    return ModelProfile(
        tier=tier,
        model_id=model_id,
        tradeoff=_TIER_TRADEOFFS[tier],
    )


def recommend_model(
    task: TaskType | str,
    *,
    stakes: Stakes | str = Stakes.MEDIUM,
    effort: Effort | str = Effort.AUTO,
) -> ModelRecommendation:
    """Select a model tier based on task shape, stakes, and requested effort."""
    task_type = _enum_value(TaskType, task, "task")
    task_stakes = _enum_value(Stakes, stakes, "stakes")
    requested_effort = _enum_value(Effort, effort, "effort")

    tier_index = _TIER_ORDER.index(_TASK_TIERS[task_type])
    reasons = [f"{task_type.value} tasks start at the {_TIER_ORDER[tier_index]} tier"]

    if requested_effort != Effort.AUTO:
        effort_tier = _EFFORT_TIERS[requested_effort]
        tier_index = max(tier_index, _TIER_ORDER.index(effort_tier))
        reasons.append(f"{requested_effort.value} effort requires at least the {effort_tier} tier")

    if task_stakes == Stakes.HIGH:
        tier_index = max(tier_index, _TIER_ORDER.index("capable"))
        reasons.append("high stakes require the capable tier")
    elif task_stakes == Stakes.MEDIUM:
        reasons.append("medium stakes do not change the task's default tier")
    else:
        reasons.append("low stakes allow the task's default tier")

    tier = _TIER_ORDER[tier_index]
    return ModelRecommendation(
        profile=_profile(tier),
        effort=(
            Effort.MEDIUM
            if requested_effort == Effort.AUTO and tier == "balanced"
            else Effort.LOW
            if requested_effort == Effort.AUTO and tier == "fast"
            else Effort.HIGH
            if requested_effort == Effort.AUTO
            else requested_effort
        ),
        reason="; ".join(reasons) + ".",
    )


def run_agent(
    prompt: str,
    *,
    task: TaskType | str = TaskType.CODE_REVIEW,
    stakes: Stakes | str = Stakes.MEDIUM,
    effort: Effort | str = Effort.AUTO,
    model: str | None = None,
    client: Any | None = None,
) -> tuple[ModelRecommendation, str]:
    """Run a prompt with the recommended Gemini model or an explicit model ID."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    recommendation = recommend_model(task, stakes=stakes, effort=effort)
    if model is not None:
        model_id = model.strip()
        if not model_id:
            raise ValueError("Model override must not be empty.")
        recommendation = ModelRecommendation(
            profile=ModelProfile(
                tier=recommendation.profile.tier,
                model_id=model_id,
                tradeoff=recommendation.profile.tradeoff,
            ),
            effort=recommendation.effort,
            reason=recommendation.reason + " An explicit model override was supplied.",
        )

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the GEMINI_API_KEY environment variable before calling Gemini."
            )
        client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=recommendation.profile.model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_EFFORT_INSTRUCTIONS[recommendation.effort],
        ),
    )
    answer = response.text
    if not answer:
        raise RuntimeError("Gemini returned no text response.")
    return recommendation, answer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose a Gemini model for a task, then run a prompt."
    )
    parser.add_argument(
        "--task",
        choices=[item.value for item in TaskType],
        default=TaskType.CODE_REVIEW.value,
        help="Task shape used for model selection (default: code-review).",
    )
    parser.add_argument(
        "--stakes",
        choices=[item.value for item in Stakes],
        default=Stakes.MEDIUM.value,
        help="Impact if the answer is wrong (default: medium).",
    )
    parser.add_argument(
        "--effort",
        choices=[item.value for item in Effort],
        default=Effort.AUTO.value,
        help="Response depth; explicit effort can raise, but not lower, the task tier.",
    )
    parser.add_argument(
        "--model",
        help="Override the recommended model with an explicit Gemini model ID.",
    )
    parser.add_argument("prompt", nargs="+", help="Prompt text to send to Gemini.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recommendation, answer = run_agent(
        " ".join(args.prompt),
        task=args.task,
        stakes=args.stakes,
        effort=args.effort,
        model=args.model,
    )
    print(
        f"Selected {recommendation.profile.model_id} "
        f"({recommendation.profile.tier}; {recommendation.profile.tradeoff})"
    )
    print(f"Why: {recommendation.reason}")
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
