"""Load reusable Gemini workflow instructions from the project skill files."""

from pathlib import Path

SKILL_DIRECTORY = Path(__file__).with_name(".gemini") / "skills"
SKILL_NAMES = ("pr-digest", "changelog-entry")


def load_skills() -> dict[str, str]:
    """Read all required skill instructions, raising if any file is unavailable."""
    skills: dict[str, str] = {}
    for name in SKILL_NAMES:
        path = SKILL_DIRECTORY / name / "SKILL.md"
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Could not load skill instructions from {path}: {error}") from error
        if not content.strip():
            raise RuntimeError(f"Skill instructions in {path} are empty.")
        skills[name] = content
    return skills
