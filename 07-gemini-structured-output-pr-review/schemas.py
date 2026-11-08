from typing import Any

ALLOWED_SEVERITIES = ("blocking", "warning", "nit")

FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the file this finding applies to, relative to repo root.",
        },
        "line": {
            "type": "integer",
            "description": "The line number this finding applies to.",
        },
        "severity": {
            "type": "string",
            "enum": list(ALLOWED_SEVERITIES),
            "description": "Exactly one of blocking, warning, or nit. Never invent a fourth label.",
        },
        "message": {
            "type": "string",
            "description": "One or two sentences on the issue. Be specific, not generic.",
        },
        "suggested_fix": {
            "type": ["string", "null"],
            "description": (
                "A concrete code-level fix, if you're confident in one. "
                "Pass null if you aren't."
            ),
        },
    },
    "required": ["file_path", "line", "severity", "message", "suggested_fix"],
    "additionalProperties": False,
}


def validate_finding(args: dict[str, Any]) -> list[str]:
    """Validate one structured review finding and return specific correction notes."""
    errors: list[str] = []

    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        errors.append("file_path must be a non-empty string.")

    line = args.get("line")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        errors.append("line must be a positive integer (line numbers start at 1).")

    severity = args.get("severity")
    if severity not in ALLOWED_SEVERITIES:
        errors.append(
            f"severity must be exactly one of {list(ALLOWED_SEVERITIES)} -- got {severity!r}."
        )

    message = args.get("message")
    if not isinstance(message, str) or not message.strip():
        errors.append("message must be a non-empty string.")

    if "suggested_fix" not in args:
        errors.append("suggested_fix is required -- pass null if you don't have a confident fix.")
    else:
        suggested_fix = args["suggested_fix"]
        if suggested_fix is not None and not isinstance(suggested_fix, str):
            errors.append("suggested_fix must be a string or null.")

    return errors
