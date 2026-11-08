"""Repository inspection tools for the Episode 3 Gemini agent."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 1_000_000


def check_docstring_coverage(args: dict[str, object]) -> dict[str, str]:
    """Report Python functions and classes without docstrings."""
    file_path = args.get("file_path")
    include_private = args.get("include_private", False)

    if not isinstance(file_path, str) or not file_path.strip():
        return {"error": "file_path must be a non-empty repository-relative path."}
    if not isinstance(include_private, bool):
        return {"error": "include_private must be a boolean."}

    try:
        root = REPO_ROOT.resolve(strict=True)
        candidate = (root / file_path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return {
            "error": f"'{file_path}' is not a readable path inside the repository."
        }

    if not candidate.is_file():
        return {"error": f"'{file_path}' is not a file."}
    if candidate.suffix.lower() != ".py":
        return {"error": f"'{file_path}' is not a Python file."}

    try:
        with candidate.open("rb") as source_file:
            source_bytes = source_file.read(MAX_FILE_BYTES + 1)
    except OSError as error:
        return {"error": f"Could not read '{file_path}': {error}"}

    if len(source_bytes) > MAX_FILE_BYTES:
        return {
            "error": (
                f"'{file_path}' exceeds the {MAX_FILE_BYTES}-byte file size limit."
            )
        }

    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": f"'{file_path}' is not a UTF-8 text file."}

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as error:
        return {
            "error": (
                f"Could not parse '{file_path}': syntax error on line "
                f"{error.lineno}: {error.msg}"
            )
        }

    missing: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if not include_private and node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            missing.append((node.lineno, kind, node.name))

    missing.sort()
    if not missing:
        report = f"{file_path}: every function and class has a docstring."
    else:
        details = "\n".join(
            f"line {line}: {kind} '{name}'" for line, kind, name in missing
        )
        report = f"{file_path}: {len(missing)} missing docstring(s)\n{details}"

    return {"result": report}
