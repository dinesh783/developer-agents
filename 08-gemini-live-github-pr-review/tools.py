import ast
import json
from datetime import datetime
from pathlib import Path

from github_api import GitHubAPIError, GitHubClient
from schemas import validate_finding

MAX_FILE_BYTES = 1_000_000

_GITHUB_CLIENT: GitHubClient | None = None
_ACTIVE_PR: dict[str, object] | None = None


def configure_github(repository: str, pr_number: int) -> dict[str, str] | None:
    """Set the live GitHub context for one run and validate its pull request."""
    global _GITHUB_CLIENT, _ACTIVE_PR
    try:
        client = GitHubClient(repository)
        pr = client.get_pull_request(pr_number)
    except (GitHubAPIError, ValueError, RuntimeError) as error:
        _GITHUB_CLIENT = None
        _ACTIVE_PR = None
        return {"error": str(error)}
    _GITHUB_CLIENT = client
    _ACTIVE_PR = pr
    return None


def _require_github_context() -> tuple[GitHubClient, dict[str, object]]:
    if _GITHUB_CLIENT is None or _ACTIVE_PR is None:
        raise RuntimeError("Live GitHub context is not configured. Select a repository and pull request first.")
    return _GITHUB_CLIENT, _ACTIVE_PR


def check_docstrings(args: dict[str, object]) -> dict[str, str]:
    """Return missing top-level docstrings in the PR's version of a Python file."""
    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return {"error": "file_path must be a non-empty repository-relative path."}

    normalized_path = file_path.replace("\\", "/")
    if normalized_path.startswith("/") or any(part in {".", ".."} for part in normalized_path.split("/")):
        return {"error": f"'{file_path}' must be a repository-relative file path."}
    if Path(normalized_path).suffix.lower() != ".py":
        return {"error": f"'{file_path}' is not a Python file."}

    try:
        client, pr = _require_github_context()
        head = pr.get("head")
        if not isinstance(head, dict) or not isinstance(head.get("sha"), str):
            raise GitHubAPIError("GitHub pull request response has no head commit.")
        source = client.get_file(normalized_path, head["sha"])
    except (GitHubAPIError, RuntimeError) as error:
        return {"error": f"Could not read '{file_path}' from the PR: {error}"}
    if len(source.encode("utf-8")) > MAX_FILE_BYTES:
        return {"error": f"'{file_path}' exceeds the {MAX_FILE_BYTES}-byte file size limit."}
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as error:
        return {"error": f"Could not parse '{file_path}': syntax error on line {error.lineno}: {error.msg}"}

    missing: list[tuple[int, str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            missing.append((node.lineno, kind, node.name))
    if not missing:
        report = f"{file_path}: every top-level function and class has a docstring."
    else:
        details = "\n".join(f"line {line}: {kind} '{name}'" for line, kind, name in missing)
        report = f"{file_path}: {len(missing)} missing docstring(s)\n{details}"
    return {"result": report}


def fetch_pr_metadata(args: dict[str, object]) -> dict[str, str]:
    """Return live PR metadata and changed-file patches from GitHub."""
    pr_number = args.get("pr_number")
    if not isinstance(pr_number, int) or isinstance(pr_number, bool):
        return {"error": "pr_number must be an integer."}
    try:
        client, pr = _require_github_context()
        if pr.get("number") != pr_number:
            raise GitHubAPIError(
                f"Configured PR is #{pr.get('number')}; requested #{pr_number}."
            )
        files = client.list_pull_request_files(pr_number)
        created_at = pr.get("created_at")
        head = pr.get("head")
        base = pr.get("base")
        if not isinstance(created_at, str) or not isinstance(head, dict) or not isinstance(base, dict):
            raise GitHubAPIError("GitHub returned incomplete pull request metadata.")
        normalized = {
            "created_at": int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()),
            "state": 1 if pr.get("state") == "open" else 3,
            "body": pr.get("body") or "",
            "title": pr.get("title", ""),
            "html_url": pr.get("html_url", ""),
            "repository": client.repository,
            "head": {"ref": head.get("ref"), "sha": head.get("sha")},
            "base": {"ref": base.get("ref"), "sha": base.get("sha")},
            "merged": pr.get("merged", False),
            "changed_files": [
                {
                    key: item[key]
                    for key in ("filename", "status", "additions", "deletions", "patch")
                    if key in item
                }
                for item in files
            ],
        }
        return {"result": json.dumps(normalized, ensure_ascii=False)}
    except (GitHubAPIError, RuntimeError, ValueError, OverflowError) as error:
        return {"error": str(error)}


_FINDINGS: list[dict[str, object]] = []


def reset_findings() -> None:
    """Clear all previously submitted findings."""
    _FINDINGS.clear()


def submit_finding(args: dict[str, object]) -> dict[str, str]:
    """Validate and store one structured review finding."""
    errors = validate_finding(args)
    if errors:
        return {"error": "Finding rejected -- fix and resubmit: " + " ".join(errors)}
    _FINDINGS.append(args)
    return {
        "result": (
            f"Recorded finding #{len(_FINDINGS)}: [{args['severity']}] "
            f"{args['file_path']}:{args['line']}"
        )
    }


def _format_review_comment() -> str:
    if not _FINDINGS:
        return "No issues found. LGTM."
    order = {"blocking": 0, "warning": 1, "nit": 2}
    lines = []
    for finding in sorted(_FINDINGS, key=lambda f: order[f["severity"]]):
        lines.append(
            f"[{finding['severity'].upper()}] {finding['file_path']}:{finding['line']} -- {finding['message']}"
        )
        if finding.get("suggested_fix"):
            lines.append(f"    suggested fix: {finding['suggested_fix']}")
    return "\n".join(lines)


def publish_review(args: dict[str, object] | None = None) -> dict[str, str]:
    """Post one real GitHub review containing all submitted findings."""
    try:
        client, pr = _require_github_context()
    except RuntimeError as error:
        return {"error": str(error)}
    head = pr.get("head")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str):
        return {"error": "GitHub pull request response has no head commit."}
    comment = _format_review_comment()
    try:
        result = client.create_review(
            int(pr["number"]),
            commit_id=head["sha"],
            body=comment,
        )
    except (GitHubAPIError, RuntimeError, ValueError) as error:
        return {"error": f"Could not publish GitHub review: {error}"}
    return {"result": f"Posted GitHub review: {result['html_url']}\n{comment}"}
