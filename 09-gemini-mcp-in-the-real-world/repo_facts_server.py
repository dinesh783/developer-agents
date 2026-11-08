"""Standalone MCP server for small, read-only repository facts."""

import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer

WORKSPACE_ROOT = Path.cwd().resolve()
MAX_FILES = 2_000
MAX_FILE_BYTES = 1_000_000
MAX_REPORTED_HITS = 500
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "node_modules"}

mcp = MCPServer("repo_facts")


def _count_todos(directory: str, *, root: Path) -> str:
    """Count TODO and FIXME comments in Python files beneath ``root``."""
    root = root.resolve()
    target = (root / directory).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("The requested directory must be inside the workspace.") from error
    if not target.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    hits: list[str] = []
    scanned = 0
    skipped_large = 0
    truncated = False
    for path in sorted(target.rglob("*.py")):
        relative_path = path.relative_to(target)
        if EXCLUDED_DIRECTORIES.intersection(relative_path.parts):
            continue
        if path.is_symlink():
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
        scanned += 1
        if scanned > MAX_FILES:
            truncated = True
            break
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped_large += 1
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if re.search(r"#\s*(TODO|FIXME)\b", line):
                hits.append(f"{path.relative_to(root)}:{line_number}  {line.strip()}")
                if len(hits) >= MAX_REPORTED_HITS:
                    truncated = True
                    break
        if truncated and len(hits) >= MAX_REPORTED_HITS:
            break

    if hits:
        report = f"{len(hits)} TODO/FIXME comments found:\n" + "\n".join(hits)
    else:
        report = "No TODO or FIXME comments found."
    if skipped_large:
        report += f"\nSkipped {skipped_large} Python file(s) larger than 1 MB."
    if truncated:
        report += "\nResults were capped; narrow the directory for a complete scan."
    return report


@mcp.tool()
def count_todos(directory: str = ".") -> str:
    """Count TODO and FIXME comments in Python files under a workspace directory."""
    return _count_todos(directory, root=WORKSPACE_ROOT)


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
