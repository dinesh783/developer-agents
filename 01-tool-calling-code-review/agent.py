"""A small Gemini tool-calling agent for the episode 1 demo repository."""

import argparse
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 8
MAX_FILE_BYTES = 1_000_000

REPO_FILES = {
    "README.md": "# PR Review Agent\nDemo repo for this series.",
    "src/auth.py": (
        "def check_token(token):\n"
        "    # TODO: never actually validates anything\n"
        "    return True"
    ),
    "src/utils.py": (
        "def slugify(text):\n"
        "    return text.lower().replace(' ', '-')"
    ),
}

TOOL_DESCRIPTION = (
    "Read the full text contents of a single file from the demo repository, "
    "given its path relative to the repo root. Use this whenever you need to "
    "see what a file contains before answering a question about it."
)

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="read_repo_file",
                description=TOOL_DESCRIPTION,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(
                            type=types.Type.STRING,
                            description="Path relative to the repo root, e.g. src/utils.py",
                        )
                    },
                    required=["path"],
                ),
            )
        ]
    )
]

SYSTEM_INSTRUCTION = (
    "You are a senior code reviewer. Use read_repo_file to inspect files "
    "before answering questions about their contents. Do not guess at file "
    "contents you have not read. If a file read returns an error, explain "
    "that you could not inspect the file instead of guessing."
)


def read_repo_file(path: str, repo_root: Path | None = None) -> str:
    """Read one demo file or a bounded UTF-8 file under the selected root."""
    if repo_root is None:
        try:
            return REPO_FILES[path]
        except KeyError:
            return f"Error: no file at '{path}' in this demo repository."

    try:
        root = repo_root.resolve(strict=True)
        candidate = (root / path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return f"Error: '{path}' is not a readable path inside the selected repository."

    if not candidate.is_file():
        return f"Error: '{path}' is not a file."

    try:
        with candidate.open("rb") as file:
            content = file.read(MAX_FILE_BYTES + 1)
    except OSError:
        return f"Error: could not read '{path}'."

    if len(content) > MAX_FILE_BYTES:
        return f"Error: '{path}' exceeds the {MAX_FILE_BYTES}-byte file size limit."

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file."


def _tool_result(
    name: str,
    args: dict[str, object],
    repo_root: Path | None,
) -> dict[str, str]:
    if name != "read_repo_file":
        return {"error": f"Unknown tool: {name}"}

    path = args.get("path")
    if not isinstance(path, str):
        return {"error": "The read_repo_file tool requires a string path."}

    return {"result": read_repo_file(path, repo_root)}


def run_agent(
    user_message: str,
    *,
    client: genai.Client | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    repo_root: Path | None = None,
) -> str:
    """Ask Gemini a question and fulfill any requested repository reads."""
    if not user_message.strip():
        raise ValueError("user_message must not be empty.")
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1.")

    resolved_repo_root = None
    if repo_root is not None:
        resolved_repo_root = Path(repo_root).resolve(strict=True)
        if not resolved_repo_root.is_dir():
            raise ValueError("repo_root must be an existing directory.")

    gemini = client or genai.Client()
    history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    ]

    tool_rounds = 0
    while True:
        response = gemini.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=TOOLS,
            ),
        )

        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("Gemini returned no candidate content.")

        model_content = response.candidates[0].content
        history.append(model_content)
        parts = model_content.parts or []
        function_calls = [
            part.function_call
            for part in parts
            if part.function_call is not None
        ]

        if not function_calls:
            answer = "".join(part.text or "" for part in parts)
            if not answer:
                raise RuntimeError("Gemini returned neither text nor a tool call.")
            return answer

        if tool_rounds >= max_tool_rounds:
            raise RuntimeError(
                f"Gemini requested tools for more than {max_tool_rounds} rounds."
            )

        tool_responses = []
        for function_call in function_calls:
            function_name = function_call.name
            if not function_name:
                raise RuntimeError("Gemini returned a function call without a name.")
            args = function_call.args or {}
            result = _tool_result(
                function_name,
                args,
                resolved_repo_root,
            )
            tool_responses.append(
                types.Part.from_function_response(
                    name=function_name,
                    response=result,
                )
            )

        history.append(types.Content(role="user", parts=tool_responses))
        tool_rounds += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask Gemini to review files in a demo or local repository."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Read files from this repository directory instead of the demo files.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What does src/auth.py do, and does it look finished to you?",
        help="Question to ask about the repository.",
    )
    args = parser.parse_args()
    print(
        run_agent(args.question, repo_root=args.repo_root)
    )
