"""A Gemini agent that checks docstring coverage in the example repositories."""

from google import genai
from google.genai import types

from tools import check_docstring_coverage

MODEL = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 6

PROMPT = (
    "Check docstring coverage on 01-tool-calling-code-review/agent.py "
    "and 02-agent-code-summarizer/agent.py. "
    "For each file, tell me what's missing and whether it's worth fixing "
    "before merging."
)

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="check_docstring_coverage",
                description=(
                    "Parse a Python file in the repository and report every "
                    "function and class missing a docstring, with line numbers. "
                    "Private functions and methods are skipped unless "
                    "include_private is true."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "file_path": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Path to a Python file relative to the repository "
                                "root, such as "
                                "01-tool-calling-code-review/agent.py."
                            ),
                        ),
                        "include_private": types.Schema(
                            type=types.Type.BOOLEAN,
                            description=(
                                "Whether to include private functions and classes."
                            ),
                        ),
                    },
                    required=["file_path"],
                ),
            )
        ]
    )
]

SYSTEM_INSTRUCTION = (
    "You are a senior Python reviewer. Use check_docstring_coverage on each "
    "requested file before assessing its docstrings. Base your answer on the "
    "tool results, and be concise and specific. If a tool reports an error, "
    "explain that you could not inspect that file instead of guessing."
)


def _tool_result(name: str, args: object) -> dict[str, str]:
    """Validate and execute a Gemini function call."""
    if name != "check_docstring_coverage":
        return {"error": f"Unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"error": "The tool arguments must be an object."}

    return check_docstring_coverage(args)


def run_agent(*, client: genai.Client | None = None) -> str:
    """Ask Gemini to inspect docstring coverage in the first two episodes."""
    gemini = client or genai.Client()
    history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=PROMPT)],
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

        if tool_rounds >= MAX_TOOL_ROUNDS:
            raise RuntimeError(
                f"Gemini requested tools for more than {MAX_TOOL_ROUNDS} rounds."
            )

        tool_responses = []
        for function_call in function_calls:
            result = _tool_result(
                function_call.name,
                function_call.args or {},
            )
            tool_responses.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=result,
                )
            )

        history.append(types.Content(role="user", parts=tool_responses))
        tool_rounds += 1


if __name__ == "__main__":
    print(run_agent())
