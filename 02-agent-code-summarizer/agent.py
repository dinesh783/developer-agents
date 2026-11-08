"""Summarize the Episode 1 agent using Gemini function calling."""

from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 5
EP1_AGENT = (
    Path(__file__).resolve().parent.parent
    / "01-tool-calling-code-review"
    / "agent.py"
)

PROMPT = (
    "Read 01-tool-calling-code-review/agent.py and give me a 3-bullet "
    "plain-English summary of what it does."
)

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="read_ep1_agent",
                description="Read the source code of the Episode 1 agent.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                ),
            )
        ]
    )
]

SYSTEM_INSTRUCTION = (
    "You are a senior Python reviewer. Use read_ep1_agent to inspect "
    "01-tool-calling-code-review/agent.py before summarizing it. "
    "Be concise and return exactly "
    "three plain-English bullet points."
)


def read_ep1_agent() -> str:
    """Read the Episode 1 agent source file."""
    try:
        return EP1_AGENT.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Could not read {EP1_AGENT}: {error}") from error


def run_agent(*, client: genai.Client | None = None) -> str:
    """Ask Gemini to inspect and summarize the Episode 1 agent."""
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
            if function_call.name != "read_ep1_agent":
                raise RuntimeError(
                    f"Gemini requested an unknown tool: {function_call.name}"
                )
            tool_responses.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": read_ep1_agent()},
                )
            )

        history.append(types.Content(role="user", parts=tool_responses))
        tool_rounds += 1


if __name__ == "__main__":
    print(run_agent())
