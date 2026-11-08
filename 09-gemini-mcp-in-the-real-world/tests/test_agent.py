import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from google.genai import types

from agent import (
    MCPTool,
    _as_gemini_tools,
    _discover_tools,
    _expand_environment,
    _run_conversation,
    _to_gemini_schema,
)


class GeminiSchemaTests(unittest.TestCase):
    def test_mcp_json_schema_is_converted_for_gemini(self) -> None:
        schema = _to_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory under the workspace.",
                    }
                },
                "required": ["directory"],
            }
        )

        self.assertEqual(schema.type, types.Type.OBJECT)
        self.assertEqual(schema.required, ["directory"])
        self.assertEqual(schema.properties["directory"].type, types.Type.STRING)

    def test_function_declarations_use_namespaced_tool_names(self) -> None:
        declarations = _as_gemini_tools(
            [
                MCPTool(
                    gemini_name="repo_facts__count_todos",
                    server_name="repo_facts",
                    tool_name="count_todos",
                    description="Count TODOs.",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

        self.assertEqual(
            declarations[0].function_declarations[0].name,
            "repo_facts__count_todos",
        )

    def test_mcp_credential_reference_uses_environment_only(self) -> None:
        with patch.dict("os.environ", {"GITHUB_PAT": "test-token"}, clear=True):
            self.assertEqual(
                _expand_environment("Bearer ${GITHUB_PAT}"),
                "Bearer test-token",
            )

    def test_missing_mcp_credential_fails_explicitly(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GITHUB_PAT"):
                _expand_environment("Bearer ${GITHUB_PAT}")


class MCPDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_allowlisted_server_tools_are_exposed(self) -> None:
        allowed = SimpleNamespace(
            name="count_todos",
            description="Count comments.",
            input_schema={"type": "object", "properties": {}},
        )
        denied = SimpleNamespace(
            name="delete_file",
            description="Delete a file.",
            inputSchema={"type": "object", "properties": {}},
        )
        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[allowed, denied]))
        )

        tools = await _discover_tools({"repo_facts": session})

        self.assertEqual([tool.tool_name for tool in tools], ["count_todos"])


class GeminiToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_tool_call_is_forwarded_to_mcp(self) -> None:
        call_content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="repo_facts__count_todos",
                        args={"directory": "."},
                    )
                )
            ],
        )
        final_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text="There are two TODO comments.")],
        )
        responses = [
            SimpleNamespace(candidates=[SimpleNamespace(content=call_content)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_content)]),
        ]
        client = Mock()
        client.models.generate_content.side_effect = responses
        mcp_session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="count_todos",
                            description="Count comments.",
                            input_schema={
                                "type": "object",
                                "properties": {"directory": {"type": "string"}},
                            },
                        )
                    ]
                )
            ),
            call_tool=AsyncMock(
                return_value=SimpleNamespace(
                    content=[SimpleNamespace(text="2 TODO/FIXME comments found.")],
                    structured_content=None,
                    is_error=False,
                )
            ),
        )

        answer = await _run_conversation(
            "Count TODOs.",
            client=client,
            sessions={"repo_facts": mcp_session},
        )

        self.assertEqual(answer, "There are two TODO comments.")
        mcp_session.call_tool.assert_awaited_once_with(
            "count_todos",
            arguments={"directory": "."},
        )
        self.assertEqual(client.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()
