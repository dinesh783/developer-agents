import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types

from agent import read_ep1_agent, run_agent


class AgentTests(unittest.TestCase):
    def test_reads_episode_one_source(self):
        self.assertIn("Gemini", read_ep1_agent())

    def test_reads_tool_result_then_returns_summary(self):
        tool_call = types.FunctionCall(
            name="read_ep1_agent",
            args={},
        )
        tool_turn = types.Content(
            role="model",
            parts=[types.Part(function_call=tool_call)],
        )
        final_turn = types.Content(
            role="model",
            parts=[types.Part.from_text(text="- First\n- Second\n- Third")],
        )

        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(candidates=[SimpleNamespace(content=tool_turn)]),
            SimpleNamespace(candidates=[SimpleNamespace(content=final_turn)]),
        ]

        answer = run_agent(client=client)

        self.assertEqual(answer, "- First\n- Second\n- Third")
        self.assertEqual(client.models.generate_content.call_count, 2)
        history = client.models.generate_content.call_args_list[1].kwargs["contents"]
        function_response = history[-2].parts[0].function_response
        self.assertEqual(function_response.name, "read_ep1_agent")
        self.assertIn("Gemini", function_response.response["result"])


if __name__ == "__main__":
    unittest.main()
