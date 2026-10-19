from types import SimpleNamespace

from supply_chain_agent.agent import answer_with_tools
from supply_chain_agent.data_repository import SupplyChainRepository


class FakeAssistantMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, *, exclude_none: bool):
        result = {"role": "assistant"}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return result


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="get_inventory",
                    arguments='{"product_id": "SKU-102"}',
                ),
            )
            message = FakeAssistantMessage(tool_calls=[tool_call])
        else:
            tool_result = self.calls[1]["messages"][-1]
            assert tool_result["role"] == "tool"
            assert '"available_units": 100' in tool_result["content"]
            message = FakeAssistantMessage(
                content="SKU-102 has 100 available units."
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_executes_requested_tool_and_returns_final_answer() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    answer = answer_with_tools(
        "What is the available stock of SKU-102?",
        repository=SupplyChainRepository(),
        client=client,
        model="fake-model",
    )

    assert answer == "SKU-102 has 100 available units."
    assert len(completions.calls) == 2
    assert completions.calls[0]["tools"]


def test_agent_can_answer_without_using_a_tool() -> None:
    message = FakeAssistantMessage(content="Hello! How can I help?")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    answer = answer_with_tools(
        "Hello",
        repository=SupplyChainRepository(),
        client=client,
        model="fake-model",
    )

    assert answer == "Hello! How can I help?"
