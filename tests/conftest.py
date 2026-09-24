import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Answers with a fixed list of messages, in order."""

    script: list[AIMessage]
    calls: int = 0
    model: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        msg = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, args: dict[str, Any], i: int = 0) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"x{i}", "type": "tool_call"}]
    )


GOOD = [
    call("lookup_order", {"order_id": 42}, 1),
    call("refund_policy", {}, 2),
    AIMessage(content="Order 42 was opened, so you can get store credit but not a cash refund."),
]
