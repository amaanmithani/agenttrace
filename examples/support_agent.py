"""A small customer-support agent (LangGraph): a model node and a tool node in a loop.
Used by the tests (with a scripted model) and the case study (with a local model)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

ORDERS = {
    42: {"item": "headphones", "price": 89.0, "days_since_delivery": 12, "opened": True},
    77: {"item": "keyboard", "price": 129.0, "days_since_delivery": 45, "opened": False},
}

POLICY = (
    "Refunds: unopened items within 30 days of delivery get a full refund. "
    "Opened items within 30 days get store credit only. Nothing after 30 days."
)


@tool
def lookup_order(order_id: int) -> str:
    """Look up an order by its number."""
    o = ORDERS.get(order_id)
    return "no such order" if o is None else ", ".join(f"{k}={v}" for k, v in o.items())


@tool
def refund_policy() -> str:
    """The store's refund policy."""
    return POLICY


TOOLS = [lookup_order, refund_policy]

SYSTEM = (
    "You are a support agent. Before answering a refund question, look up the order "
    "and read the refund policy, then answer in one sentence."
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(model: Any, tools: list[Any] = TOOLS, system: str = SYSTEM) -> Any:
    bound = model.bind_tools(tools)

    def agent(state: State) -> dict[str, Any]:
        return {"messages": [bound.invoke([SystemMessage(system), *state["messages"]])]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()
