import random

import pytest

from agenttrace.diff import delta, diff_runs, myers
from agenttrace.trace import Run, Step


def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


@pytest.mark.parametrize("seed", range(300))
def test_myers_is_a_minimal_valid_edit_script(seed: int) -> None:
    rng = random.Random(seed)
    a = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
    b = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
    ops = myers(a, b)
    # Replaying the script rebuilds both sequences in order.
    assert [a[i] for op, i, _ in ops if op != "added"] == a
    assert [b[j] for op, _, j in ops if op != "removed"] == b
    assert all(a[i] == b[j] for op, i, j in ops if op == "equal")
    # Minimal: the number of equal pairs is the LCS length.
    assert sum(op == "equal" for op, _, _ in ops) == lcs_len(a, b)


def run(*steps: tuple[str, str, object, object], parents: dict[int, int] | None = None) -> Run:
    parents = parents or {}
    return Run(
        [
            Step(id=i + 1, kind=k, name=n, input=inp, output=out, parent=parents.get(i + 1))  # type: ignore[arg-type]
            for i, (k, n, inp, out) in enumerate(steps)
        ]
    )


def test_identical_runs() -> None:
    a = run(("llm", "m", {"q": 1}, "x"), ("tool", "t", {"a": 1}, "y"))
    d = diff_runs(a, a)
    assert d.identical
    assert d.stats["equal"] == 2
    assert d.to_json()["first_divergence"] is None


def test_changed_prompt_is_first_divergence_even_inside_a_node() -> None:
    a = run(
        ("node", "agent", {"s": 0}, {"s": 1}),
        ("llm", "chat", {"messages": [{"role": "system", "content": "be terse"}]}, "call search"),
        ("tool", "search", {"q": "refund"}, "policy"),
        parents={2: 1},
    )
    b = run(
        ("node", "agent", {"s": 0}, {"s": 2}),
        ("llm", "chat", {"messages": [{"role": "system", "content": "be verbose"}]}, "call lookup"),
        ("tool", "lookup", {"id": 7}, "order"),
        parents={2: 1},
    )
    d = diff_runs(a, b)
    assert d.first is not None
    assert d.first.op == "changed"
    assert d.first.a is not None and d.first.a.name == "chat"
    j = d.to_json()["first_divergence"]
    assert "-be terse" in j["input_delta"] and "+be verbose" in j["input_delta"]
    assert d.stats == {"equal": 1, "changed": 1, "removed": 1, "added": 1, "output_differs": 1}


def test_same_request_different_answer() -> None:
    a = run(("tool", "clock", {}, "09:00"), ("llm", "m", {"t": "x"}, "a"))
    b = run(("tool", "clock", {}, "10:00"), ("llm", "m", {"t": "x"}, "a"))
    d = diff_runs(a, b)
    assert d.first is not None and d.first.op == "equal" and d.first.output_differs
    assert d.first.a is not None and d.first.a.name == "clock"


def test_container_output_only_divergence_is_reported_when_nothing_else_is() -> None:
    a = run(("node", "n", {}, 1), ("tool", "t", {}, "same"), parents={2: 1})
    b = run(("node", "n", {}, 2), ("tool", "t", {}, "same"), parents={2: 1})
    d = diff_runs(a, b)
    assert d.first is not None and d.first.a is not None and d.first.a.name == "n"


def test_extra_and_missing_steps() -> None:
    a = run(("tool", "t", 1, 1), ("tool", "u", 1, 1))
    b = run(("tool", "t", 1, 1), ("tool", "v", 1, 1), ("tool", "u", 1, 1))
    d = diff_runs(a, b)
    assert d.first is not None and d.first.op == "added" and d.first.b is not None and d.first.b.name == "v"
    d2 = diff_runs(b, a)
    assert d2.first is not None and d2.first.op == "removed"


def test_delta_text_forms() -> None:
    a = Step(
        1,
        "llm",
        "m",
        input={"messages": [{"role": "user", "content": "hi", "tool_calls": [{"n": 1}]}, 3], "t": 0.2},
    )
    b = Step(1, "llm", "m", input={"messages": [{"role": "user", "content": ["x"]}], "t": 0.2})
    lines = delta(a, b, "input")
    assert "-hi" in lines and '+["x"]' in lines and any("tool_calls" in x for x in lines)
    assert delta(a, a, "input") == []
    assert delta(None, Step(2, "tool", "t", output="line1\nline2"), "output")[-1] == "+line2"
