import math

import pytest

from agent_backend.tools.base import ToolError
from agent_backend.tools.calculator import (
    CalculatorArgs,
    CalculatorTool,
    UnsafeExpressionError,
    format_number,
    safe_eval,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3),
        ("12 * (3 + 4)", 84),
        ("2 ** 10", 1024),
        ("2 ^ 10", 1024),
        ("-5 + +3", -2),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("10 / 4", 2.5),
        ("sqrt(16) + abs(-2)", 6.0),
        ("max(1, 9, 3) - min(4, 2)", 7),
        ("round(pi, 2)", 3.14),
        ("log10(1000)", 3.0),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "().__class__.__bases__",
        "open('/etc/passwd')",
        "x + 1",
        "'a' * 3",
        "[1, 2, 3]",
        "lambda: 1",
        "sqrt(x=4)",
        "math.sqrt(4)",
        "1 if True else 2",
        "True + 1",
    ],
)
def test_rejects_unsafe_or_unsupported_syntax(expression: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression)


@pytest.mark.parametrize("expression", ["9 ** 9 ** 9", "10 ** 5000", "2 ** 400", "(-8) ** 0.5"])
def test_rejects_resource_exhaustion_and_non_real_results(expression: str) -> None:
    with pytest.raises(ToolError):
        safe_eval(expression)


def test_rejects_overlong_and_deep_expressions() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1+" * 150 + "1")
    with pytest.raises((UnsafeExpressionError, RecursionError)):
        safe_eval("(" * 60 + "1" + ")" * 60 + "+" + "-" * 40 + "1")


def test_math_errors_become_tool_errors() -> None:
    with pytest.raises(ToolError, match="division by zero"):
        safe_eval("1 / 0")
    with pytest.raises(ToolError, match="math error"):
        safe_eval("sqrt(-1)")
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1 +")


def test_format_number() -> None:
    assert format_number(4.0) == "4"
    assert format_number(1 / 3) == "0.3333333333"
    assert format_number(math.pi) == "3.141592654"
    assert format_number(7) == "7"


async def test_tool_returns_formatted_string() -> None:
    assert await CalculatorTool().run(CalculatorArgs(expression="12 * (3 + 4)")) == "84"
