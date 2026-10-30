"""A calculator that evaluates arithmetic safely by walking the AST.

`eval()` on model output is remote code execution. Instead, the expression is parsed with
`ast.parse(mode="eval")` and only an explicit allow-list of node types, operators,
functions and constants is interpreted. Size limits stop resource-exhaustion inputs such
as `9**9**9`.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from agent_backend.tools.base import Tool, ToolError

MAX_EXPRESSION_LENGTH = 200
MAX_DEPTH = 30
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class UnsafeExpressionError(ToolError):
    pass


def safe_eval(expression: str) -> float | int:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("not a valid arithmetic expression") from exc
    try:
        result = _eval(tree.body, depth=0)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"math error: {exc}") from exc
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise UnsafeExpressionError("expression did not produce a number")
    return result


def _check_magnitude(value: Any) -> Any:
    if isinstance(value, complex):
        raise ToolError("result is not a real number")
    if isinstance(value, (int, float)) and abs(value) > MAX_MAGNITUDE:
        raise ToolError("result is too large")
    return value


def _eval(node: ast.AST, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise UnsafeExpressionError("expression is nested too deeply")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"unsupported literal {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise UnsafeExpressionError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > MAX_EXPONENT
            or (abs(left) > 1 and abs(right) * math.log10(abs(left)) > 100)
        ):
            raise ToolError("exponent is too large")
        return _check_magnitude(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise UnsafeExpressionError("only basic math functions are allowed")
        if node.keywords:
            raise UnsafeExpressionError("keyword arguments are not allowed")
        args = [_eval(arg, depth + 1) for arg in node.args]
        return _check_magnitude(_FUNCTIONS[node.func.id](*args))
    raise UnsafeExpressionError(f"unsupported syntax: {type(node).__name__}")


def format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ..., min_length=1, description="Arithmetic expression, e.g. '(2 + 3) * sqrt(16)'."
    )


class CalculatorTool(Tool[CalculatorArgs]):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression. Supports + - * / // % ** (or ^), parentheses, "
        "sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max and the constants pi, e, tau."
    )
    args_model = CalculatorArgs

    async def run(self, args: CalculatorArgs) -> str:
        return format_number(safe_eval(args.expression))
