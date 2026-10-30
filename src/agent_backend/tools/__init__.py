from typing import Any

from agent_backend.rag.service import KnowledgeBase
from agent_backend.tools.base import Tool, ToolError, ToolRegistry, ToolResult
from agent_backend.tools.calculator import CalculatorTool
from agent_backend.tools.clock import CurrentTimeTool
from agent_backend.tools.knowledge import KnowledgeSearchTool


def build_default_registry(
    knowledge_base: KnowledgeBase | None, default_timeout_s: float = 10.0
) -> ToolRegistry:
    tools: list[Tool[Any]] = [CalculatorTool(), CurrentTimeTool()]
    if knowledge_base is not None:
        tools.append(KnowledgeSearchTool(knowledge_base))
    return ToolRegistry(tools, default_timeout_s=default_timeout_s)


__all__ = [
    "CalculatorTool",
    "CurrentTimeTool",
    "KnowledgeSearchTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]
