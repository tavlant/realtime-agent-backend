"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from agent_backend.tools.base import Tool, ToolError


class CurrentTimeArgs(BaseModel):
    timezone: str = Field(
        default="UTC", max_length=64, description="IANA timezone name, e.g. 'Europe/Berlin'."
    )


class CurrentTimeTool(Tool[CurrentTimeArgs]):
    name = "current_time"
    description = "Get the current date and time in a given IANA timezone (default UTC)."
    args_model = CurrentTimeArgs

    async def run(self, args: CurrentTimeArgs) -> str:
        try:
            tz = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"unknown timezone {args.timezone!r}") from exc
        now = datetime.now(tz)
        return f"{now.strftime('%A %Y-%m-%d %H:%M:%S')} ({args.timezone})"
