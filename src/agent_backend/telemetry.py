"""Optional, lightweight OpenTelemetry hooks.

If OTEL_ENABLED=true and the `otel` extra is installed, traces are exported over OTLP/HTTP
and FastAPI is auto-instrumented. Otherwise `span()` is a zero-cost no-op, so the rest of
the code can create spans unconditionally.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from agent_backend.config import Settings

logger = logging.getLogger(__name__)

_tracer: Any = None


def setup_telemetry(app: Any, settings: Settings) -> None:
    global _tracer
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTEL_ENABLED is set but the 'otel' extra is not installed; skipping")
        return

    provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
    exporter = (
        OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        if settings.otel_exporter_otlp_endpoint
        else OTLPSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    _tracer = trace.get_tracer("agent_backend")
    logger.info("OpenTelemetry tracing enabled")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    if _tracer is None:
        with nullcontext():
            yield
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield
