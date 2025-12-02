"""
Custom Callback Handlers for observability, logging, and telemetry.
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SentinelCallbackHandler(BaseCallbackHandler):
    """Main callback handler for Sentinel RAG agent."""
    
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._tool_start_times: dict[str, float] = {}
        self._chain_start_times: dict[str, float] = {}
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        logger.debug("LLM started: session=%s, run_id=%s", self.session_id, run_id)
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.debug("LLM completed: session=%s, run_id=%s", self.session_id, run_id)
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.error("LLM error: session=%s, error=%s", self.session_id, str(error))
    
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times[str(run_id)] = time.time()
        tool_name = serialized.get("name", "unknown")
        logger.info("Tool started: session=%s, tool=%s", self.session_id, tool_name)
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        duration = time.time() - self._tool_start_times.pop(str(run_id), time.time())
        logger.info("Tool completed: session=%s, duration=%.3fs", self.session_id, duration)
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.error("Tool error: session=%s, error=%s", self.session_id, str(error))
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.info("Agent action: session=%s, tool=%s", self.session_id, action.tool)
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.info("Agent finished: session=%s", self.session_id)


class TelemetryCallbackHandler(BaseCallbackHandler):
    """OpenTelemetry callback handler for distributed tracing."""
    
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._spans: dict[str, Any] = {}
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = tracer.start_span("llm.invoke")
        span.set_attribute("session.id", self.session_id)
        self._spans[str(run_id)] = span
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        span = tracer.start_span(f"tool.{tool_name}")
        span.set_attribute("session.id", self.session_id)
        self._spans[str(run_id)] = span
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()

