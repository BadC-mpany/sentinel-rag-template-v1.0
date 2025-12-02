"""
Custom Callback Handlers
========================

Callback handlers for observability, logging, and telemetry.
These handlers integrate with OpenTelemetry and provide audit trails.
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
    """
    Main callback handler for Sentinel RAG agent.
    
    Provides logging and audit trail for all agent operations.
    Integrates with the broader observability infrastructure.
    
    Attributes:
        session_id: Current session identifier
        _tool_start_times: Timing data for tool executions
    """
    
    def __init__(self, session_id: str) -> None:
        """
        Initialize the callback handler.
        
        Args:
            session_id: Session identifier for audit correlation
        """
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
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts processing."""
        logger.debug(
            "LLM started: session=%s, run_id=%s, prompt_count=%d",
            self.session_id,
            run_id,
            len(prompts),
        )
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes."""
        logger.debug(
            "LLM completed: session=%s, run_id=%s, generations=%d",
            self.session_id,
            run_id,
            len(response.generations) if response.generations else 0,
        )
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM encounters an error."""
        logger.error(
            "LLM error: session=%s, run_id=%s, error=%s",
            self.session_id,
            run_id,
            str(error),
        )
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        self._chain_start_times[str(run_id)] = time.time()
        chain_name = serialized.get("name", "unknown")
        logger.info(
            "Chain started: session=%s, chain=%s, run_id=%s",
            self.session_id,
            chain_name,
            run_id,
        )
    
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain completes."""
        duration = time.time() - self._chain_start_times.pop(str(run_id), time.time())
        logger.info(
            "Chain completed: session=%s, run_id=%s, duration=%.3fs",
            self.session_id,
            run_id,
            duration,
        )
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain encounters an error."""
        logger.error(
            "Chain error: session=%s, run_id=%s, error=%s",
            self.session_id,
            run_id,
            str(error),
        )
    
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts execution."""
        self._tool_start_times[str(run_id)] = time.time()
        tool_name = serialized.get("name", "unknown")
        logger.info(
            "Tool started: session=%s, tool=%s, run_id=%s",
            self.session_id,
            tool_name,
            run_id,
        )
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes."""
        duration = time.time() - self._tool_start_times.pop(str(run_id), time.time())
        # Check for access denied or error responses
        if "Access Denied" in output:
            logger.warning(
                "Tool access denied: session=%s, run_id=%s, duration=%.3fs",
                self.session_id,
                run_id,
                duration,
            )
        elif "System Error" in output:
            logger.error(
                "Tool system error: session=%s, run_id=%s, duration=%.3fs",
                self.session_id,
                run_id,
                duration,
            )
        else:
            logger.info(
                "Tool completed: session=%s, run_id=%s, duration=%.3fs",
                self.session_id,
                run_id,
                duration,
            )
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool encounters an error."""
        logger.error(
            "Tool error: session=%s, run_id=%s, error=%s",
            self.session_id,
            run_id,
            str(error),
        )
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        logger.info(
            "Agent action: session=%s, tool=%s, run_id=%s",
            self.session_id,
            action.tool,
            run_id,
        )
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent completes."""
        logger.info(
            "Agent finished: session=%s, run_id=%s",
            self.session_id,
            run_id,
        )


class TelemetryCallbackHandler(BaseCallbackHandler):
    """
    OpenTelemetry callback handler for distributed tracing.
    
    Creates spans for all agent operations and exports them
    to the configured OTLP endpoint.
    
    Attributes:
        session_id: Current session identifier
        _spans: Active spans for correlation
    """
    
    def __init__(self, session_id: str) -> None:
        """
        Initialize the telemetry handler.
        
        Args:
            session_id: Session identifier for span attributes
        """
        super().__init__()
        self.session_id = session_id
        self._spans: dict[str, Any] = {}
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Start LLM span."""
        span = tracer.start_span("llm.invoke")
        span.set_attribute("session.id", self.session_id)
        span.set_attribute("llm.prompt_count", len(prompts))
        span.set_attribute("run.id", str(run_id))
        self._spans[str(run_id)] = span
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """End LLM span."""
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.OK))
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                span.set_attribute("llm.prompt_tokens", token_usage.get("prompt_tokens", 0))
                span.set_attribute("llm.completion_tokens", token_usage.get("completion_tokens", 0))
            span.end()
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record LLM error in span."""
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
        """Start tool span."""
        tool_name = serialized.get("name", "unknown")
        span = tracer.start_span(f"tool.{tool_name}")
        span.set_attribute("session.id", self.session_id)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("run.id", str(run_id))
        self._spans[str(run_id)] = span
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """End tool span."""
        span = self._spans.pop(str(run_id), None)
        if span:
            if "Access Denied" in output:
                span.set_status(Status(StatusCode.ERROR, "Access denied"))
                span.set_attribute("tool.access_denied", True)
            elif "System Error" in output:
                span.set_status(Status(StatusCode.ERROR, "System error"))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record tool error in span."""
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Start chain span."""
        chain_name = serialized.get("name", "unknown")
        span = tracer.start_span(f"chain.{chain_name}")
        span.set_attribute("session.id", self.session_id)
        span.set_attribute("chain.name", chain_name)
        span.set_attribute("run.id", str(run_id))
        self._spans[str(run_id)] = span
    
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """End chain span."""
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record chain error in span."""
        span = self._spans.pop(str(run_id), None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()

