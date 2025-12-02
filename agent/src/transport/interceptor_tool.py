"""
Sentinel Secure Tool

Agent sends POST requests to Interceptor. Agent cannot communicate with MCP directly.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Import pending results from main
_pending_results = None

def set_pending_results_store(store: dict) -> None:
    """Set the pending results store from main module."""
    global _pending_results
    _pending_results = store


class SecurityBlockException(ToolException):
    """Exception raised when a security policy blocks a tool execution."""

    def __init__(self, message: str, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' was blocked by a security policy. Reason: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class SentinelSecureTool(BaseTool):
    """
    Tool wrapper that sends POST requests to Interceptor.
    Agent cannot communicate with MCP directly.
    """

    def __init__(self, name: str, description: str, args_schema: Optional[type[BaseModel]] = None, timeout_seconds: float = 30.0, **kwargs: Any):
        super().__init__(name=name, description=description, args_schema=args_schema, **kwargs)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "_api_key", os.getenv("SENTINEL_API_KEY"))
        object.__setattr__(self, "_interceptor_url", os.getenv("SENTINEL_URL", "http://localhost:8000"))
        object.__setattr__(self, "_session_id", None)

        if not self._api_key:
            logger.warning("SENTINEL_API_KEY is not set - tool calls will fail")

    def set_session_id(self, session_id: str) -> None:
        object.__setattr__(self, "_session_id", session_id)

    def _parse_input(self, *args: Any, **kwargs: Any) -> dict:
        if kwargs:
            return kwargs

        if args:
            tool_input = args[0]
            if isinstance(tool_input, dict):
                return tool_input
            try:
                return json.loads(tool_input)
            except (json.JSONDecodeError, TypeError):
                if self.args_schema:
                    model_fields = getattr(self.args_schema, 'model_fields', None) or getattr(self.args_schema, '__fields__', {})
                    if len(model_fields) == 1:
                        field_name = list(model_fields.keys())[0]
                        return {field_name: tool_input}
                raise ValueError(f"Received a non-JSON string '{tool_input}' and could not map it to the tool schema.")

        return {}

    def _run(self, *args: Any, **kwargs: Any) -> str:
        if self._session_id is None:
            raise ValueError("Session ID must be set before running the tool.")

        with tracer.start_as_current_span(f"tool.{self.name}") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("session.id", self._session_id)

            try:
                args_dict = self._parse_input(*args, **kwargs)
            except ValueError as e:
                return f"Input Parsing Error: {e}"

            agent_url = os.environ.get("AGENT_URL", "http://localhost:8001")
            callback_url = f"{agent_url}/tool-result"
            
            payload = {
                "session_id": self._session_id,
                "tool_name": self.name,
                "args": args_dict,
                "agent_callback_url": callback_url,
            }
            headers = {
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            }

            logger.info("Tool invocation: name=%s, session=%s, args=%s", self.name, self._session_id, args_dict)
            logger.info("Sending request to Interceptor: url=%s/v1/proxy-execute", self._interceptor_url)

            try:
                with httpx.Client() as client:
                    response = client.post(
                        f"{self._interceptor_url}/v1/proxy-execute",
                        json=payload,
                        headers=headers,
                        timeout=5.0,  # Short timeout for request acceptance
                    )
                    logger.info("Interceptor response: status=%d, body=%s", response.status_code, response.text[:200])
                    response.raise_for_status()
                
                # Wait for result from MCP (sent directly to Agent)
                result_key = f"{self._session_id}:{self.name}"
                max_wait = self.timeout_seconds
                start_time = time.time()
                
                logger.info("=" * 60)
                logger.info("Waiting for tool result:")
                logger.info("  Result Key: %s", result_key)
                logger.info("  Session ID: %s", self._session_id)
                logger.info("  Tool Name: %s", self.name)
                logger.info("  Timeout: %s seconds", max_wait)
                logger.info("  Pending results store: %s", "initialized" if _pending_results is not None else "NOT INITIALIZED")
                if _pending_results:
                    logger.info("  Current pending keys: %s", list(_pending_results.keys())[:10])
                logger.info("=" * 60)
                
                while time.time() - start_time < max_wait:
                    if _pending_results is None:
                        logger.error("Pending results store is None!")
                        time.sleep(0.1)
                        continue
                    
                    if result_key in _pending_results:
                        result_data = _pending_results.pop(result_key)
                        logger.info("=" * 60)
                        logger.info("Received tool result:")
                        logger.info("  Key: %s", result_key)
                        logger.info("  Success: %s", result_data["success"])
                        logger.info("  Result: %s", str(result_data.get("result", ""))[:200])
                        logger.info("=" * 60)
                        if result_data["success"]:
                            return str(result_data["result"])
                        else:
                            return f"Error: {result_data.get('error', 'Unknown error')}"
                    
                    # Log available keys periodically
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0 and elapsed > 0:
                        logger.debug("Still waiting... elapsed=%.1fs, available keys: %s", 
                                    elapsed, list(_pending_results.keys())[:5] if _pending_results else [])
                    
                    time.sleep(0.1)
                
                logger.warning("=" * 60)
                logger.warning("TIMEOUT waiting for tool result:")
                logger.warning("  Key: %s", result_key)
                logger.warning("  Waited: %.1f seconds", max_wait)
                if _pending_results:
                    logger.warning("  Available keys: %s", list(_pending_results.keys()))
                logger.warning("=" * 60)
                return f"Error: Timeout waiting for tool result from MCP (key: {result_key})"

            except httpx.HTTPStatusError as e:
                try:
                    detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    detail = e.response.text

                if "Unreachable" in detail or "upstream" in detail.lower():
                    logger.error(f"MCP server unreachable for tool '{self.name}': {detail}")
                    return f"Error: MCP server is not reachable. Please ensure the MCP server is running."
                
                logger.warning(f"Access Denied for tool '{self.name}': {detail}")
                raise SecurityBlockException(
                    message=f"Tool '{self.name}' was blocked by a security policy.",
                    tool_name=self.name,
                    reason=detail,
                )

            except httpx.TimeoutException:
                return f"Result: TIMEOUT_ERROR: Request timed out after {self.timeout_seconds}s"

            except Exception as e:
                logger.error(f"System error for tool '{self.name}': {e}", exc_info=True)
                return f"Result: SYSTEM_ERROR: {str(e)}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        if self._session_id is None:
            raise ValueError("Session ID must be set before running the tool.")

        with tracer.start_as_current_span(f"tool.{self.name}.async") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("session.id", self._session_id)

            try:
                args_dict = self._parse_input(*args, **kwargs)
            except ValueError as e:
                return f"Input Parsing Error: {e}"

            agent_url = os.environ.get("AGENT_URL", "http://localhost:8001")
            callback_url = f"{agent_url}/tool-result"
            
            payload = {
                "session_id": self._session_id,
                "tool_name": self.name,
                "args": args_dict,
                "agent_callback_url": callback_url,
            }
            headers = {
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            }

            logger.info("(Async) Tool invocation: name=%s, session=%s", self.name, self._session_id)

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self._interceptor_url}/v1/proxy-execute",
                        json=payload,
                        headers=headers,
                        timeout=5.0,  # Short timeout for request acceptance
                    )
                    response.raise_for_status()
                
                # Wait for result from MCP (sent directly to Agent)
                result_key = f"{self._session_id}:{self.name}"
                max_wait = self.timeout_seconds
                start_time = time.time()
                
                logger.info("(Async) Waiting for tool result: key=%s, timeout=%s", result_key, max_wait)
                
                while time.time() - start_time < max_wait:
                    if _pending_results and result_key in _pending_results:
                        result_data = _pending_results.pop(result_key)
                        logger.info("(Async) Received tool result: key=%s, success=%s", result_key, result_data["success"])
                        if result_data["success"]:
                            return str(result_data["result"])
                        else:
                            return f"Error: {result_data.get('error', 'Unknown error')}"
                    await asyncio.sleep(0.1)
                
                logger.warning("(Async) Timeout waiting for tool result: key=%s", result_key)
                return "Error: Timeout waiting for tool result from MCP"

            except httpx.HTTPStatusError as e:
                try:
                    detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    detail = e.response.text
                
                if "Unreachable" in detail or "upstream" in detail.lower():
                    logger.error(f"(Async) MCP server unreachable for tool '{self.name}': {detail}")
                    return f"Error: MCP server is not reachable. Please ensure the MCP server is running."

                logger.warning(f"(Async) Access Denied for tool '{self.name}': {detail}")
                raise SecurityBlockException(
                    message=f"Tool '{self.name}' was blocked by a security policy.",
                    tool_name=self.name,
                    reason=detail,
                )

            except httpx.TimeoutException:
                return f"Result: TIMEOUT_ERROR: Request timed out after {self.timeout_seconds}s"

            except Exception as e:
                logger.error(f"(Async) System error for tool '{self.name}': {e}", exc_info=True)
                return f"Result: SYSTEM_ERROR: {str(e)}"


def create_tool_args_schema(tool_name: str, parameters: list[dict[str, Any]]) -> type[BaseModel]:
    from pydantic import create_model
    
    fields: dict[str, Any] = {}
    
    for param in parameters:
        param_name = param["name"]
        param_type = _python_type_from_string(param.get("type", "string"))
        is_required = param.get("required", False)
        default = param.get("default", None)
        description = param.get("description", "")
        
        if is_required:
            fields[param_name] = (param_type, Field(description=description))
        else:
            fields[param_name] = (Optional[param_type], Field(default=default, description=description))
    
    model_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(model_name, **fields)


def _python_type_from_string(type_str: str) -> type:
    type_mapping = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return type_mapping.get(type_str.lower(), str)


def create_tools_from_config(tools_config: list[dict[str, Any]], session_id: str) -> list[SentinelSecureTool]:
    tools: list[SentinelSecureTool] = []
    
    for tool_def in tools_config:
        tool_name = tool_def["name"]
        description = tool_def.get("description", "")
        parameters = tool_def.get("parameters", [])
        
        args_schema = None
        if parameters:
            args_schema = create_tool_args_schema(tool_name, parameters)
        
        tool = SentinelSecureTool(
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )
        tool.set_session_id(session_id)
        tools.append(tool)
    
    logger.info("Created %d tools from configuration", len(tools))
    return tools


def update_tools_session(tools: list[SentinelSecureTool], new_session_id: str) -> None:
    for tool in tools:
        tool.set_session_id(new_session_id)

