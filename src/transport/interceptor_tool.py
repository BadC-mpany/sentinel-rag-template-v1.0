"""
Sentinel Secure Tool

Agent sends POST requests to Interceptor. Agent cannot communicate with MCP directly.
MCP receives requests from Interceptor and sends results back through Interceptor.
"""

import json
import logging
import os
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SecurityBlockException(ToolException):
    """
    Exception raised when a security policy blocks a tool execution.
    This is a permanent denial - the agent should not retry.
    """

    def __init__(self, message: str, tool_name: str, reason: str):
        # The 'message' is what the LLM will see.
        super().__init__(f"Tool '{tool_name}' was blocked by a security policy. Reason: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class SentinelSecureTool(BaseTool):
    """
    Tool wrapper that sends POST requests to Interceptor.
    
    Agent cannot communicate with MCP directly. All tool calls go through Interceptor.
    Interceptor forwards to MCP, MCP executes and returns results through Interceptor.
    
    Environment Variables:
        SENTINEL_API_KEY: API key for Interceptor
        SENTINEL_URL: Base URL of Interceptor service
    """

    def __init__(self, name: str, description: str, args_schema: Optional[type[BaseModel]] = None, timeout_seconds: float = 30.0, **kwargs: Any):
        super().__init__(name=name, description=description, args_schema=args_schema, **kwargs)
        self.timeout_seconds = timeout_seconds
        self._api_key = os.getenv("SENTINEL_API_KEY")
        self._interceptor_url = os.getenv("SENTINEL_URL", "http://localhost:8000")
        self._session_id: Optional[str] = None

        if not self._api_key:
            logger.warning("SENTINEL_API_KEY is not set - tool calls will fail")

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for this tool instance."""
        self._session_id = session_id

    def _parse_input(self, *args: Any, **kwargs: Any) -> dict:
        """
        Universally handles arguments, whether they are passed positionally
        (as a single dict or JSON string) or as keyword arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary of parsed arguments
        """
        if kwargs:
            return kwargs

        if args:
            tool_input = args[0]
            if isinstance(tool_input, dict):
                return tool_input
            try:
                # Agent is passing a JSON string
                return json.loads(tool_input)
            except (json.JSONDecodeError, TypeError):
                # Agent is passing a raw string, wrap it in the schema
                if self.args_schema:
                    model_fields = getattr(self.args_schema, 'model_fields', None) or getattr(self.args_schema, '__fields__', {})
                    if len(model_fields) == 1:
                        field_name = list(model_fields.keys())[0]
                        return {field_name: tool_input}
                raise ValueError(
                    f"Received a non-JSON string '{tool_input}' and could not map it to the tool schema."
                )

        return {}

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Synchronous execution - sends request to Interceptor.
        
        This method:
        1. Validates session ID is set
        2. Parses input arguments
        3. Sends POST request to Interceptor
        4. Handles responses (success, access denied, or error)
        
        Args:
            *args: Positional arguments
            **kwargs: Tool arguments from the LLM
            
        Returns:
            String result for the LLM to process
            
        Raises:
            SecurityBlockException: If access is denied by security policy
        """
        if self._session_id is None:
            raise ValueError("Session ID must be set before running the tool.")

        with tracer.start_as_current_span(f"tool.{self.name}") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("session.id", self._session_id)

            try:
                args_dict = self._parse_input(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Input parsing error for tool '{self.name}': {e}")
                return f"Input Parsing Error: {e}"

            span.set_attribute("tool.args_count", len(args_dict))

            payload = {
                "session_id": self._session_id,
                "tool_name": self.name,
                "args": args_dict,
            }
            headers = {
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            }

            logger.info(
                "Tool invocation: name=%s, session=%s",
                self.name,
                self._session_id,
            )

            try:
                with httpx.Client() as client:
                    response = client.post(
                        f"{self._interceptor_url}/v1/proxy-execute",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()

                span.set_attribute("http.status_code", response.status_code)
                logger.info(
                    "Tool execution successful: name=%s, session=%s",
                    self.name,
                    self._session_id,
                )
                return str(response.json())

            except httpx.HTTPStatusError as e:
                span.set_attribute("http.status_code", e.response.status_code)
                
                try:
                    detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    detail = e.response.text

                logger.warning(f"Access Denied for tool '{self.name}': {detail}")
                raise SecurityBlockException(
                    message=f"Tool '{self.name}' was blocked by a security policy.",
                    tool_name=self.name,
                    reason=detail,
                )

            except httpx.TimeoutException as e:
                logger.error(f"Timeout while running tool '{self.name}': {e}")
                return f"Result: TIMEOUT_ERROR: Request timed out after {self.timeout_seconds}s"

            except Exception as e:
                logger.error(
                    f"An unexpected system error occurred while running tool '{self.name}': {e}",
                    exc_info=True,
                )
                return f"Result: SYSTEM_ERROR: {str(e)}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """
        Asynchronous execution - sends request to Interceptor.
        
        This method:
        1. Validates session ID is set
        2. Parses input arguments
        3. Sends async POST request to Interceptor
        4. Handles responses (success, access denied, or error)
        
        Args:
            *args: Positional arguments
            **kwargs: Tool arguments from the LLM
            
        Returns:
            String result for the LLM to process
            
        Raises:
            SecurityBlockException: If access is denied by security policy
        """
        if self._session_id is None:
            raise ValueError("Session ID must be set before running the tool.")

        with tracer.start_as_current_span(f"tool.{self.name}.async") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("session.id", self._session_id)

            try:
                args_dict = self._parse_input(*args, **kwargs)
            except ValueError as e:
                logger.error(f"(Async) Input parsing error for tool '{self.name}': {e}")
                return f"Input Parsing Error: {e}"

            span.set_attribute("tool.args_count", len(args_dict))

            payload = {
                "session_id": self._session_id,
                "tool_name": self.name,
                "args": args_dict,
            }
            headers = {
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            }

            logger.info(
                "(Async) Tool invocation: name=%s, session=%s",
                self.name,
                self._session_id,
            )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self._interceptor_url}/v1/proxy-execute",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()

                span.set_attribute("http.status_code", response.status_code)
                logger.info(
                    "(Async) Tool execution successful: name=%s, session=%s",
                    self.name,
                    self._session_id,
                )
                return str(response.json())

            except httpx.HTTPStatusError as e:
                span.set_attribute("http.status_code", e.response.status_code)
                
                try:
                    detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    detail = e.response.text

                logger.warning(f"(Async) Access Denied for tool '{self.name}': {detail}")
                raise SecurityBlockException(
                    message=f"Tool '{self.name}' was blocked by a security policy.",
                    tool_name=self.name,
                    reason=detail,
                )

            except httpx.TimeoutException as e:
                logger.error(f"(Async) Timeout while running tool '{self.name}': {e}")
                return f"Result: TIMEOUT_ERROR: Request timed out after {self.timeout_seconds}s"

            except Exception as e:
                logger.error(
                    f"(Async) An unexpected system error occurred while running tool '{self.name}': {e}",
                    exc_info=True,
                )
                return f"Result: SYSTEM_ERROR: {str(e)}"


def create_tool_args_schema(
    tool_name: str,
    parameters: list[dict[str, Any]],
) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model for tool arguments.
    
    Args:
        tool_name: Name of the tool
        parameters: Parameter definitions from config
        
    Returns:
        Pydantic model class for the tool's arguments
    """
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
    
    # Create dynamic model with sanitized name
    model_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(model_name, **fields)


def _python_type_from_string(type_str: str) -> type:
    """Convert string type name to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return type_mapping.get(type_str.lower(), str)


def create_tools_from_config(
    tools_config: list[dict[str, Any]],
    session_id: str,
) -> list[SentinelSecureTool]:
    """
    Create SentinelSecureTool instances from configuration.
    
    This factory function reads tool definitions from the config
    and creates properly configured tool instances.
    
    Args:
        tools_config: List of tool definitions from sentinel_config.yaml
        session_id: Session identifier to inject into all requests
        
    Returns:
        List of configured SentinelSecureTool instances
    """
    tools: list[SentinelSecureTool] = []
    
    for tool_def in tools_config:
        tool_name = tool_def["name"]
        description = tool_def.get("description", "")
        parameters = tool_def.get("parameters", [])
        
        # Create args schema if parameters are defined
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
        logger.debug("Created tool: %s", tool_name)
    
    logger.info("Created %d tools from configuration", len(tools))
    return tools


def update_tools_session(
    tools: list[SentinelSecureTool],
    new_session_id: str,
) -> None:
    """
    Update the session ID for a list of tools.
    
    Used when creating a new session/conversation to ensure
    all tool calls are tagged with the correct session.
    
    Args:
        tools: List of existing tools
        new_session_id: New session identifier
    """
    for tool in tools:
        tool.set_session_id(new_session_id)
    
    logger.debug("Updated session ID for %d tools", len(tools))
