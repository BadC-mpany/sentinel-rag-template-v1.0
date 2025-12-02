"""
Transport Layer
===============

Handles communication with the Interceptor backend.
Agent sends POST requests to Interceptor. Agent cannot communicate with MCP directly.
"""

from .interceptor_tool import (
    SentinelSecureTool,
    SecurityBlockException,
    create_tools_from_config,
    update_tools_session,
)

__all__ = [
    "SentinelSecureTool",
    "SecurityBlockException",
    "create_tools_from_config",
    "update_tools_session",
]
