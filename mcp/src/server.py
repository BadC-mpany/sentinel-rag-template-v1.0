"""
MCP Server - Secure Execution Environment

This server receives requests from the Interceptor (not directly from the Agent).
It executes tools and sends results directly to the Agent.

The Agent cannot communicate with this server directly.
"""

import logging
import os
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()

logger = logging.getLogger(__name__)

MCP_ROOT = Path(__file__).parent.parent
DATA_ROOT = MCP_ROOT / "data"

# Agent URL for sending results directly
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8001")


class ToolRequest(BaseModel):
    """Request from Interceptor to execute a tool."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    session_id: str = Field(..., description="Session identifier")
    agent_callback_url: str | None = Field(None, description="URL to send result to Agent")


class ToolResponse(BaseModel):
    """Response from tool execution (acknowledgment only)."""
    success: bool = Field(..., description="Whether execution was accepted")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service_name: str
    data_root: str


def verify_interceptor_token(x_sentinel_token: str | None = Header(None, alias="X-Sentinel-Token")) -> bool:
    """
    Verify the request came from the Interceptor (legacy endpoint).
    """
    expected_token = os.environ.get("MCP_INTERCEPTOR_TOKEN", "mcp-secret-token")
    if x_sentinel_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing Interceptor token")
    return True


def verify_jwt_token(authorization: str | None = Header(None)) -> bool:
    """
    Verify JWT token from Interceptor (JSON-RPC endpoint).
    In production, this would verify the Ed25519 signature.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.replace("Bearer ", "")
    # For now, accept any token (in production, verify Ed25519 signature)
    # TODO: Verify JWT signature using MCP public key
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting MCP Server...")
    logger.info("Data root: %s", DATA_ROOT)
    
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "public").mkdir(exist_ok=True)
    (DATA_ROOT / "confidential").mkdir(exist_ok=True)
    
    yield
    
    logger.info("MCP Server shutdown complete")


app = FastAPI(
    title="Sentinel MCP Server",
    description="Secure Execution Environment - receives requests from Interceptor only",
    version="1.0.0",
    lifespan=lifespan,
)


def setup_logging():
    """Setup logging for MCP server."""
    # Create logs directory in project root
    # Path: mcp/src/server.py -> src -> mcp -> project_root
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "mcp.log"
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_format = os.environ.get("LOG_FORMAT", "text")
    
    # File handler for mcp.log
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler (still output to console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if log_format == "json":
        import json
        
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data)
        
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger.info("MCP logging configured: file=%s, level=%s", log_file, log_level)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service_name="sentinel-mcp-server",
        data_root=str(DATA_ROOT),
    )


@app.post("/")
async def execute_tool_jsonrpc(
    request: dict,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    """
    Handle JSON-RPC 2.0 requests from Interceptor.
    """
    if request.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid Request"},
            "id": request.get("id")
        }
    
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    # Verify JWT token
    try:
        verify_jwt_token(authorization)
    except HTTPException:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32001, "message": "Unauthorized"},
            "id": request_id
        }
    
    if method != "tools/call":
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": request_id
        }
    
    tool_name = params.get("name")
    tool_args = params.get("arguments", {})
    session_id = params.get("session_id", "unknown")
    agent_callback_url = params.get("agent_callback_url")
    
    logger.info("JSON-RPC tool execution: tool=%s, session=%s", tool_name, session_id)
    
    try:
        result = await _execute_tool(tool_name, tool_args)
        
        # Send result directly to Agent
        callback_url = agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=session_id,
            tool_name=tool_name,
            result=result,
            success=True,
        )
        
        return {
            "jsonrpc": "2.0",
            "result": {"success": True, "message": "Tool executed and result sent to Agent"},
            "id": request_id
        }
        
    except FileNotFoundError as e:
        callback_url = agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=session_id,
            tool_name=tool_name,
            result=None,
            success=False,
            error=f"File not found: {str(e)}",
        )
        return {
            "jsonrpc": "2.0",
            "result": {"success": True, "message": "Error sent to Agent"},
            "id": request_id
        }
        
    except PermissionError as e:
        callback_url = agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=session_id,
            tool_name=tool_name,
            result=None,
            success=False,
            error=f"Permission denied: {str(e)}",
        )
        return {
            "jsonrpc": "2.0",
            "result": {"success": True, "message": "Error sent to Agent"},
            "id": request_id
        }
        
    except Exception as e:
        logger.error("Tool execution error: %s", str(e), exc_info=True)
        callback_url = agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=session_id,
            tool_name=tool_name,
            result=None,
            success=False,
            error=str(e),
        )
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)},
            "id": request_id
        }


@app.post("/execute", response_model=ToolResponse)
async def execute_tool(
    request: ToolRequest,
    authorized: bool = Depends(verify_interceptor_token),
) -> ToolResponse:
    """
    Execute a tool. Only accepts requests from Interceptor.
    Executes the tool and sends result directly to Agent.
    """
    logger.info("Tool execution request: tool=%s, session=%s", request.tool_name, request.session_id)
    
    # Execute tool asynchronously and send result to Agent
    try:
        result = await _execute_tool(request.tool_name, request.args)
        
        # Send result directly to Agent
        callback_url = request.agent_callback_url
        if not callback_url:
            callback_url = f"{AGENT_URL}/tool-result"
            logger.warning("No callback URL provided, using default: %s", callback_url)
        
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=request.session_id,
            tool_name=request.tool_name,
            result=result,
            success=True,
        )
        
        return ToolResponse(success=True, message="Tool executed and result sent to Agent")
        
    except FileNotFoundError as e:
        logger.warning("File not found: %s", str(e))
        callback_url = request.agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=request.session_id,
            tool_name=request.tool_name,
            result=None,
            success=False,
            error=f"File not found: {str(e)}",
        )
        return ToolResponse(success=True, message="Error sent to Agent")
        
    except PermissionError as e:
        logger.warning("Permission denied: %s", str(e))
        callback_url = request.agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=request.session_id,
            tool_name=request.tool_name,
            result=None,
            success=False,
            error=f"Permission denied: {str(e)}",
        )
        return ToolResponse(success=True, message="Error sent to Agent")
        
    except Exception as e:
        logger.error("Tool execution error: %s", str(e), exc_info=True)
        callback_url = request.agent_callback_url or f"{AGENT_URL}/tool-result"
        await _send_result_to_agent(
            agent_url=callback_url,
            session_id=request.session_id,
            tool_name=request.tool_name,
            result=None,
            success=False,
            error=str(e),
        )
        return ToolResponse(success=True, message="Error sent to Agent")


async def _send_result_to_agent(
    agent_url: str,
    session_id: str,
    tool_name: str,
    result: Any,
    success: bool,
    error: str | None = None,
) -> None:
    """Send tool execution result directly to Agent."""
    payload = {
        "session_id": session_id,
        "tool_name": tool_name,
        "success": success,
        "result": result,
        "error": error,
    }
    
    result_key = f"{session_id}:{tool_name}"
    logger.info("=" * 60)
    logger.info("Sending result to Agent:")
    logger.info("  URL: %s", agent_url)
    logger.info("  Tool: %s", tool_name)
    logger.info("  Session: %s", session_id)
    logger.info("  Result Key: %s", result_key)
    logger.info("  Success: %s", success)
    logger.info("  Result: %s", str(result)[:200] if result else None)
    logger.info("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(agent_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.info("Result sent successfully to Agent: status=%d, response=%s", 
                       response.status_code, response_data)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error sending result to Agent: status=%d, response=%s", 
                     e.response.status_code, e.response.text)
    except Exception as e:
        logger.error("Failed to send result to Agent: %s", str(e), exc_info=True)


async def _execute_tool(tool_name: str, args: dict[str, Any]) -> Any:
    """Execute the requested tool."""
    
    if tool_name == "read_file":
        return await _read_file(args)
    elif tool_name == "web_search":
        return await _web_search(args)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


async def _read_file(args: dict[str, Any]) -> dict:
    """Reads content from a file at the specified path."""
    file_path = args.get("path", "")
    
    if not file_path:
        raise ValueError("path is required")
    
    full_path = DATA_ROOT / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not str(full_path.resolve()).startswith(str(DATA_ROOT.resolve())):
        raise PermissionError("Access denied: path traversal detected")
    
    content = full_path.read_text(encoding="utf-8")
    
    return {
        "path": file_path,
        "content": content,
        "size": len(content),
    }


async def _web_search(args: dict[str, Any]) -> dict:
    """Searches the web for information using a search query."""
    query = args.get("query", "")
    
    if not query:
        raise ValueError("query is required")
    
    # Placeholder: In production, integrate with actual search API
    return {
        "query": query,
        "results": [
            {
                "title": f"Search result for: {query}",
                "snippet": f"This is a placeholder result for the query '{query}'. In production, this would return actual web search results.",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            }
        ],
        "total": 1,
    }


def main():
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "9000"))
    
    uvicorn.run(
        "mcp.src.server:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

