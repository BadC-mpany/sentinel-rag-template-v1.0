"""Sentinel RAG Agent - FastAPI entry point."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()

from agent.src.bootstrap import bootstrap, setup_logging
from agent.src.core import (
    SentinelAgent,
    AgentConfig,
    AgentResponse,
    ConversationMessage,
    create_agent,
)
from agent.src.transport.interceptor_tool import set_pending_results_store

logger = logging.getLogger(__name__)

_agent: SentinelAgent | None = None
_config: dict | None = None

# Store pending tool results by session_id + tool_name
_pending_tool_results: dict[str, dict[str, Any]] = {}


class ToolResultRequest(BaseModel):
    """Request from MCP with tool execution result."""
    session_id: str = Field(..., description="Session identifier")
    tool_name: str = Field(..., description="Tool name")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(None, description="Tool result")
    error: str | None = Field(None, description="Error message if failed")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    conversation_history: list[ConversationMessage] | None = Field(None, description="Previous conversation messages")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Agent response")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tools that were invoked")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _config
    
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("Starting Sentinel RAG Agent...")
    
    try:
        _config = bootstrap()
        
        agent_config = AgentConfig(
            model_name=os.environ.get("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "10")),
            verbose=os.environ.get("AGENT_VERBOSE", "false").lower() == "true",
        )
        
        tools_config = _config.get("tools", [])
        
        _agent = create_agent(
            tools_config=tools_config,
            agent_config=agent_config,
        )
        
        # Share pending results store with tools
        set_pending_results_store(_pending_tool_results)
        
        logger.info("Agent initialized: model=%s, tools=%d", agent_config.model_name, len(tools_config))
        
        yield
        
    except Exception as e:
        logger.error("Startup failed: %s", str(e))
        raise
    finally:
        logger.info("Sentinel RAG Agent shutdown complete")


app = FastAPI(
    title="Sentinel RAG Agent",
    description="Secure RAG client implementing the Blind Courier pattern in a Zero Trust architecture.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_session(x_session_id: str | None = Header(None, alias="X-Session-ID")) -> str | None:
    return x_session_id


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _agent else "degraded",
        service_name=_config.get("service_name", "sentinel-rag-agent") if _config else "sentinel-rag-agent",
        version=_config.get("version", "unknown") if _config else "unknown",
    )


@app.post("/tool-result")
async def receive_tool_result(request: ToolResultRequest) -> dict:
    """
    Receive tool execution result from MCP.
    MCP sends results directly to this endpoint.
    """
    result_key = f"{request.session_id}:{request.tool_name}"
    _pending_tool_results[result_key] = {
        "success": request.success,
        "result": request.result,
        "error": request.error,
    }
    logger.info("Received tool result: session=%s, tool=%s, success=%s, result=%s", 
                request.session_id, request.tool_name, request.success, 
                str(request.result)[:100] if request.result else None)
    return {"status": "received", "result_key": result_key}


@app.get("/debug/tools")
async def debug_tools() -> dict:
    """Debug endpoint to check tools configuration."""
    if not _agent:
        return {"error": "Agent not initialized"}
    
    tools_info = []
    if hasattr(_agent, '_tools'):
        for tool in _agent._tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "has_session": hasattr(tool, '_session_id') and tool._session_id is not None,
            })
    
    return {
        "agent_initialized": _agent is not None,
        "tools_count": len(tools_info),
        "tools": tools_info,
        "config": {
            "model": _agent.config.model_name if _agent else None,
            "has_api_key": bool(os.getenv("OPENROUTER_API_KEY")),
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str | None = Depends(verify_session)) -> ChatResponse:
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    effective_session_id = session_id or request.session_id or str(uuid.uuid4())
    
    logger.info("=" * 60)
    logger.info("Chat request: session=%s, message_length=%d", effective_session_id, len(request.message))
    logger.info("Message: %s", request.message)
    logger.info("=" * 60)
    
    try:
        response = await _agent.invoke(
            message=request.message,
            session_id=effective_session_id,
            conversation_history=request.conversation_history,
        )
        
        logger.info("=" * 60)
        logger.info("Chat response: session=%s, message_length=%d, tool_calls=%d", 
                   effective_session_id, len(response.message), len(response.tool_calls))
        logger.info("Response message: %s", response.message[:200])
        logger.info("Tool calls: %s", response.tool_calls)
        logger.info("=" * 60)
        
        return ChatResponse(
            session_id=response.session_id,
            response=response.message,
            tool_calls=response.tool_calls,
        )
    
    except Exception as e:
        import traceback
        logger.error("=" * 60)
        logger.error("Chat error: session=%s, error=%s", effective_session_id, str(e))
        logger.error("Traceback:\n%s", traceback.format_exc())
        logger.error("=" * 60)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def main():
    import sys
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", "8001"))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "agent.src.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()

