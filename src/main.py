"""
Sentinel RAG Agent

Agent sends POST requests to Interceptor. Agent cannot communicate with MCP directly.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.bootstrap import bootstrap, setup_logging
from src.agent.core import (
    SentinelAgent,
    AgentConfig,
    AgentResponse,
    ConversationMessage,
    create_agent,
)

logger = logging.getLogger(__name__)

# Global state
_agent: SentinelAgent | None = None
_config: dict | None = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    conversation_history: list[ConversationMessage] | None = Field(
        None,
        description="Previous conversation messages",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Agent response")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools that were invoked",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")


class ConfigResponse(BaseModel):
    """Response model for configuration endpoint."""
    
    service_name: str
    version: str
    tools: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks including:
    - Bootstrap configuration
    - Initialize agent
    """
    global _agent, _config
    
    # Setup logging
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
        
        logger.info(
            "Agent initialized: model=%s, tools=%d",
            agent_config.model_name,
            len(tools_config),
        )
        
        yield
        
    except Exception as e:
        logger.error("Startup failed: %s", str(e))
        raise
    finally:
        logger.info("Sentinel RAG Agent shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Sentinel RAG Agent",
    description=(
        "Secure RAG client implementing the Blind Courier pattern "
        "in a Zero Trust architecture."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_session(
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
) -> str | None:
    """
    Dependency to extract and validate session ID from headers.
    
    If no session ID is provided, one will be generated.
    """
    return x_session_id


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _agent else "degraded",
        service_name=_config.get("service_name", "sentinel-rag-agent") if _config else "sentinel-rag-agent",
        version=_config.get("version", "unknown") if _config else "unknown",
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get current configuration."""
    if not _config:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return ConfigResponse(
        service_name=_config.get("service_name", "sentinel-rag-agent"),
        version=_config.get("version", "1.0"),
        tools=[tool.get("name") for tool in _config.get("tools", [])],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session_id: str | None = Depends(verify_session),
) -> ChatResponse:
    """
    Main chat endpoint.
    
    Process a user message through the RAG agent.
    Agent sends POST requests to Interceptor. Agent cannot communicate with MCP directly.
    
    Args:
        request: Chat request with message and optional history
        session_id: Optional session ID from header
        
    Returns:
        Agent response with tool call information
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Use header session ID if provided, otherwise use request session ID
    effective_session_id = session_id or request.session_id or str(uuid.uuid4())
    
    logger.info(
        "Chat request: session=%s, message_length=%d",
        effective_session_id,
        len(request.message),
    )
    
    try:
        response = await _agent.invoke(
            message=request.message,
            session_id=effective_session_id,
            conversation_history=request.conversation_history,
        )
        
        return ChatResponse(
            session_id=response.session_id,
            response=response.message,
            tool_calls=response.tool_calls,
        )
    
    except Exception as e:
        logger.error(
            "Chat error: session=%s, error=%s",
            effective_session_id,
            str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request",
        )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    session_id: str | None = Depends(verify_session),
):
    """
    Streaming chat endpoint.
    
    Stream responses from the RAG agent.
    
    Note: This is a placeholder - full SSE implementation
    would require additional setup.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    effective_session_id = session_id or request.session_id or str(uuid.uuid4())
    
    # For now, return non-streaming response
    # Full SSE implementation would use StreamingResponse
    response = await _agent.invoke(
        message=request.message,
        session_id=effective_session_id,
        conversation_history=request.conversation_history,
    )
    
    return ChatResponse(
        session_id=response.session_id,
        response=response.message,
        tool_calls=response.tool_calls,
    )


# CLI functionality
def run_cli():
    """
    Run the agent in CLI mode for testing.
    """
    import asyncio
    
    async def cli_main():
        global _agent, _config
        
        setup_logging("INFO")
        
        print("=" * 60)
        print("Sentinel RAG Agent - CLI Mode")
        print("=" * 60)
        print("\nInitializing...")
        
        try:
            _config = bootstrap()
            
            # Create agent
            agent_config = AgentConfig(
                model_name=os.environ.get("LLM_MODEL", "gpt-4-turbo-preview"),
                temperature=0.1,
            )
            
            tools_config = _config.get("tools", [])
            
            _agent = create_agent(
                tools_config=tools_config,
                agent_config=agent_config,
            )
            
            print(f"\nAgent ready! (Model: {agent_config.model_name})")
            print("Type 'quit' or 'exit' to stop.\n")
            
            session_id = str(uuid.uuid4())
            history: list[ConversationMessage] = []
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ["quit", "exit"]:
                        print("\nGoodbye!")
                        break
                    
                    if user_input.lower() == "clear":
                        history.clear()
                        session_id = str(uuid.uuid4())
                        print("Conversation cleared.\n")
                        continue
                    
                    # Add user message to history
                    history.append(ConversationMessage(role="user", content=user_input))
                    
                    # Get response
                    response = await _agent.invoke(
                        message=user_input,
                        session_id=session_id,
                        conversation_history=history[:-1],  # Exclude current message
                    )
                    
                    # Add assistant response to history
                    history.append(
                        ConversationMessage(role="assistant", content=response.message)
                    )
                    
                    print(f"\nAssistant: {response.message}")
                    
                    if response.tool_calls:
                        print(f"\n[Tools used: {', '.join(tc['name'] for tc in response.tool_calls)}]")
                    
                    print()
                
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {str(e)}\n")
        
        except Exception as e:
            logger.error("CLI initialization failed: %s", str(e))
            print(f"\nFailed to initialize: {str(e)}")
    
    asyncio.run(cli_main())


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        # Run as API server
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8001"))
        debug = os.environ.get("DEBUG", "false").lower() == "true"
        
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info",
        )


if __name__ == "__main__":
    main()
