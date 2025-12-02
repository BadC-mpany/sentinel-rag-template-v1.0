"""
Agent Core

LangChain agent that sends POST requests to Interceptor.
Agent cannot communicate with MCP directly.
"""

import logging
import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from opentelemetry import trace
from pydantic import BaseModel, Field

from config.agent_prompts import get_system_prompt_with_context
from src.transport.interceptor_tool import (
    SentinelSecureTool,
    create_tools_from_config,
    update_tools_session,
)
from .callbacks import SentinelCallbackHandler, TelemetryCallbackHandler

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AgentConfig(BaseModel):
    """Configuration for the Sentinel Agent."""
    
    model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="LLM model to use (OpenRouter compatible)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in response",
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum agent iterations",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )
    # OpenRouter specific
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (falls back to env var)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class AgentResponse(BaseModel):
    """Response from the agent."""
    
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="Agent response message")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools that were invoked",
    )


class SentinelAgent:
    """
    RAG Agent that sends POST requests to Interceptor.
    
    Agent cannot communicate with MCP directly. All tool calls go through Interceptor.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tools_config: list[dict[str, Any]],
    ) -> None:
        """
        Initialize the Sentinel Agent.
        
        Args:
            config: Agent configuration
            tools_config: Tool definitions from config
        """
        self.config = config
        self.tools_config = tools_config
        
        # Get API key from config or environment
        api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required but not set")
        
        # Initialize LLM with OpenRouter
        self._llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key,
            openai_api_base=config.openrouter_base_url,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/sentinel-rag"),
                "X-Title": "Sentinel RAG Agent",
            },
        )
        
        self._agent: Any = None
        self._current_session_id: str | None = None
        self._tools: list[SentinelSecureTool] = []
    
    def _create_agent_for_session(self, session_id: str) -> Any:
        """
        Create or update agent for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Configured LangGraph agent
        """
        # Create tools with session context
        self._tools = create_tools_from_config(
            self.tools_config,
            session_id,
        )
        
        logger.info("Created %d tools for session %s: %s", 
                   len(self._tools), 
                   session_id,
                   [t.name for t in self._tools])
        
        # Create react agent using LangGraph
        agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
        )
        
        logger.debug("Agent created successfully for session %s", session_id)
        return agent
    
    async def invoke(
        self,
        message: str,
        session_id: str | None = None,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> AgentResponse:
        """
        Process a user message through the agent.
        
        Args:
            message: User's input message
            session_id: Optional session ID (generated if not provided)
            conversation_history: Optional conversation history
            
        Returns:
            AgentResponse with the agent's reply
        """
        # Generate or use provided session ID
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with tracer.start_as_current_span("agent.invoke") as span:
            span.set_attribute("session.id", session_id)
            
            # Create callbacks for this session
            callbacks = [
                SentinelCallbackHandler(session_id),
                TelemetryCallbackHandler(session_id),
            ]
            
            # Update agent for this session if needed
            if self._current_session_id != session_id or self._agent is None:
                self._agent = self._create_agent_for_session(session_id)
                self._current_session_id = session_id
            else:
                # Update existing tools with new session ID
                update_tools_session(self._tools, session_id)
            
            # Build messages list
            messages: list[Any] = []
            
            # Add system prompt first
            system_prompt = get_system_prompt_with_context(session_id)
            messages.append(SystemMessage(content=system_prompt))
            
            if conversation_history:
                for hist_msg in conversation_history:
                    if hist_msg.role == "user":
                        messages.append(HumanMessage(content=hist_msg.content))
                    elif hist_msg.role == "assistant":
                        messages.append(AIMessage(content=hist_msg.content))
                    elif hist_msg.role == "system":
                        messages.append(SystemMessage(content=hist_msg.content))
            
            messages.append(HumanMessage(content=message))
            
            logger.info(
                "Agent invocation: session=%s, message_length=%d",
                session_id,
                len(message),
            )
            
            try:
                # Invoke the agent
                logger.debug("Invoking agent with %d messages", len(messages))
                
                # LangGraph agents return state dict with messages
                result = await self._agent.ainvoke(
                    {"messages": messages},
                    config={"callbacks": callbacks},
                )
                
                logger.debug("Agent result type: %s", type(result))
                logger.debug("Agent result keys: %s", result.keys() if isinstance(result, dict) else "N/A")
                
                # Extract response - result is a state dict
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict result, got {type(result)}")
                
                response_messages = result.get("messages", [])
                logger.debug("Response messages count: %d", len(response_messages))
                
                final_message = ""
                tool_calls: list[dict[str, Any]] = []
                
                if not response_messages:
                    logger.warning("No messages in agent result, result: %s", result)
                    final_message = "No response generated"
                else:
                    # Log all message types for debugging
                    for i, msg in enumerate(response_messages):
                        logger.debug("Message %d: type=%s, content=%s", i, type(msg).__name__, str(msg)[:200])
                    
                    # Collect all AIMessages and their tool calls
                    ai_messages = [msg for msg in response_messages if isinstance(msg, AIMessage)]
                    tool_messages = [msg for msg in response_messages if isinstance(msg, ToolMessage)]
                    
                    logger.debug("Found %d AIMessages, %d ToolMessages", len(ai_messages), len(tool_messages))
                    
                    # Get the last AIMessage (final response)
                    last_ai_message = None
                    if ai_messages:
                        last_ai_message = ai_messages[-1]
                    
                    if last_ai_message:
                        # Extract content
                        if last_ai_message.content:
                            final_message = str(last_ai_message.content)
                        
                        # Extract tool calls from all AIMessages (not just the last one)
                        for ai_msg in ai_messages:
                            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                                logger.debug("Found %d tool calls in AIMessage", len(ai_msg.tool_calls))
                                for tc in ai_msg.tool_calls:
                                    if isinstance(tc, dict):
                                        tool_calls.append({
                                            "name": tc.get("name", "unknown"),
                                            "args": tc.get("args", {}),
                                        })
                                    else:
                                        tool_calls.append({
                                            "name": getattr(tc, "name", "unknown"),
                                            "args": getattr(tc, "args", {}),
                                        })
                        
                        # If we have tool calls but no final message, the agent is still processing
                        if tool_calls and not final_message:
                            final_message = "Processing tool calls..."
                    else:
                        logger.warning("No AIMessage found in response")
                        # Fallback: try to get any text content
                        for msg in reversed(response_messages):
                            if hasattr(msg, "content") and msg.content:
                                final_message = str(msg.content)
                                break
                
                logger.info(
                    "Agent completed: session=%s, tool_calls=%d",
                    session_id,
                    len(tool_calls),
                )
                
                return AgentResponse(
                    session_id=session_id,
                    message=final_message,
                    tool_calls=tool_calls,
                )
            
            except Exception as e:
                logger.error(
                    "Agent error: session=%s, error=%s",
                    session_id,
                    str(e),
                    exc_info=True,
                )
                span.record_exception(e)
                raise
    
    async def stream(
        self,
        message: str,
        session_id: str | None = None,
        conversation_history: list[ConversationMessage] | None = None,
    ):
        """
        Stream a response from the agent.
        
        Args:
            message: User's input message
            session_id: Optional session ID
            conversation_history: Optional conversation history
            
        Yields:
            Chunks of the response as they're generated
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Update agent for this session if needed
        if self._current_session_id != session_id or self._agent is None:
            self._agent = self._create_agent_for_session(session_id)
            self._current_session_id = session_id
        else:
            update_tools_session(self._tools, session_id)
        
        callbacks = [
            SentinelCallbackHandler(session_id),
            TelemetryCallbackHandler(session_id),
        ]
        
        messages: list[Any] = []
        
        # Add system prompt first
        system_prompt = get_system_prompt_with_context(session_id)
        messages.append(SystemMessage(content=system_prompt))
        
        if conversation_history:
            for hist_msg in conversation_history:
                if hist_msg.role == "user":
                    messages.append(HumanMessage(content=hist_msg.content))
                elif hist_msg.role == "assistant":
                    messages.append(AIMessage(content=hist_msg.content))
        
        messages.append(HumanMessage(content=message))
        
        async for chunk in self._agent.astream(
            {"messages": messages},
            config={"callbacks": callbacks},
        ):
            yield chunk


def create_agent(
    tools_config: list[dict[str, Any]],
    agent_config: AgentConfig | None = None,
) -> SentinelAgent:
    """
    Factory function to create a configured SentinelAgent.
    
    Args:
        tools_config: Tool definitions from config
        agent_config: Optional agent configuration
        
    Returns:
        Configured SentinelAgent instance
    """
    if agent_config is None:
        agent_config = AgentConfig()
    
    # Create the agent
    agent = SentinelAgent(
        config=agent_config,
        tools_config=tools_config,
    )
    
    logger.info(
        "Agent created: model=%s, tools=%d",
        agent_config.model_name,
        len(tools_config),
    )
    
    return agent
