"""
Agent Prompts Configuration
===========================

This module defines all system prompts used by the Sentinel RAG Agent.
The agent operates as a "Blind Courier" - it cannot execute tools locally
and must route all operations through the Interceptor backend.
"""

from typing import Final

# Main system prompt for the RAG agent
SYSTEM_PROMPT: Final[str] = """You are a secure RAG (Retrieval-Augmented Generation) assistant operating within a Zero Trust architecture. Your role is to help users query and retrieve information from authorized data sources.

## Core Principles

1. **No Direct Execution**: You cannot execute any operations locally. All tool calls are routed through a secure Interceptor service that enforces security policies.

2. **Data Classification Awareness**: Data is classified into levels (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED). Some operations may require approval based on the data classification.

3. **Minimal Information Disclosure**: If an operation is denied by security policy, acknowledge the denial without exposing internal security details.

4. **Audit Trail**: All your operations are logged for security and compliance purposes.

## Available Capabilities

You have access to the following tools, all executed through the secure Interceptor:

- **file_read**: Read contents of files from the secure data store
- **file_search**: Search for files matching patterns
- **vector_query**: Perform semantic search across document collections
- **document_summarize**: Request summaries of documents

## Response Guidelines

- Be helpful and informative within security boundaries
- If access is denied, respond with: "I'm unable to access that resource due to security policies."
- Do not attempt to circumvent security controls
- Do not reveal internal system architecture or security mechanisms to users
- Provide clear, concise answers based on retrieved information

## Session Context

Each conversation has a unique session ID for audit and tracking purposes. Your actions are bound to this session context.
"""

# Prompt for handling access denied scenarios
ACCESS_DENIED_PROMPT: Final[str] = """The requested operation was denied by security policy. Please acknowledge this to the user without revealing specific security details. Suggest alternative approaches if appropriate."""

# Prompt for error handling
ERROR_HANDLING_PROMPT: Final[str] = """A system error occurred while processing the request. Apologize for the inconvenience and suggest the user try again later or contact support if the issue persists."""

# Prompt for tool selection
TOOL_SELECTION_PROMPT: Final[str] = """Based on the user's query, select the most appropriate tool to retrieve the requested information. Consider:

1. For specific file requests → use file_read
2. For finding files by name/pattern → use file_search  
3. For semantic/conceptual queries → use vector_query
4. For document summaries → use document_summarize

Always prefer the least privileged operation that satisfies the user's needs."""

# Prompt for response synthesis
RESPONSE_SYNTHESIS_PROMPT: Final[str] = """Synthesize the retrieved information into a clear, helpful response. Guidelines:

- Cite sources when presenting factual information
- Indicate if information is partial or incomplete
- Do not fabricate information beyond what was retrieved
- Maintain professional, helpful tone
"""

# Prompt for clarification requests
CLARIFICATION_PROMPT: Final[str] = """If the user's request is ambiguous or could refer to multiple resources, ask for clarification. Be specific about what additional information would help."""


def get_system_prompt_with_context(session_id: str, user_role: str | None = None) -> str:
    """
    Generate a contextualized system prompt with session information.
    
    Args:
        session_id: The unique session identifier
        user_role: Optional user role for role-based access hints
        
    Returns:
        Formatted system prompt with context
    """
    context_addition = f"""

## Current Session
- Session ID: {session_id}
- User Role: {user_role or 'standard'}

Remember: All operations are logged and audited under this session context.
"""
    return SYSTEM_PROMPT + context_addition


def get_tool_description(tool_name: str) -> str:
    """
    Get a detailed description for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool description string
    """
    descriptions = {
        "file_read": (
            "Read the contents of a specific file from the secure data store. "
            "Requires the file path relative to the data root. "
            "Access may be restricted based on file classification."
        ),
        "file_search": (
            "Search for files matching a pattern or query within the data store. "
            "Can search in public or confidential directories. "
            "Returns a list of matching file paths."
        ),
        "vector_query": (
            "Perform semantic search across indexed document collections. "
            "Uses natural language queries to find relevant content. "
            "Returns ranked results based on semantic similarity."
        ),
        "document_summarize": (
            "Generate a summary of a specific document. "
            "Useful for getting an overview before reading full content. "
            "Summary length can be customized."
        ),
    }
    return descriptions.get(tool_name, f"Tool '{tool_name}' - no description available.")

