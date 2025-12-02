# Sentinel RAG Template - Architecture Flow

This document explains the technical architecture and message lifecycle of the Sentinel RAG system, implementing the "Blind Courier" pattern in a Zero Trust architecture.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ZONE A: Client Zone                            │
│  ┌──────────┐                                                               │
│  │   User   │                                                               │
│  └────┬─────┘                                                               │
│       │ HTTP Request                                                        │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Sentinel RAG Agent (This Service)                  │  │
│  │                                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────┐ │  │
│  │  │  FastAPI    │───▶│  LangChain  │───▶│   SentinelSecureTool      │ │  │
│  │  │  Endpoint   │    │   Agent     │    │   (Blind Courier)         │ │  │
│  │  └─────────────┘    └─────────────┘    └───────────┬───────────────┘ │  │
│  │                                                     │                 │  │
│  └─────────────────────────────────────────────────────┼─────────────────┘  │
│                                                        │ POST /v1/proxy-execute
└────────────────────────────────────────────────────────┼────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ZONE B: Security Gateway                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Interceptor Service                           │  │
│  │                                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────┐ │  │
│  │  │   Auth &    │───▶│   Taint     │───▶│   Policy                  │ │  │
│  │  │   Session   │    │   Checker   │    │   Enforcement             │ │  │
│  │  └─────────────┘    └─────────────┘    └───────────┬───────────────┘ │  │
│  │                                                     │                 │  │
│  │                         ◄─── 403 if denied ───────┘                  │  │
│  │                                                     │ if allowed      │  │
│  └─────────────────────────────────────────────────────┼─────────────────┘  │
│                                                        │                    │
└────────────────────────────────────────────────────────┼────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ZONE C: Execution Zone                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     MCP Server / Tool Execution                       │  │
│  │                                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────┐ │  │
│  │  │   File      │    │   Vector    │    │   Web Search              │ │  │
│  │  │   System    │    │   Database  │    │   API                     │ │  │
│  │  └─────────────┘    └─────────────┘    └───────────────────────────┘ │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User Flow Example

**User Prompt:** "I want to know what is a panda"

### Step-by-Step Flow:

```
1. User → POST /chat {"message": "I want to know what is a panda"}

2. Agent receives request, creates session_id

3. LLM analyzes prompt, decides: "I need web_search tool"

4. SentinelSecureTool creates request:
   {
     "session_id": "abc-123",
     "tool_name": "web_search",
     "args": {
       "query": "what is a panda",
       "context": "user wants to learn about pandas"
     }
   }

5. POST to Interceptor: /v1/proxy-execute
   Headers: X-API-Key: <SENTINEL_API_KEY>
   Body: <above payload>

6. Interceptor:
   - Validates API key
   - Checks taint rules (web_search = LOW, allowed)
   - Executes tool
   - Returns result

7. Agent receives result, synthesizes response

8. User receives: "A panda is a bear species..."
```

---

## The "Blind Courier" Pattern

The Sentinel RAG Agent operates as a **Blind Courier** - it can request operations but cannot execute them directly. This provides several security benefits:

1. **No Local Execution**: The agent cannot access files, databases, or external services directly
2. **Policy Enforcement**: All operations must pass through the Interceptor's security checks
3. **Audit Trail**: Every operation is logged and traceable
4. **Taint Tracking**: Data classification is enforced at the execution layer

### Key Implementation: `SentinelSecureTool`

```python
class SentinelSecureTool(BaseTool):
    """Tool that sends ALL executions to Interceptor."""
    
    def _run(self, *args, **kwargs):
        # Package request
        payload = {
            "session_id": self._session_id,
            "tool_name": self.name,
            "args": self._parse_input(*args, **kwargs)
        }
        
        # Send to Interceptor - NOT local execution
        response = httpx.post(
            f"{self._interceptor_url}/v1/proxy-execute",
            json=payload,
            headers={"X-API-Key": self._api_key}
        )
        
        if response.status_code == 403:
            raise SecurityBlockException(...)
        
        return response.json()
```

---

## Message Lifecycle

### Phase 1: User Request

```
User                    Agent API
  │                         │
  │  POST /chat             │
  │  { message: "..." }     │
  │────────────────────────▶│
  │                         │
  │                         │ 1. Validate request
  │                         │ 2. Generate session_id
  │                         │ 3. Create callbacks
  │                         │
```

**Code Path:** `src/main.py` → `chat()` endpoint

### Phase 2: LLM Processing

```
Agent API               LangChain Agent
  │                         │
  │  invoke(message)        │
  │────────────────────────▶│
  │                         │
  │                         │ 1. Load system prompt
  │                         │ 2. Add conversation history
  │                         │ 3. Send to LLM (OpenRouter)
  │                         │
  │                         │     ┌───────────┐
  │                         │────▶│ OpenRouter│
  │                         │     │   LLM     │
  │                         │◀────│           │
  │                         │     └───────────┘
  │                         │
  │                         │ 4. LLM decides tool call
  │                         │
```

**Code Path:** `src/agent/core.py` → `SentinelAgent.invoke()`

### Phase 3: Tool Invocation (Blind Courier)

When the LLM decides to use a tool:

```
LangChain Agent         SentinelSecureTool      Interceptor
  │                         │                       │
  │  call tool              │                       │
  │────────────────────────▶│                       │
  │                         │                       │
  │                         │ 1. Package request    │
  │                         │    - tool_name        │
  │                         │    - args             │
  │                         │    - session_id       │
  │                         │                       │
  │                         │  POST /v1/proxy-execute
  │                         │──────────────────────▶│
  │                         │  X-API-Key header     │
  │                         │                       │
  │                         │                       │ ← TAINT CHECK HERE
  │                         │                       │
```

**Code Path:** `src/transport/interceptor_tool.py` → `SentinelSecureTool._run()`

### Phase 4: Taint Checking (Zone B)

The Interceptor performs security checks:

```
                        Interceptor
                            │
                            │ 1. Authenticate (X-API-Key)
                            │ 2. Validate session
                            │ 3. Load taint rules
                            │
                            ▼
                    ┌───────────────┐
                    │  Taint Rules  │
                    │               │
                    │  tool: file_read
                    │  condition:   │
                    │  "file_path   │
                    │   .startswith │
                    │   ('confid')  │
                    │               │
                    │  taint_level: │
                    │  "HIGH"       │
                    │               │
                    │  requires_    │
                    │  approval:    │
                    │  true         │
                    └───────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │   ALLOWED     │               │    DENIED     │
    │               │               │               │
    │  Continue to  │               │  Return 403   │
    │  Zone C       │               │  + detail     │
    └───────────────┘               └───────────────┘
```

**Configuration:** `config/sentinel_config.yaml` → `taint_rules`

### Phase 5: Tool Execution (Zone C)

If allowed, the operation executes:

```
Interceptor                 MCP Server / Backend
    │                           │
    │  Execute tool             │
    │──────────────────────────▶│
    │                           │
    │                           │ 1. Access resource
    │                           │ 2. Perform operation
    │                           │ 3. Return result
    │                           │
    │◀──────────────────────────│
    │                           │
```

---

## Request/Response Format

### Tool Execution Request

```json
POST /v1/proxy-execute
Headers:
  X-API-Key: <SENTINEL_API_KEY>
  Content-Type: application/json

Body:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "tool_name": "web_search",
  "args": {
    "query": "what is a panda",
    "context": "user question about animals"
  }
}
```

### Success Response (200)

```json
{
  "result": {
    "query": "what is a panda",
    "results": [
      {
        "title": "Giant Panda - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Giant_panda",
        "snippet": "The giant panda is a bear species endemic to China..."
      }
    ],
    "total": 1
  },
  "status": "success"
}
```

### Access Denied Response (403)

```json
{
  "detail": "Access denied by security policy. Taint level: HIGH"
}
```

This triggers `SecurityBlockException` in the agent, which the LLM sees as:
> "Tool 'file_read' was blocked by a security policy. Reason: Access denied..."

---

## Error Handling

### SecurityBlockException

When taint rules block a request:

```python
except httpx.HTTPStatusError as e:
    if e.response.status_code == 403:
        raise SecurityBlockException(
            message=f"Tool '{tool_name}' was blocked by a security policy.",
            tool_name=tool_name,
            reason=detail
        )
```

The LLM receives this exception and can:
1. Inform the user about the restriction
2. Suggest alternative approaches
3. Request different, allowed resources

### System Errors

```python
except Exception as e:
    return f"Result: SYSTEM_ERROR: {str(e)}"
```

Non-blocking - the conversation continues.

---

## Taint Rule Examples

### File Access Rules

```yaml
tools:
  - name: "file_read"
    taint_rules:
      # Block confidential files
      - rule_id: "file_read_confidential"
        condition: "file_path.startswith('confidential/')"
        taint_level: "HIGH"
        requires_approval: true
      
      # Allow public files
      - rule_id: "file_read_public"
        condition: "file_path.startswith('public/')"
        taint_level: "LOW"
        requires_approval: false
```

### Web Search Rules

```yaml
  - name: "web_search"
    taint_rules:
      # Allow all web searches (low risk)
      - rule_id: "web_search_default"
        condition: "true"
        taint_level: "LOW"
        requires_approval: false
```

---

## Observability

### OpenTelemetry Spans

```
Root Span: agent.invoke
├── Span: tool.web_search
│   └── Attributes: 
│       - tool.name: "web_search"
│       - session.id: "abc-123"
│       - http.status_code: 200
├── Span: tool.file_read (if called)
└── Span: llm.invoke
```

### Viewing Traces

1. Open Jaeger UI: http://localhost:16686
2. Select service: `sentinel-rag-agent`
3. Find traces by session ID or time

---

## Security Summary

| Zone | Component | Can Execute? | Trust Level |
|------|-----------|--------------|-------------|
| A | RAG Agent | NO | Untrusted |
| B | Interceptor | NO (checks only) | Trusted |
| C | MCP Server | YES | Highly Trusted |

The agent in Zone A:
- ✅ CAN request tool execution
- ✅ CAN process LLM responses
- ❌ CANNOT execute tools locally
- ❌ CANNOT bypass taint rules
- ❌ CANNOT access data directly

---

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `SENTINEL_URL` | Interceptor base URL | SentinelSecureTool |
| `SENTINEL_API_KEY` | API key for auth | SentinelSecureTool |
| `OPENROUTER_API_KEY` | LLM API key | Agent Core |

---

## Extending the System

### Adding New Tools

1. Define in `config/sentinel_config.yaml`:
```yaml
tools:
  - name: "new_tool"
    description: "Does something new"
    parameters:
      - name: "param1"
        type: "string"
        required: true
    taint_rules:
      - rule_id: "new_tool_default"
        condition: "true"
        taint_level: "LOW"
```

2. Implement handler in Interceptor backend

3. Restart agent to load new configuration

### Custom Taint Rules

Conditions can use:
- Parameter values: `file_path.startswith('...')`
- Equality: `directory == 'confidential'`
- Boolean: `true` (always matches)
