# Sentinel RAG Template

A secure-by-default RAG (Retrieval-Augmented Generation) client implementing the **"Blind Courier"** pattern in a Zero Trust architecture.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The Sentinel RAG Agent acts as a secure intermediary between users and sensitive data. Unlike traditional RAG implementations, this agent has **no local execution authority** - all tool calls are wrapped and sent to a separate backend (Interceptor) for policy enforcement and execution.

### Key Features

- 🔒 **Zero Trust Architecture**: Agent cannot execute tools locally
- 🏷️ **Taint Tracking**: Data classification with policy enforcement
- 📊 **Full Observability**: OpenTelemetry integration for tracing
- 🔌 **Modular Design**: Clean separation of concerns
- 🐳 **Docker Ready**: Production-ready containerization

---

## How It Works

```
User: "I want to know what is a panda"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Sentinel RAG Agent                                         │
│                                                             │
│  1. Receive user prompt                                     │
│  2. LLM decides: need web_search tool                       │
│  3. SentinelSecureTool packages request:                    │
│     {session_id, tool_name: "web_search", args: {...}}      │
│  4. POST to Interceptor /v1/proxy-execute                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Interceptor (Zone B)                                       │
│                                                             │
│  1. Validate X-API-Key                                      │
│  2. Check taint rules                                       │
│  3. Execute tool (if allowed)                               │
│  4. Return result or 403                                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
User receives: "A panda is a bear species endemic to China..."
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- OpenRouter API Key

### 1. Clone and Configure

```bash
git clone https://github.com/your-org/sentinel-rag-template.git
cd sentinel-rag-template

# Create environment file
cp env.template .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```bash
SENTINEL_URL=http://localhost:8080
SENTINEL_API_KEY=your-api-key
OPENROUTER_API_KEY=your-openrouter-key
```

### 2. Start Services

```bash
docker compose up -d
```

### 3. Test the Agent

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a panda?"}'
```

📚 **Full setup guide:** [docs/SETUP.md](./docs/SETUP.md)

---

## Project Structure

```
sentinel-rag-template/
├── config/
│   ├── sentinel_config.yaml    # Tool definitions & taint rules
│   └── agent_prompts.py        # System prompts
├── data/
│   ├── public/                 # Open access files
│   └── confidential/           # Restricted files
├── src/
│   ├── main.py                 # FastAPI entry point
│   ├── bootstrap.py            # Startup & configuration
│   ├── agent/
│   │   ├── core.py             # LangChain agent
│   │   └── callbacks.py        # Observability callbacks
│   └── transport/
│       ├── client.py           # Admin HTTP client
│       └── interceptor_tool.py # SentinelSecureTool (Blind Courier)
├── monitoring/
│   └── otel_config.yaml        # OpenTelemetry configuration
├── docs/
│   ├── SETUP.md                # Setup guide
│   └── ARCHITECTURE_FLOW.md    # Architecture documentation
├── docker-compose.yaml
├── Dockerfile
└── requirements.txt
```

---

## The Blind Courier Pattern

The core of this system is `SentinelSecureTool`:

```python
class SentinelSecureTool(BaseTool):
    """Tool that routes ALL executions to Interceptor."""
    
    def _run(self, *args, **kwargs):
        payload = {
            "session_id": self._session_id,
            "tool_name": self.name,
            "args": self._parse_input(*args, **kwargs)
        }
        
        # Send to Interceptor - NO local execution
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

## Configuration

### Tool Definitions

Define available tools in `config/sentinel_config.yaml`:

```yaml
tools:
  - name: "file_read"
    description: "Read file contents"
    parameters:
      - name: "file_path"
        type: "string"
        required: true
    taint_rules:
      - rule_id: "confidential_check"
        condition: "file_path.startswith('confidential/')"
        taint_level: "HIGH"
        requires_approval: true

  - name: "web_search"
    description: "Search the web"
    parameters:
      - name: "query"
        type: "string"
        required: true
    taint_rules:
      - rule_id: "web_search_default"
        condition: "true"
        taint_level: "LOW"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTINEL_URL` | Interceptor URL | `http://localhost:8080` |
| `SENTINEL_API_KEY` | Interceptor auth key | Required |
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `LLM_MODEL` | Model to use | `gpt-4-turbo-preview` |
| `OTEL_ENABLED` | Enable telemetry | `true` |

---

## API Reference

### POST /chat

Send a message to the agent.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What files are in the public folder?",
    "session_id": "optional-session-id"
  }'
```

Response:
```json
{
  "session_id": "abc-123",
  "response": "The public folder contains...",
  "tool_calls": [
    {"name": "file_search", "args": {"directory": "public"}}
  ]
}
```

### GET /health

Check service health.

### GET /config

View current configuration.

---

## Security Model

| Zone | Component | Can Execute? | Trust Level |
|------|-----------|--------------|-------------|
| A | RAG Agent | ❌ NO | Untrusted |
| B | Interceptor | Checks only | Trusted |
| C | MCP Server | ✅ YES | Highly Trusted |

The agent:
- ✅ CAN request tool execution
- ❌ CANNOT execute tools locally
- ❌ CANNOT bypass taint rules
- ❌ CANNOT access data directly

---

## Development

### Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run mock interceptor
python mock_interceptor.py

# Run agent (separate terminal)
python -m src.main
```

### CLI Mode

```bash
python -m src.main cli
```

---

## Documentation

- [Setup Guide](./docs/SETUP.md) - Detailed setup instructions
- [Architecture Flow](./docs/ARCHITECTURE_FLOW.md) - Technical architecture and message flow

---

## License

MIT License - see [LICENSE](./LICENSE) for details.
