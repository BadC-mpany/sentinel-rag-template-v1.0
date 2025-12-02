# Sentinel RAG Template

A secure RAG client implementing the "Blind Courier" pattern in a Zero Trust architecture.

## Architecture

```
sentinel-rag-template/
├── agent/                 # RAG Agent (Blind Courier)
│   ├── src/
│   │   ├── main.py       # FastAPI entry point
│   │   ├── bootstrap.py  # Configuration loading
│   │   ├── core/         # Agent logic
│   │   └── transport/    # Interceptor communication
│   └── config/           # Agent configuration
├── mcp/                   # MCP Server (Secure Execution)
│   ├── src/
│   │   └── server.py     # Tool execution server
│   └── data/             # Secure data store
│       ├── public/
│       └── confidential/
├── venv/
├── docker-compose.yaml
├── Dockerfile.agent
├── Dockerfile.mcp
├── env.template
├── requirements.txt
└── README.md
```

## Flow

```
User -> Agent (port 8001) -> Interceptor (port 8000) -> MCP (port 9000)
                                    |
                               Redis (6379)
```

The Agent cannot communicate with MCP directly.

## Setup

### 1. Configure environment

```bash
cp env.template .env
# Edit .env with your API keys
```

### 2. Start services

**Option A: Local development**

Terminal 1 - MCP Server:
```bash
source venv/bin/activate
python -m mcp.src.server
```

Terminal 2 - Agent:
```bash
source venv/bin/activate
python -m agent.src.main
```

**Option B: Docker**

```bash
docker compose up -d
```

### 3. Test

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a panda?"}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SENTINEL_URL | Interceptor URL | http://localhost:8000 |
| SENTINEL_API_KEY | Interceptor auth key | Required |
| OPENROUTER_API_KEY | OpenRouter API key | Required |
| AGENT_PORT | Agent server port | 8001 |
| MCP_PORT | MCP server port | 9000 |
| MCP_INTERCEPTOR_TOKEN | Token for MCP auth | mcp-secret-token |
