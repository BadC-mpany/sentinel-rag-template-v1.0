# Quick Start Guide

## Prerequisites

1. Interceptor running on port 8000
2. Redis running on port 6379
3. MCP server running (managed by Interceptor)
4. OpenRouter API key

## Setup

### 1. Create .env file

```bash
cp env.template .env
```

### 2. Edit .env

```bash
# Required - Interceptor URL (already running on 8000)
SENTINEL_URL=http://localhost:8000

# Required - API key from interceptor policies.yaml
SENTINEL_API_KEY=sk_live_demo_123

# Required - Your OpenRouter API key
OPENROUTER_API_KEY=your-openrouter-key-here

# Optional - Model selection
LLM_MODEL=qwen/qwen3-next-80b-a3b-instruct
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the agent

```bash
python -m src.main
```

Agent will start on port 8001 (to avoid conflict with Interceptor on 8000).

## Test

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a panda?"}'
```

## Flow

1. Agent receives user message
2. LLM decides which tool to use
3. Agent sends POST to Interceptor: `POST http://localhost:8000/v1/proxy-execute`
4. Interceptor checks policies, forwards to MCP
5. MCP executes tool, returns result through Interceptor
6. Agent receives result, synthesizes response

## CLI Mode

```bash
python -m src.main cli
```

