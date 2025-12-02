# Sentinel RAG Template - Setup Guide

> **Time to Complete:** ~10 minutes

This guide will walk you through setting up the Sentinel RAG Template from scratch.

---

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Check Command |
|------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Python | 3.11+ | `python --version` |
| Git | Any | `git --version` |

### API Keys Required

- **OpenRouter API Key**: For LLM operations
  - Get one at: https://openrouter.ai/keys
- **Sentinel API Key**: For Interceptor authentication (use any string for development)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/sentinel-rag-template.git
cd sentinel-rag-template
```

---

## Step 2: Environment Configuration

### 2.1 Create your environment file

```bash
cp env.template .env
```

### 2.2 Edit `.env` with your credentials

Open `.env` in your editor and fill in the required values:

```bash
# Required - Sentinel Interceptor
SENTINEL_URL=http://localhost:8080
SENTINEL_API_KEY=your-sentinel-api-key-here

# Required - Your OpenRouter API key
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Optional - Customize model
LLM_MODEL=gpt-4-turbo-preview
```

### 2.3 Verify your configuration

```bash
# Check that required variables are set
grep "OPENROUTER_API_KEY\|SENTINEL_API_KEY" .env
```

---

## Step 3: Configuration Review

### 3.1 Review `config/sentinel_config.yaml`

This file defines:
- Available tools and their parameters
- Taint rules for security policy enforcement
- Session and rate limiting settings

Key sections to review:

```yaml
# Tool definitions - what the agent can do
tools:
  - name: "file_read"
    # ... tool configuration
  - name: "web_search"
    # ... tool configuration

# Taint rules - security policies
taint_rules:
  - rule_id: "file_read_confidential"
    condition: "file_path.startswith('confidential/')"
    taint_level: "HIGH"
    requires_approval: true
```

### 3.2 Customize prompts (optional)

Edit `config/agent_prompts.py` to customize the agent's behavior and responses.

---

## Step 4: Start the Services

### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f agent
```

This starts:
- **agent**: The Sentinel RAG Agent (port 8000)
- **redis**: Session storage (port 6379)
- **mock-interceptor**: Development backend (port 8080)
- **otel-collector**: Telemetry collection
- **jaeger**: Tracing UI (port 16686)

### Option B: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the mock interceptor (in a separate terminal)
python mock_interceptor.py

# Start the agent
python -m src.main
```

---

## Step 5: Verify Installation

### 5.1 Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service_name": "sentinel-rag-agent",
  "version": "1.0"
}
```

### 5.2 Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a panda?"}'
```

The flow:
1. User sends: "What is a panda?"
2. LLM decides to use `web_search` tool
3. Agent sends to Interceptor: `POST /v1/proxy-execute` with `{tool_name, args, session_id}`
4. Interceptor checks taint rules and executes
5. Result returned to agent → synthesized into response

### 5.3 Test Interceptor Directly

Test the `/v1/proxy-execute` endpoint that tools call:

```bash
curl -X POST http://localhost:8080/v1/proxy-execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "session_id": "test-123",
    "tool_name": "web_search",
    "args": {"query": "what is a panda"}
  }'
```

📚 **Full testing guide:** [TESTING.md](./TESTING.md)

### 5.3 View Traces (Optional)

Open Jaeger UI at: http://localhost:16686

---

## Step 6: Add Your Data

### 6.1 Add public files

Place unrestricted files in `data/public/`:

```bash
# Example: Add a document
cp your-document.txt data/public/

# Or create directories
mkdir -p data/public/docs
cp *.md data/public/docs/
```

### 6.2 Add confidential files

Place sensitive files in `data/confidential/`:

```bash
# Example: Add restricted documents
cp sensitive-report.pdf data/confidential/
```

> **Note:** Files in `confidential/` are subject to taint rules and may require approval.

### 6.3 Restart to sync files

```bash
docker compose restart agent
```

---

## Quick Reference

### Common Commands

| Action | Command |
|--------|---------|
| Start all services | `docker compose up -d` |
| Stop all services | `docker compose down` |
| View agent logs | `docker compose logs -f agent` |
| Restart agent | `docker compose restart agent` |
| Run CLI mode | `python -m src.main cli` |

### Service URLs

| Service | URL |
|---------|-----|
| Agent API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |
| Mock Interceptor | http://localhost:8080 |
| Jaeger UI | http://localhost:16686 |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | View configuration |
| POST | `/chat` | Send message to agent |
| POST | `/chat/stream` | Stream response |

---

## Troubleshooting

### Agent won't start

1. Check logs: `docker compose logs agent`
2. Verify `.env` file exists and has valid `OPENROUTER_API_KEY` and `SENTINEL_API_KEY`
3. Ensure port 8000 is not in use

### "Tool was blocked by a security policy"

This is expected behavior when:
- Accessing files in `confidential/` directory
- Taint rules are blocking the request

Check `config/sentinel_config.yaml` for rule definitions.

### Connection refused to Interceptor

1. Ensure mock-interceptor is running: `docker compose ps`
2. Check interceptor logs: `docker compose logs mock-interceptor`
3. Verify `SENTINEL_URL` in `.env`

### OpenTelemetry errors

If you see OTLP connection errors, they're non-fatal. To disable:
```bash
OTEL_ENABLED=false docker compose up -d agent
```

---

## Next Steps

1. Read [ARCHITECTURE_FLOW.md](./ARCHITECTURE_FLOW.md) to understand the message lifecycle
2. Customize `config/sentinel_config.yaml` for your use case
3. Implement your actual Interceptor backend (replace mock)
4. Configure S3 for file synchronization (see `src/bootstrap.py`)

---

## Support

For issues and questions:
- Check existing issues in the repository
- Create a new issue with logs and configuration details
