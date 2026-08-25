# AgenticAlphaa — Multi-Agent Financial Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Async-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Claude-Anthropic-412991?style=for-the-badge&logo=anthropic&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS-ECS%2FECR-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white"/>
</p>

> **AgenticAlphaa** is a production-grade, multi-agent financial intelligence platform. Given a natural-language query such as *"Is NVIDIA a good buy for Q1 2025?"*, it orchestrates a coordinated swarm of specialist AI agents to produce a structured investment analysis report combining live market data, SEC regulatory filings, and real-time social/news sentiment.

---

## Architecture Overview

```
User Query (HTTP)
      │
      ▼
  FastAPI Layer  ──────────────────────────────────────────────────────┐
      │                                                                │
      ▼                                                                │
 ManagerAgent (LangGraph StateGraph)                                   │
  ┌──────────────────────────────────────────────────────────────┐    │
  │  hydrate → brain_route → dispatch → evaluate → persist       │    │
  │  ↓ conditional routing (max 8 loops)                         │    │
  │  → ResearchAgent   → MCP Research  Server (Tavily, NewsAPI)  │    │
  │  → FinancialAgent  → MCP Financial Server (Yahoo, SEC EDGAR) │    │
  │  → SentimentAgent  → MCP Sentiment Server (FinBERT, VADER)   │    │
  └──────────────────────────────────────────────────────────────┘    │
      │                                                                │
      ▼                                                                │
  Hybrid RAG Engine                                                    │
  ┌──────────────────────────────┐                                    │
  │  Supabase pgvector (dense)   │                                    │
  │  Neo4j Knowledge Graph       │                                    │
  │  BAAI/bge-small-en-v1.5      │                                    │
  └──────────────────────────────┘                                    │
      │                                                                │
      ▼                                                                │
  Supabase PostgreSQL (persistence)  ◄───────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (async/await throughout) |
| Web Framework | FastAPI + Uvicorn + Pydantic v2 |
| LLM | Anthropic Claude (`claude-haiku-4-5`) |
| Agent Orchestration | LangGraph `StateGraph` |
| Tool Protocol | MCP (Model Context Protocol) — stdio JSON-RPC |
| Vector Store | Supabase pgvector (`alpha_hybrid_search` RPC) |
| Knowledge Graph | Neo4j (AuraDB / self-hosted, Bolt protocol) |
| Embedding Model | BAAI/bge-small-en-v1.5 (384-dim, sentence-transformers) |
| Sentiment — NLP | ProsusAI/FinBERT (HuggingFace Transformers) |
| Sentiment — Lexical | NLTK VADER |
| Financial Data | Yahoo Finance (`yfinance`) |
| Regulatory Filings | SEC EDGAR (XBRL API + full-text search) |
| News | Tavily API + NewsAPI |
| Persistence | Supabase PostgreSQL |
| Error Tracking | Sentry SDK (optional) |
| LLM Tracing | LangSmith (optional) |
| CI/CD | GitHub Actions (deploy.yml + daily_refresh.yml) |
| Infra | AWS ECS + ECR (OIDC role-based auth) |
| Evaluation | RAGAS framework |

---

## Project Structure

```
AgenticAlphaa/
├── agents/
│   ├── manager_agent.py        # Central LangGraph orchestrator
│   ├── financial_agent.py      # Financial data & ratio specialist
│   ├── research_agent.py       # News & SEC research specialist
│   ├── sentiment_agent.py      # Market sentiment specialist
│   └── state.py                # SharedManagerState schema
├── api/
│   ├── main.py                 # FastAPI application entry point
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze endpoint
│   │   └── progress.py         # SSE progress stream
│   └── core/
│       ├── observability.py    # Structured logging & tracing
│       └── exceptions.py       # Domain exception types
├── tools/
│   ├── financial_tools/        # Yahoo Finance, SEC EDGAR, ratio calc
│   ├── research_tools/         # Tavily, NewsAPI, context synthesizer
│   └── sentiment_tools/        # FinBERT, VADER, Fear & Greed
├── rag/
│   ├── hybrid_rag.py           # Dense + sparse retrieval fusion
│   ├── graph_store.py          # Neo4j knowledge graph interface
│   ├── vector_store.py         # Supabase pgvector client
│   ├── ingestion.py            # Document ingestion pipeline
│   └── embedding_manager.py    # BAAI/bge embedding service
├── memory/
│   └── manager_memory.py       # Short-term + long-term memory
├── scheduler/
│   └── daily_refresh.py        # Nightly RAG re-ingestion
├── evaluation/
│   └── run_ragas.py            # RAGAS evaluation pipeline
├── infra/
│   ├── ecr-stack.yaml          # AWS ECR CloudFormation
│   ├── ecs-stack.yaml          # AWS ECS CloudFormation
│   └── oidc-role-stack.yaml    # GitHub OIDC IAM role
├── tests/
│   ├── unit_tests/             # Per-component unit tests
│   └── integration_tests/      # Full-pipeline integration tests
├── frontend/
│   └── alpha-agent-app.html    # Single-page frontend
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Agent Graph — ManagerAgent

The `ManagerAgent` is a LangGraph `StateGraph` with the following node topology:

```
START → hydrate → brain_route ──┬──► dispatch → evaluate → persist ──┐
                                │                                      │
                                ├──► finalise → END                   │
                                │                                      │
                                └──► abort → END     ◄────────────────┘
                                                (conditional re-route)
```

**Routing Vocabulary:**

| Action | Target |
|---|---|
| `run_research` | Dispatch `ResearchAgent` |
| `run_financial` | Dispatch `FinancialAnalystAgent` |
| `run_sentiment` | Dispatch `SentimentAgent` |
| `rerun_*` | Re-dispatch with updated directives |
| `finalise` | Synthesise final report and exit |
| `abort` | Exit with error (guardrail: max 8 loops) |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Supabase project (pgvector enabled)
- Neo4j instance (AuraDB or self-hosted)
- Anthropic API key

### 1. Clone & Install

```bash
git clone https://github.com/Behroozfili/AgenticAlphaa.git
cd AgenticAlphaa
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=...
NEO4J_URI=bolt://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...      # optional
SENTRY_DSN=...             # optional
```

### 3. Run Locally

```bash
# With Docker Compose
docker-compose up --build

# Full local startup: seeds RAG first, then starts FastAPI/Uvicorn
python main.py

# Developer startup: starts only FastAPI, without running the RAG seed
uvicorn api.main:app --reload --port 3000
```

Both local startup modes default to port `3000` with reload enabled. Override
it with `API_HOST`, `API_PORT`, `API_RELOAD`, or set a comma-separated seed
list with `RAG_SEED_TICKERS`. The direct Uvicorn command intentionally bypasses
the repository-level launcher and therefore does not run `rag.seed`.

### 4. Query the API

```bash
curl -X POST http://localhost:3000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze NVIDIA for Q1 2025", "ticker": "NVDA"}'
```

---

## CI/CD & Deployment

GitHub Actions pipelines handle:
- **CI** (`deploy.yml`): lint, type-check, unit + integration tests
- **Deploy**: build Docker image → push to AWS ECR → deploy to ECS (Fargate)
- **Daily Refresh** (`daily_refresh.py`): nightly RAG document re-ingestion

OIDC-based IAM roles eliminate long-lived AWS credentials from CI.

---

## Evaluation

Run the RAGAS evaluation suite against the live RAG pipeline:

```bash
python evaluation/run_ragas.py
python evaluation/validate_period_consistency.py
```

---

## License

MIT
