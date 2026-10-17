# Multi-Agent Orchestrator

[![tests](https://github.com/Rishichamp/multi-agent-orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/multi-agent-orchestrator/actions/workflows/tests.yml)

An orchestrator agent that breaks a complex request into subtasks, routes each one to the right specialist agent, and combines their results into one answer, keeping a full execution trace of who did what.

## How it works

```
task ──► Orchestrator ──► plan (subtasks + chosen specialist)
              │
              ├──► ResearchAgent   web search, page fetch, summaries
              ├──► AnalysisAgent   calculations, trends, data analysis
              ├──► WriterAgent     reports, emails, summaries
              └──► CodeAgent       writes and reviews code
              │
              ▼
        synthesised answer + execution trace
```

## Features

- **Task decomposition and routing:** the orchestrator plans subtasks and classifies each one to a specialist.
- **Four specialist agents**, each a LangChain agent with its own tools and system prompt.
- **Execution trace:** every subtask's agent, input, output preview and status (including failures) is returned with the answer.
- **Two model tiers:** `gpt-4o` orchestrates and `gpt-4o-mini` runs the workers.
- **Sandbox mode** and step/time limits for safe experimentation.

## Tech stack

Python 3.11 · LangChain 0.3 · OpenAI · FastAPI · Pydantic · pytest

## Project structure

```
agents/
  orchestrator.py   planning, routing, synthesis, execution trace
  specialists.py    Research, Analysis, Writer and Code agents and their tools
app/
  main.py           FastAPI app and endpoints
  models.py         request/response schemas
  config.py         settings from .env
tests/              14 tests
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # then add your OPENAI_API_KEY
python run.py                  # http://127.0.0.1:8000/docs
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/run` | Submit a task to the agent ecosystem |
| GET | `/info` | Configuration and available agents |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/ -q
```

14 tests covering task classification, execution traces (including failed subtasks), plan defaults and the API, with the LLM mocked.

## Limitations

- `web_search` and `fetch_url` return mock results. Plug in a real search API (Tavily, Serper, DuckDuckGo) for live research.
- The `calculate` tool uses a restricted `eval`. Swap in a safe expression parser before production use.
- Built as part of Module 3 of a Generative AI engineering programme.

## Author

**Rishi Singh** — [GitHub](https://github.com/Rishichamp)
