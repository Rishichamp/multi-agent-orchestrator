"""
Custom Agentic AI Ecosystem — FastAPI Application
==================================================
Endpoints:
  POST /run                — submit task to the multi-agent ecosystem
  GET  /info               — ecosystem configuration and available agents
  GET  /health             — health check
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import (
    EcosystemRequest, EcosystemResponse, SubtaskResultSchema,
    EcosystemInfoResponse, AgentInfo,
)
from agents.orchestrator import get_orchestrator
from agents.specialists import (
    ResearchAgent, AnalysisAgent, WriterAgent, CodeAgent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Custom Agentic AI Ecosystem",
    description=(
        "Multi-agent AI ecosystem with a supervisor orchestrator coordinating "
        "specialised Research, Analysis, Writer, and Code agents."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:sans-serif;max-width:700px;margin:40px auto">
    <h1>🤖 Custom Agentic AI Ecosystem</h1>
    <p>Multi-agent system: <b>Orchestrator</b> coordinates <b>Research + Analysis + Writer + Code</b> agents.</p>
    <p><a href="/docs">📖 Swagger UI</a> | <a href="/info">ℹ️ Ecosystem Info</a></p>
    <h3>Example complex tasks:</h3>
    <ul>
      <li>Research recent AI agent frameworks, analyse their trade-offs, and write a comparison report.</li>
      <li>Find the latest Python async best practices, then write production-quality async code for a web scraper.</li>
      <li>Analyse trends in cloud computing costs, calculate 3-year TCO for a mid-size startup, and draft an executive memo.</li>
      <li>Research RAG techniques, identify the top 3 improvements, write a technical blog post about them.</li>
    </ul>
    </body></html>"""


@app.get("/health")
async def health():
    return {"status": "ok", "orchestrator_model": settings.ORCHESTRATOR_MODEL, "worker_model": settings.WORKER_MODEL}


@app.get("/info", response_model=EcosystemInfoResponse)
async def info():
    return EcosystemInfoResponse(
        status="ready",
        orchestrator_model=settings.ORCHESTRATOR_MODEL,
        worker_model=settings.WORKER_MODEL,
        sandbox_mode=settings.SANDBOX_MODE,
        available_agents=[
            AgentInfo(name=ResearchAgent.NAME, description=ResearchAgent.DESCRIPTION, model=settings.WORKER_MODEL),
            AgentInfo(name=AnalysisAgent.NAME, description=AnalysisAgent.DESCRIPTION, model=settings.WORKER_MODEL),
            AgentInfo(name=WriterAgent.NAME,   description=WriterAgent.DESCRIPTION,   model=settings.ORCHESTRATOR_MODEL),
            AgentInfo(name=CodeAgent.NAME,     description=CodeAgent.DESCRIPTION,     model=settings.ORCHESTRATOR_MODEL),
        ],
    )


@app.post("/run", response_model=EcosystemResponse)
async def run_ecosystem(req: EcosystemRequest):
    """
    Submit a complex task to the multi-agent ecosystem.

    The orchestrator will:
    1. Decompose the task into subtasks
    2. Assign each subtask to the most appropriate specialist
    3. Execute subtasks (respecting dependencies)
    4. Synthesise results into a final comprehensive answer

    Returns the full execution trace including plan, each agent's output,
    and the synthesised final answer.
    """
    if not req.task.strip():
        raise HTTPException(400, "Task cannot be empty.")
    if len(req.task) > 3000:
        raise HTTPException(400, "Task too long (max 3000 chars).")

    try:
        orchestrator = get_orchestrator()
        trace = orchestrator.run(req.task)
        trace_dict = trace.to_dict()
        return EcosystemResponse(
            task=trace_dict["task"],
            plan=trace_dict["plan"],
            subtask_results=[SubtaskResultSchema(**r) for r in trace_dict["subtask_results"]],
            final_answer=trace_dict["final_answer"],
            total_duration_ms=trace_dict["total_duration_ms"],
            agents_used=trace_dict["agents_used"],
            steps_count=trace_dict["steps_count"],
        )
    except Exception as e:
        logger.error(f"Ecosystem error: {e}", exc_info=True)
        raise HTTPException(500, f"Ecosystem execution failed: {str(e)}")
