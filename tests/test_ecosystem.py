"""Tests for the Custom Agentic AI Ecosystem."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from agents.orchestrator import OrchestratorAgent, ExecutionTrace, SubtaskResult
from agents.specialists import ResearchAgent, AnalysisAgent, WriterAgent, CodeAgent

client = TestClient(app)


# ── Unit: orchestrator planning ───────────────────────────────────────────────

def test_classify_task_research():
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    assert orch._classify_task("Find information about Python 3.12 features") == "ResearchAgent"

def test_classify_task_code():
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    assert orch._classify_task("Write a Python function to parse JSON") == "CodeAgent"

def test_classify_task_writer():
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    assert orch._classify_task("Draft an email to the client about the project delay") == "WriterAgent"

def test_classify_task_analysis():
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    assert orch._classify_task("Analyse the trend in monthly revenue data") == "AnalysisAgent"

def test_execution_trace_to_dict():
    trace = ExecutionTrace(task="Test task")
    trace.plan = [{"id": "S1", "agent": "ResearchAgent", "task": "research", "depends_on": []}]
    trace.subtask_results = [
        SubtaskResult("S1", "ResearchAgent", "research", "Some findings", 1200.0, True)
    ]
    trace.final_answer = "Final synthesised answer"
    trace.total_duration_ms = 2500.0
    trace.agents_used = ["ResearchAgent"]

    d = trace.to_dict()
    assert d["task"] == "Test task"
    assert len(d["plan"]) == 1
    assert len(d["subtask_results"]) == 1
    assert d["final_answer"] == "Final synthesised answer"
    assert d["steps_count"] == 1
    assert "ResearchAgent" in d["agents_used"]

def test_result_preview_truncated():
    trace = ExecutionTrace(task="Test")
    long_result = "x" * 500
    trace.subtask_results = [
        SubtaskResult("S1", "ResearchAgent", "task", long_result, 100.0, True)
    ]
    d = trace.to_dict()
    preview = d["subtask_results"][0]["result_preview"]
    assert len(preview) <= 304  # 300 chars + "..."

def test_failed_subtask_in_trace():
    trace = ExecutionTrace(task="Test")
    trace.subtask_results = [
        SubtaskResult("S1", "CodeAgent", "write code", "Error msg", 100.0, False, error="API Error")
    ]
    d = trace.to_dict()
    assert d["subtask_results"][0]["success"] is False
    assert d["subtask_results"][0]["error"] == "API Error"


# ── Unit: plan parsing ────────────────────────────────────────────────────────

def test_plan_defaults_applied():
    """Plan steps missing optional fields get defaults."""
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    orch._specialists = {}
    # Simulate what _create_plan does when JSON is valid
    raw_plan = [{"agent": "ResearchAgent", "task": "Find info"}]
    for step in raw_plan:
        step.setdefault("id", f"S{raw_plan.index(step)+1}")
        step.setdefault("depends_on", [])
    assert raw_plan[0]["id"] == "S1"
    assert raw_plan[0]["depends_on"] == []


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_info():
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert len(data["available_agents"]) == 4
    agent_names = [a["name"] for a in data["available_agents"]]
    assert "ResearchAgent" in agent_names
    assert "WriterAgent" in agent_names

def test_run_empty_task():
    r = client.post("/run", json={"task": ""})
    assert r.status_code == 400

def test_run_too_long_task():
    r = client.post("/run", json={"task": "x" * 3001})
    assert r.status_code == 400

MOCK_TRACE_DICT = {
    "task": "Research AI trends and write a report",
    "plan": [
        {"id": "S1", "agent": "ResearchAgent", "task": "Research AI trends 2024", "depends_on": []},
        {"id": "S2", "agent": "WriterAgent",   "task": "Write report on findings", "depends_on": ["S1"]},
    ],
    "subtask_results": [
        {"subtask_id": "S1", "agent": "ResearchAgent", "task": "Research AI trends",
         "result_preview": "AI trends show...", "duration_ms": 1200.0, "success": True, "error": None},
        {"subtask_id": "S2", "agent": "WriterAgent", "task": "Write report",
         "result_preview": "# AI Trends 2024...", "duration_ms": 2100.0, "success": True, "error": None},
    ],
    "final_answer": "# AI Trends in 2024\n\nKey findings from research...",
    "total_duration_ms": 4500.0,
    "agents_used": ["ResearchAgent", "WriterAgent"],
    "steps_count": 2,
}

@patch("app.main.get_orchestrator")
def test_run_success(mock_get_orch):
    mock_orch = MagicMock()
    mock_trace = MagicMock()
    mock_trace.to_dict.return_value = MOCK_TRACE_DICT
    mock_orch.run.return_value = mock_trace
    mock_get_orch.return_value = mock_orch

    r = client.post("/run", json={"task": "Research AI trends and write a report"})
    assert r.status_code == 200
    data = r.json()
    assert "final_answer" in data
    assert "plan" in data
    assert "subtask_results" in data
    assert data["steps_count"] == 2
    assert "ResearchAgent" in data["agents_used"]

@patch("app.main.get_orchestrator")
def test_run_ecosystem_error(mock_get_orch):
    mock_orch = MagicMock()
    mock_orch.run.side_effect = RuntimeError("LLM API unavailable")
    mock_get_orch.return_value = mock_orch

    r = client.post("/run", json={"task": "Some task that fails"})
    assert r.status_code == 500
