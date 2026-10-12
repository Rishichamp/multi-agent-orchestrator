"""
Orchestrator Agent — the "brain" of the multi-agent ecosystem.

The orchestrator:
  1. Receives a complex task from the user
  2. Analyses the task and creates a decomposition plan
  3. Routes subtasks to the appropriate specialist agents
  4. Collects results and synthesises a final answer
  5. Logs the full execution trace for transparency

Coordination pattern: Supervisor-Worker
  - One orchestrator (gpt-4o) reasons about routing
  - Four specialists (gpt-4o-mini for research/analysis, gpt-4o for writing/code)
  - Orchestrator synthesises final output from specialist results

This is intentionally transparent — every routing decision, specialist
assignment, and result is logged in the execution trace.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI

from app.config import settings
from agents.specialists import ResearchAgent, AnalysisAgent, WriterAgent, CodeAgent

logger = logging.getLogger(__name__)


# ── Execution trace ───────────────────────────────────────────────────────────

@dataclass
class SubtaskResult:
    subtask_id: str
    agent_name: str
    task: str
    result: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    task: str
    plan: list[dict] = field(default_factory=list)
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    agents_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "plan": self.plan,
            "subtask_results": [
                {
                    "subtask_id": r.subtask_id,
                    "agent": r.agent_name,
                    "task": r.task,
                    "result_preview": r.result[:300] + ("..." if len(r.result) > 300 else ""),
                    "duration_ms": round(r.duration_ms, 1),
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.subtask_results
            ],
            "final_answer": self.final_answer,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "agents_used": self.agents_used,
            "steps_count": len(self.subtask_results),
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are an AI Orchestrator managing a team of specialist agents.

Available specialists:
  - ResearchAgent:  Searches web, retrieves and summarises information
  - AnalysisAgent:  Analyses data, performs calculations, identifies trends
  - WriterAgent:    Drafts documents, reports, emails, summaries, blog posts
  - CodeAgent:      Writes code, reviews for bugs and security issues

Your job:
  1. Analyse the user's task
  2. Decompose it into subtasks, each assigned to the most appropriate specialist
  3. Order subtasks so dependencies are respected (e.g. research before analysis)
  4. Synthesise specialist outputs into a final comprehensive answer

DECOMPOSITION RULES:
  - Each subtask must be self-contained and assignable to exactly ONE specialist
  - Research subtasks should come before analysis subtasks that depend on them
  - Writing subtasks should come last (they synthesise everything else)
  - Minimum subtasks: 1 (if the task fits one specialist perfectly)
  - Maximum subtasks: 5 (avoid over-decomposition)
  - If a task needs just one specialist, assign the whole task to them

Return your plan as JSON: 
[{"id": "S1", "agent": "AgentName", "task": "specific task description", "depends_on": []}]
"""

SYNTHESIS_SYSTEM = """You are synthesising results from multiple specialist agents into a final answer.

RULES:
  1. Combine all specialist results into a coherent, unified response
  2. Do not repeat information — consolidate it
  3. Maintain logical flow: research → analysis → recommendations/output
  4. Preserve specific numbers, facts, and code from the specialists
  5. End with clear actionable conclusions or deliverables
  6. Format appropriately for the task type (report, email, analysis, etc.)
"""


class OrchestratorAgent:
    """
    Supervisor-pattern multi-agent orchestrator.

    Workflow:
      query → plan → [specialist execution] → synthesise → return
    """

    AGENT_MAP = {
        "ResearchAgent":  ResearchAgent,
        "AnalysisAgent":  AnalysisAgent,
        "WriterAgent":    WriterAgent,
        "CodeAgent":      CodeAgent,
    }

    def __init__(self):
        self._planner_llm  = ChatOpenAI(
            model=settings.ORCHESTRATOR_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self._synthesis_llm = ChatOpenAI(
            model=settings.ORCHESTRATOR_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )
        # Lazy-init specialists to avoid startup cost
        self._specialists: dict = {}
        logger.info(
            f"OrchestratorAgent ready | planner={settings.ORCHESTRATOR_MODEL} | "
            f"workers={settings.WORKER_MODEL}"
        )

    def _get_specialist(self, name: str):
        if name not in self._specialists:
            if name not in self.AGENT_MAP:
                raise ValueError(f"Unknown specialist: {name}")
            self._specialists[name] = self.AGENT_MAP[name]()
        return self._specialists[name]

    def _create_plan(self, task: str) -> list[dict]:
        """Ask the planner LLM to decompose the task into a list of subtasks."""
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
            {"role": "user",   "content": f"Decompose this task into subtasks:\n\n{task}"},
        ]
        response = self._planner_llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            plan = json.loads(raw.strip())
            if not isinstance(plan, list):
                plan = [plan]
            # Validate each step has required fields
            for step in plan:
                step.setdefault("id", f"S{plan.index(step)+1}")
                step.setdefault("depends_on", [])
                if "agent" not in step or "task" not in step:
                    raise ValueError(f"Invalid plan step: {step}")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Plan parse failed ({e}), using fallback single-agent plan")
            # Fallback: route entire task to the most appropriate agent
            agent = self._classify_task(task)
            return [{"id": "S1", "agent": agent, "task": task, "depends_on": []}]

    def _classify_task(self, task: str) -> str:
        """Simple heuristic to classify a task to a single agent."""
        task_lower = task.lower()
        if any(w in task_lower for w in ["code", "function", "script", "debug", "implement"]):
            return "CodeAgent"
        if any(w in task_lower for w in ["write", "draft", "email", "report", "blog", "document"]):
            return "WriterAgent"
        if any(w in task_lower for w in ["analyse", "calculate", "trend", "compare", "metric"]):
            return "AnalysisAgent"
        return "ResearchAgent"

    def _execute_subtask(
        self,
        step: dict,
        prior_results: dict[str, str],
    ) -> SubtaskResult:
        """Execute a single subtask by routing to the appropriate specialist."""
        agent_name = step["agent"]
        task       = step["task"]
        step_id    = step["id"]

        # Build context from dependencies
        context = ""
        for dep_id in step.get("depends_on", []):
            if dep_id in prior_results:
                context += f"\n[{dep_id} result]:\n{prior_results[dep_id]}\n"

        start = time.time()
        try:
            specialist = self._get_specialist(agent_name)
            # Call with context if the specialist supports it
            if context and hasattr(specialist, "run"):
                import inspect
                sig = inspect.signature(specialist.run)
                if "context" in sig.parameters:
                    result = specialist.run(task, context=context)
                else:
                    result = specialist.run(f"{task}\n\nContext:\n{context}")
            else:
                result = specialist.run(task)

            duration = (time.time() - start) * 1000
            logger.info(f"  [{step_id}] {agent_name} completed in {duration:.0f}ms")
            return SubtaskResult(
                subtask_id=step_id, agent_name=agent_name, task=task,
                result=result, duration_ms=duration, success=True
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"  [{step_id}] {agent_name} failed: {e}")
            return SubtaskResult(
                subtask_id=step_id, agent_name=agent_name, task=task,
                result=f"[{agent_name} encountered an error: {e}]",
                duration_ms=duration, success=False, error=str(e)
            )

    def _synthesise(self, task: str, subtask_results: list[SubtaskResult]) -> str:
        """Use the synthesis LLM to combine all specialist outputs."""
        if len(subtask_results) == 1:
            # Single specialist — no synthesis needed
            return subtask_results[0].result

        results_text = "\n\n".join(
            f"=== {r.agent_name} ({r.subtask_id}) ===\nTask: {r.task}\nResult:\n{r.result}"
            for r in subtask_results if r.success
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": f"Original task: {task}\n\nSpecialist results:\n{results_text}\n\nSynthesise into final answer:"},
        ]
        response = self._synthesis_llm.invoke(messages)
        return response.content

    def run(self, task: str) -> ExecutionTrace:
        """
        Execute the full orchestration pipeline for a task.
        Returns an ExecutionTrace with plan, results, and final answer.
        """
        global_start = time.time()
        trace = ExecutionTrace(task=task)

        logger.info(f"[Orchestrator] New task: {task[:80]}...")

        # Step 1: Plan
        logger.info("[Orchestrator] Creating execution plan...")
        trace.plan = self._create_plan(task)
        logger.info(f"[Orchestrator] Plan: {len(trace.plan)} subtasks → {[s['agent'] for s in trace.plan]}")

        # Step 2: Execute subtasks sequentially (respecting dependencies)
        prior_results: dict[str, str] = {}
        for step in trace.plan:
            result = self._execute_subtask(step, prior_results)
            trace.subtask_results.append(result)
            prior_results[step["id"]] = result.result
            if result.agent_name not in trace.agents_used:
                trace.agents_used.append(result.agent_name)

        # Step 3: Synthesise
        logger.info("[Orchestrator] Synthesising results...")
        trace.final_answer = self._synthesise(task, trace.subtask_results)
        trace.total_duration_ms = (time.time() - global_start) * 1000

        logger.info(
            f"[Orchestrator] Complete | {len(trace.subtask_results)} steps | "
            f"{trace.total_duration_ms:.0f}ms | agents: {trace.agents_used}"
        )
        return trace


# Singleton
_orchestrator_instance: OrchestratorAgent | None = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance
