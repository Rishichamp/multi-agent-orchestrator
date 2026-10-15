"""
Specialist Agent Definitions for the Custom Agentic AI Ecosystem.

This module defines four specialist agents:
  1. ResearchAgent    — searches web, summarises findings
  2. AnalysisAgent    — analyses data, performs calculations, identifies trends
  3. WriterAgent      — drafts content, reports, emails, summaries
  4. CodeAgent        — writes, reviews, and explains code

Each agent has:
  - A focused role and goal
  - Specialised tools relevant to their domain
  - A carefully crafted system prompt that shapes behaviour
  - Cost-appropriate model assignment (worker model for simple tasks)

The OrchestratorAgent (in orchestrator.py) coordinates these specialists.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from app.config import settings
from agents.safe_math import safe_eval

logger = logging.getLogger(__name__)


# ── Shared tool implementations ───────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current information on any topic. Returns relevant results."""
    # In production: integrate Tavily, Serper, or DuckDuckGo API
    mock_results = {
        "AI agents 2024": "AI agents are autonomous systems using LLMs to plan and execute multi-step tasks. Key frameworks include LangGraph, CrewAI, and AutoGen.",
        "LangChain latest": "LangChain v0.3 introduced LangGraph for stateful agent workflows with better memory and human-in-the-loop support.",
        "default": f"Search results for '{query}': Found relevant information about {query}. Key findings: [1] Recent developments show significant progress. [2] Industry adoption is accelerating. [3] Technical standards are emerging.",
    }
    for key, val in mock_results.items():
        if key.lower() in query.lower():
            return val
    return mock_results["default"]

@tool
def fetch_url(url: str) -> str:
    """Fetch the content of a URL. Use for reading articles, documentation, or web pages."""
    return f"[Content from {url}]: This page contains relevant information about the requested topic. Key sections include: introduction, methodology, findings, and conclusions."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical or statistical expression. Returns the numeric result."""
    try:
        # never eval() model-written text: see agents/safe_math.py
        result = safe_eval(expression, {"round": round, "abs": abs, "sum": sum, "max": max, "min": min, "len": len})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def analyse_data(data_description: str, analysis_type: str) -> str:
    """
    Perform data analysis on described dataset.
    analysis_type: trend_analysis | comparison | summary_stats | anomaly_detection
    """
    return (
        f"Analysis ({analysis_type}) of '{data_description}':\n"
        f"- Mean/median values are within normal range\n"
        f"- Notable trend: upward trajectory of ~15% over the period\n"
        f"- 2 anomalous data points detected (outliers)\n"
        f"- Recommendation: investigate outliers and monitor trend continuation"
    )

@tool
def write_document(doc_type: str, topic: str, key_points: str) -> str:
    """
    Draft a document of the specified type.
    doc_type: email | report_section | summary | blog_post | memo | README
    topic: the subject matter
    key_points: comma-separated key points to include
    """
    templates = {
        "email": f"Subject: {topic}\n\nDear [Recipient],\n\nI am writing regarding {topic}.\n\n{key_points}\n\nBest regards,\n[Sender]",
        "summary": f"EXECUTIVE SUMMARY: {topic}\n\n{key_points}\n\nConclusion: Based on the above, the recommended course of action is...",
        "report_section": f"## {topic}\n\n{key_points}\n\n### Key Findings\n- Finding 1\n- Finding 2\n\n### Recommendations\n- Action item 1",
    }
    return templates.get(doc_type, f"DRAFT {doc_type.upper()}: {topic}\n\n{key_points}")

@tool
def write_code(language: str, task_description: str, requirements: str) -> str:
    """
    Write code for a specific task.
    language: python | javascript | sql | bash
    task_description: what the code should do
    requirements: specific requirements or constraints
    """
    code_templates = {
        "python": f"# {task_description}\n# Requirements: {requirements}\n\ndef solution():\n    # TODO: Implement {task_description}\n    pass\n\nif __name__ == '__main__':\n    result = solution()\n    print(result)",
        "sql": f"-- {task_description}\n-- Requirements: {requirements}\n\nSELECT *\nFROM table_name\nWHERE condition = 'value';",
    }
    return code_templates.get(language.lower(), f"# {language} code for: {task_description}\n# {requirements}")

@tool
def review_code(code: str, review_type: str = "general") -> str:
    """
    Review code for bugs, security issues, or style problems.
    review_type: bugs | security | style | performance | general
    """
    return (
        f"CODE REVIEW ({review_type}):\n"
        f"✓ Logic: Generally sound\n"
        f"⚠ Issue 1: Consider adding error handling for edge cases\n"
        f"⚠ Issue 2: Variable naming could be more descriptive\n"
        f"✓ Security: No obvious vulnerabilities detected\n"
        f"Recommendation: Add unit tests and improve documentation"
    )


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_executor(system_prompt: str, tools: list, model: str = None) -> AgentExecutor:
    """Create an AgentExecutor with the given prompt, tools, and model."""
    llm = ChatOpenAI(
        model=model or settings.WORKER_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools,
        verbose=False,
        max_iterations=settings.MAX_AGENT_STEPS,
        max_execution_time=settings.AGENT_TIMEOUT,
        handle_parsing_errors=True,
    )


# ── Specialist agents ─────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Specialises in information gathering, web search, and fact-finding.
    Uses cheaper model — the task is retrieval, not reasoning.
    """
    NAME = "ResearchAgent"
    DESCRIPTION = "Searches web, retrieves information, verifies facts, summarises sources."

    def __init__(self):
        self._executor = _make_executor(
            system_prompt=(
                "You are a Senior Research Analyst. Your sole job is to find accurate, "
                "relevant information using your search tools. Always cite where you found information. "
                "Be comprehensive but concise. If you cannot find something, say so clearly."
            ),
            tools=[web_search, fetch_url],
            model=settings.WORKER_MODEL,
        )

    def run(self, task: str) -> str:
        logger.info(f"[ResearchAgent] Task: {task[:60]}...")
        result = self._executor.invoke({"input": task})
        return result["output"]


class AnalysisAgent:
    """Specialises in data analysis, trend identification, and calculations."""
    NAME = "AnalysisAgent"
    DESCRIPTION = "Analyses data, performs calculations, identifies trends and anomalies."

    def __init__(self):
        self._executor = _make_executor(
            system_prompt=(
                "You are a Data Analyst. You excel at turning raw data and research findings "
                "into clear, quantified insights. Use tools for calculations. "
                "Lead with the key finding, support with numbers."
            ),
            tools=[calculate, analyse_data],
            model=settings.WORKER_MODEL,
        )

    def run(self, task: str, context: str = "") -> str:
        logger.info(f"[AnalysisAgent] Task: {task[:60]}...")
        full_input = f"{task}\n\nContext from research:\n{context}" if context else task
        result = self._executor.invoke({"input": full_input})
        return result["output"]


class WriterAgent:
    """Specialises in drafting documents, reports, emails, and summaries."""
    NAME = "WriterAgent"
    DESCRIPTION = "Drafts documents, reports, emails, blog posts, and summaries."

    def __init__(self):
        # Writer uses stronger model — quality matters for output text
        self._executor = _make_executor(
            system_prompt=(
                "You are an expert Technical Writer and Content Strategist. "
                "You craft clear, well-structured documents. "
                "Adapt tone to the document type: formal for reports, engaging for blog posts. "
                "Always structure content with clear headings and actionable conclusions."
            ),
            tools=[write_document],
            model=settings.ORCHESTRATOR_MODEL,  # stronger model for writing
        )

    def run(self, task: str, context: str = "") -> str:
        logger.info(f"[WriterAgent] Task: {task[:60]}...")
        full_input = f"{task}\n\nContext:\n{context}" if context else task
        result = self._executor.invoke({"input": full_input})
        return result["output"]


class CodeAgent:
    """Specialises in writing, reviewing, and explaining code."""
    NAME = "CodeAgent"
    DESCRIPTION = "Writes code, reviews for bugs/security, explains technical implementations."

    def __init__(self):
        self._executor = _make_executor(
            system_prompt=(
                "You are a Senior Software Engineer. You write clean, well-documented, "
                "production-ready code. Always include error handling. "
                "When reviewing code, prioritise security and correctness over style."
            ),
            tools=[write_code, review_code, calculate],
            model=settings.ORCHESTRATOR_MODEL,
        )

    def run(self, task: str, context: str = "") -> str:
        logger.info(f"[CodeAgent] Task: {task[:60]}...")
        full_input = f"{task}\n\nContext:\n{context}" if context else task
        result = self._executor.invoke({"input": full_input})
        return result["output"]
