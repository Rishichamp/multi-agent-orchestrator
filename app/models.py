"""Pydantic models for the Agentic Ecosystem API."""
from pydantic import BaseModel, Field
from typing import Optional


class EcosystemRequest(BaseModel):
    task: str = Field(..., description="Complex task for the multi-agent ecosystem to solve")
    model_config = {
        "json_schema_extra": {
            "example": {
                "task": (
                    "Research the latest advancements in RAG (Retrieval-Augmented Generation) in 2024, "
                    "analyse which techniques show the most promise, and write a technical blog post "
                    "summarising the findings for an AI engineering audience."
                )
            }
        }
    }


class SubtaskResultSchema(BaseModel):
    subtask_id: str
    agent: str
    task: str
    result_preview: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


class EcosystemResponse(BaseModel):
    task: str
    plan: list[dict]
    subtask_results: list[SubtaskResultSchema]
    final_answer: str
    total_duration_ms: float
    agents_used: list[str]
    steps_count: int


class AgentInfo(BaseModel):
    name: str
    description: str
    model: str


class EcosystemInfoResponse(BaseModel):
    status: str
    orchestrator_model: str
    worker_model: str
    sandbox_mode: bool
    available_agents: list[AgentInfo]
