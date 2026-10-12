"""Configuration for the Custom Agentic AI Ecosystem."""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    LLM_PROVIDER: str      = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")
    ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o")
    WORKER_MODEL: str      = os.getenv("WORKER_MODEL", "gpt-4o-mini")
    MAX_AGENT_STEPS: int   = int(os.getenv("MAX_AGENT_STEPS", "12"))
    AGENT_TIMEOUT: int     = int(os.getenv("AGENT_TIMEOUT", "180"))
    SANDBOX_MODE: bool     = os.getenv("SANDBOX_MODE", "true").lower() == "true"
    APP_HOST: str          = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int          = int(os.getenv("APP_PORT", "8001"))
    DEBUG: bool            = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
