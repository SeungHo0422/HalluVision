"""
Configuration settings for the pipeline
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration"""

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')

    # Model Settings (logical names)
    DEFAULT_MODEL = "gpt-3.5-turbo"
    ALTERNATIVE_MODEL = "gpt-4o-mini"

    # Pinned snapshot identifiers for reproducibility (Reviewer 2 #9).
    # When set, GPTClient routes the logical name to the snapshot below.
    MODEL_VERSIONS = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        # open-source / future backends
        "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    }

    # Generation hyperparameters (applied to every API call unless overridden)
    LLM_TEMPERATURE = 0.0
    LLM_TOP_P = 1.0
    LLM_SEED = 42  # OpenAI accepts `seed` for deterministic mode

    # Summarization Settings
    SUMMARY_MIN_WORDS = 50
    SUMMARY_MAX_WORDS = 70
    SUMMARY_TEMPERATURE = LLM_TEMPERATURE
    SUMMARY_SYSTEM_MESSAGE = (
        "You are a careful CTI domain expert. Follow the user instructions exactly "
        "and refuse to introduce information that is not in the source text."
    )

    # QA Settings
    QA_TEMPERATURE = LLM_TEMPERATURE
    QA_LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
    QA_SYSTEM_MESSAGE = (
        "You are a careful CTI NER expert. Always answer with the exact format the "
        "user requests; output 'no answer' when the source text does not contain "
        "an answer."
    )

    # File Paths
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_DATA_DIR = "data"

    # Scraping Settings
    DEFAULT_SOURCES = ["threatpost"]
    DEFAULT_PARAGRAPH_LEVEL = 3

    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate if OpenAI API key is available"""
        return cls.OPENAI_API_KEY is not None and len(cls.OPENAI_API_KEY) > 0

    @classmethod
    def get_model_for_task(cls, task: str = "summarization") -> str:
        """Get appropriate model for specific task."""
        task_models = {
            "summarization": cls.DEFAULT_MODEL,
            "qa": cls.DEFAULT_MODEL,
            "general": cls.DEFAULT_MODEL,
        }
        return task_models.get(task, cls.DEFAULT_MODEL)

    @classmethod
    def resolve_model_snapshot(cls, model: str) -> str:
        """Map logical model name → pinned snapshot id (returns input if unknown)."""
        return cls.MODEL_VERSIONS.get(model, model)

    @classmethod
    def llm_run_config(cls, task: str = "summarization") -> dict:
        """
        Return the full LLM run configuration for one task.

        Used by GPTClient to (a) seed deterministic mode, and (b) write a
        per-run manifest for reproducibility (Reviewer 2 #9).
        """
        if task == "qa":
            system_msg = cls.QA_SYSTEM_MESSAGE
            temperature = cls.QA_TEMPERATURE
        else:
            system_msg = cls.SUMMARY_SYSTEM_MESSAGE
            temperature = cls.SUMMARY_TEMPERATURE

        model = cls.get_model_for_task(task)
        return {
            "task": task,
            "model_logical": model,
            "model_snapshot": cls.resolve_model_snapshot(model),
            "temperature": temperature,
            "top_p": cls.LLM_TOP_P,
            "seed": cls.LLM_SEED,
            "system_message": system_msg,
        }
