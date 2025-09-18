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
    
    # Model Settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    ALTERNATIVE_MODEL = "gpt-4o-mini"
    
    # Summarization Settings
    SUMMARY_MIN_WORDS = 50
    SUMMARY_MAX_WORDS = 70
    SUMMARY_TEMPERATURE = 0
    
    # QA Settings
    QA_TEMPERATURE = 0
    QA_LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
    
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
        """
        Get appropriate model for specific task
        
        Args:
            task (str): Task type ("summarization", "qa", "general")
            
        Returns:
            str: Model name
        """
        task_models = {
            "summarization": cls.DEFAULT_MODEL,
            "qa": cls.DEFAULT_MODEL,
            "general": cls.DEFAULT_MODEL
        }
        
        return task_models.get(task, cls.DEFAULT_MODEL)
