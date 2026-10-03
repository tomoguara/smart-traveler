"""Configuration module for the Smart Travel Planning Multi-Agent System.

This module handles API key configuration and environment variable loading.

API Keys Required:
- OPENAI_API_KEY: For GPT-4/5 language models
- SERPAPI_API_KEY: For flight and hotel search (Google Flights/Hotels engine)
- TAVILY_API_KEY: For web search and deep research
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class APIConfig:
    """Configuration class for API keys and settings."""

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your environment or .env file."
            )
        return key

    @property
    def serp_api_key(self) -> str:
        """Get SerpAPI key from environment."""
        key = os.getenv("SERPAPI_API_KEY")
        if not key:
            raise ValueError(
                "SERPAPI_API_KEY not found. Please set it in your environment or .env file."
            )
        return key

    @property
    def tavily_api_key(self) -> str:
        """Get Tavily API key from environment."""
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError(
                "TAVILY_API_KEY not found. Please set it in your environment or .env file."
            )
        return key


# Global config instance
config = APIConfig()
