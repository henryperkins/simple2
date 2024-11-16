"""
Configuration Module for Azure OpenAI Integration

This module manages configuration settings for Azure OpenAI services,
including environment-specific settings, model parameters, and rate limiting.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class AzureOpenAIConfig:
    """Configuration settings for Azure OpenAI."""
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    model_name: str
    max_tokens: int
    temperature: float
    max_retries: int
    retry_delay: int
    request_timeout: int

    @classmethod
    def from_env(cls) -> 'AzureOpenAIConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            AzureOpenAIConfig: Configuration instance
        """
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            model_name=os.getenv("MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("MAX_TOKENS", 4000)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_retries=int(os.getenv("MAX_RETRIES", 3)),
            retry_delay=int(os.getenv("RETRY_DELAY", 2)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 30))
        )

    def validate(self) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid
        """
        required_fields = [
            self.endpoint,
            self.api_key,
            self.api_version,
            self.deployment_name
        ]
        missing_fields = [field for field in required_fields if not field]
        if missing_fields:
            logging.error(f"Missing configuration fields: {missing_fields}")
        return not missing_fields

# Create default configuration instance
try:
    default_config = AzureOpenAIConfig.from_env()
except ValueError as e:
    logging.error(f"Failed to create default configuration: {e}")