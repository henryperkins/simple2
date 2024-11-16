"""
Azure OpenAI API Client Module

This module provides a client for interacting with the Azure OpenAI API to generate
docstrings for Python functions. It handles configuration and initializes components
necessary for API interaction.

Version: 1.3.0
Author: Development Team
"""

from typing import List, Tuple, Optional, Dict, Any
from cache import Cache
from config import AzureOpenAIConfig
from token_management import TokenManager
from api_interaction import APIInteraction
from logger import log_info, log_error
from exceptions import TooManyRetriesError

class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI to generate docstrings.

    This class manages the configuration and initializes the components necessary
    for API interaction. It provides a high-level interface for generating docstrings
    and managing the cache.
    """

    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        """
        Initialize the AzureOpenAIClient with necessary configuration.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration instance for Azure OpenAI.
                If not provided, will load from environment variables.

        Raises:
            ValueError: If the configuration is invalid
        """
        self.config = config or AzureOpenAIConfig.from_env()
        if not self.config.validate():
            raise ValueError("Invalid Azure OpenAI configuration")

        self.token_manager = TokenManager(
            model=self.config.model_name,
            deployment_name=self.config.deployment_name
        )
        self.cache = Cache()
        self.api_interaction = APIInteraction(
            self.config,
            self.token_manager,
            self.cache
        )

        log_info("Azure OpenAI client initialized successfully")

    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a docstring for a function using Azure OpenAI.

        Args:
            func_name: Name of the function
            params: List of parameter names and types
            return_type: Return type of the function
            complexity_score: Complexity score of the function
            existing_docstring: Existing docstring if any
            decorators: List of decorators
            exceptions: List of exceptions
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature

        Returns:
            Optional[Dict[str, Any]]: Generated docstring and metadata, or None if failed

        Raises:
            TooManyRetriesError: If maximum retry attempts are exceeded
        """
        try:
            return await self.api_interaction.get_docstring(
                func_name=func_name,
                params=params,
                return_type=return_type,
                complexity_score=complexity_score,
                existing_docstring=existing_docstring,
                decorators=decorators,
                exceptions=exceptions,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except TooManyRetriesError as e:
            log_error(f"Max retries exceeded for {func_name}: {e}")
            raise
        except Exception as e:
            log_error(f"Error generating docstring for {func_name}: {e}")
            return None

    async def batch_generate_docstrings(
        self,
        functions: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate docstrings for multiple functions in batches.

        Args:
            functions: List of function metadata dictionaries
            batch_size: Number of functions to process concurrently

        Returns:
            List[Optional[Dict[str, Any]]]: List of generated docstrings and metadata
        """
        results = []
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.generate_docstring(**func) for func in batch
            ], return_exceptions=True)
            
            for func, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    log_error(f"Error processing {func['func_name']}: {result}")
                    results.append(None)
                else:
                    results.append(result)
        
        return results

    def invalidate_cache_for_function(self, func_name: str) -> bool:
        """
        Invalidate all cached responses for a specific function.

        Args:
            func_name: Name of the function to invalidate cache for

        Returns:
            bool: True if cache invalidation was successful
        """
        try:
            return self.cache.invalidate_by_tags([f"func:{func_name}"])
        except Exception as e:
            log_error(f"Failed to invalidate cache for function {func_name}: {e}")
            return False

    def invalidate_cache_by_model(self, model: str) -> bool:
        """
        Invalidate all cached responses for a specific model.

        Args:
            model: Model name to invalidate cache for

        Returns:
            bool: True if cache invalidation was successful
        """
        try:
            return self.cache.invalidate_by_tags([f"model:{model}"])
        except Exception as e:
            log_error(f"Failed to invalidate cache for model {model}: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics and client information
        """
        return {
            'cache_stats': self.cache.stats,
            'client_info': self.get_client_info()
        }

    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the client configuration.

        Returns:
            Dict[str, Any]: Client configuration details
        """
        return {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "is_ready": self.api_interaction.is_ready
        }

    async def validate_connection(self) -> bool:
        """
        Validate the connection to Azure OpenAI service.

        Returns:
            bool: True if connection is successful

        Raises:
            ConnectionError: If connection validation fails
        """
        return await self.api_interaction.validate_connection()

    async def health_check(self) -> bool:
        """
        Perform a health check to verify the service is operational.

        Returns:
            bool: True if the service is healthy
        """
        return await self.api_interaction.health_check()

    async def close(self):
        """Close the client and release any resources."""
        await self.api_interaction.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

async def test_client():
    """Test the AzureOpenAIClient functionality."""
    try:
        async with AzureOpenAIClient() as client:
            # Validate connection
            if not await client.validate_connection():
                log_error("Connection validation failed")
                return

            # Perform health check
            if not await client.health_check():
                log_error("Health check failed")
                return

            # Test docstring generation
            test_response = await client.generate_docstring(
                func_name="example_function",
                params=[("param1", "str"), ("param2", "int")],
                return_type="bool",
                complexity_score=5,
                existing_docstring="",
            )

            if test_response:
                log_info("Test successful!")
                log_info(f"Generated docstring: {test_response['content']['docstring']}")
            else:
                log_error("Test failed!")

            # Get cache statistics
            cache_stats = client.get_cache_stats()
            log_info(f"Cache statistics: {cache_stats}")

    except Exception as e:
        log_error(f"Error testing client: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())