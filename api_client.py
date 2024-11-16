"""
Azure OpenAI API Client Module

This module provides a client for interacting with the Azure OpenAI API to generate
docstrings for Python functions. It handles configuration and initializes components
necessary for API interaction.

Version: 1.3.1
Author: Development Team
"""

import asyncio
from typing import List, Tuple, Optional, Dict, Any
from cache import Cache
from config import AzureOpenAIConfig
from token_management import TokenManager
from api_interaction import APIInteraction
from logger import log_info, log_error, log_debug, log_warning
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
        log_debug("Initializing Azure OpenAI Client")
        self.config = config or AzureOpenAIConfig.from_env()
        if not self.config.validate():
            log_error("Invalid Azure OpenAI configuration")
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
        log_debug(f"Generating docstring for function: {func_name}")
        try:
            response = await self.api_interaction.get_docstring(
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
            if response:
                log_info(f"Successfully generated docstring for function: {func_name}")
            else:
                log_warning(f"Failed to generate docstring for function: {func_name}")
            return response

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
        log_debug(f"Starting batch generation of docstrings for {len(functions)} functions")
        results = []
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            log_debug(f"Processing batch of {len(batch)} functions")
            batch_results = await asyncio.gather(*[
                self.generate_docstring(**func) for func in batch
            ], return_exceptions=True)
            
            for func, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    log_error(f"Error processing {func['func_name']}: {result}")
                    results.append(None)
                else:
                    results.append(result)
        
        log_info("Batch generation of docstrings completed")
        return results

    def invalidate_cache_for_function(self, func_name: str) -> bool:
        """
        Invalidate all cached responses for a specific function.

        Args:
            func_name: Name of the function to invalidate cache for

        Returns:
            bool: True if cache invalidation was successful
        """
        log_debug(f"Invalidating cache for function: {func_name}")
        try:
            invalidated_count = self.cache.invalidate_by_tags([f"func:{func_name}"])
            if invalidated_count > 0:
                log_info(f"Successfully invalidated cache for function: {func_name}")
            else:
                log_warning(f"No cache entries found to invalidate for function: {func_name}")
            return invalidated_count > 0
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
        log_debug(f"Invalidating cache for model: {model}")
        try:
            invalidated_count = self.cache.invalidate_by_tags([f"model:{model}"])
            if invalidated_count > 0:
                log_info(f"Successfully invalidated cache for model: {model}")
            else:
                log_warning(f"No cache entries found to invalidate for model: {model}")
            return invalidated_count > 0
        except Exception as e:
            log_error(f"Failed to invalidate cache for model {model}: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics and client information
        """
        log_debug("Retrieving cache statistics")
        stats = {
            'cache_stats': self.cache.stats,
            'client_info': self.get_client_info()
        }
        log_info(f"Cache statistics retrieved: {stats}")
        return stats

    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the client configuration.

        Returns:
            Dict[str, Any]: Client configuration details
        """
        client_info = {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "is_ready": self.api_interaction.is_ready
        }
        log_debug(f"Client information: {client_info}")
        return client_info

    async def validate_connection(self) -> bool:
        """
        Validate the connection to Azure OpenAI service.

        Returns:
            bool: True if connection is successful

        Raises:
            ConnectionError: If connection validation fails
        """
        log_debug("Validating connection to Azure OpenAI service")
        try:
            result = await self.api_interaction.validate_connection()
            if result:
                log_info("Connection to Azure OpenAI service validated successfully")
            else:
                log_warning("Connection validation failed")
            return result
        except Exception as e:
            log_error(f"Connection validation failed: {e}")
            raise ConnectionError(f"Connection validation failed: {e}")

    async def health_check(self) -> bool:
        """
        Perform a health check to verify the service is operational.

        Returns:
            bool: True if the service is healthy
        """
        log_debug("Performing health check on Azure OpenAI service")
        try:
            result = await self.api_interaction.health_check()
            if result:
                log_info("Health check passed")
            else:
                log_warning("Health check failed")
            return result
        except Exception as e:
            log_error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the client and release any resources."""
        log_debug("Closing Azure OpenAI client")
        try:
            await self.api_interaction.close()
            log_info("Azure OpenAI client closed successfully")
        except Exception as e:
            log_error(f"Error closing Azure OpenAI client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        log_debug("Entering async context manager for Azure OpenAI client")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        log_debug("Exiting async context manager for Azure OpenAI client")
        await self.close()

async def test_client():
    """Test the AzureOpenAIClient functionality."""
    log_debug("Testing AzureOpenAIClient functionality")
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