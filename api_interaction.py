"""
API Interaction Module

This module handles interactions with the Azure OpenAI API, including making requests,
handling retries, managing rate limits, and validating connections.

Version: 1.1.0
Author: Development Team
"""

import asyncio
import time
import openai
from typing import List, Tuple, Optional, Dict, Any
from openai.error import OpenAIError
from logger import log_info, log_error, log_debug, log_warning
from token_management import TokenManager
from cache import Cache
from response_parser import ResponseParser
from config import AzureOpenAIConfig
from exceptions import TooManyRetriesError


class APIInteraction:
    """Handles interactions with the Azure OpenAI API."""

    def __init__(
        self, config: AzureOpenAIConfig, token_manager: TokenManager, cache: Cache
    ):
        """Initializes the APIInteraction with necessary components."""
        log_debug("Initializing APIInteraction with Azure OpenAI configuration")
        openai.api_key = config.api_key
        openai.api_base = config.endpoint  # Azure-specific endpoint
        openai.api_version = config.api_version
        self.token_manager = token_manager
        self.cache = cache
        self.parser = ResponseParser()
        self.config = config
        self.current_retry = 0
        log_info("APIInteraction initialized successfully.")

    async def get_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generates a docstring for a function using Azure OpenAI."""
        log_debug(f"Generating docstring for function: {func_name}")
        cache_key = f"{func_name}:{hash(str(params))}:{hash(return_type)}"

        try:
            # Check cache first
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                log_info(f"Cache hit for function: {func_name}")
                return cached_response

            # Create and optimize prompt
            prompt = self._create_prompt(
                func_name,
                params,
                return_type,
                complexity_score,
                existing_docstring,
                decorators,
                exceptions,
            )
            log_debug(f"Created prompt for function {func_name}: {prompt[:50]}...")

            # Validate token limits
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt, max_completion_tokens=max_tokens or self.config.max_tokens
            )

            if not is_valid:
                log_error(f"Token validation failed: {message}")
                return None

            optimized_prompt, token_usage = self.token_manager.optimize_prompt(
                prompt, max_tokens=max_tokens or self.config.max_tokens
            )
            log_debug(
                f"Optimized prompt for function {func_name}: {optimized_prompt[:50]}..."
            )

            # Make API request with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    log_debug(
                        f"Attempting API request for function {func_name}, attempt {attempt + 1}"
                    )
                    response = await self._make_api_request(
                        optimized_prompt, max_tokens, temperature, attempt
                    )

                    if response:
                        # Cache successful generation
                        await self.cache.save_docstring(
                            cache_key,
                            response,
                            tags=[
                                f"func:{func_name}",
                                f"model:{self.config.deployment_name}",
                            ],
                        )
                        log_info(
                            f"Successfully generated and cached docstring for function: {func_name}"
                        )
                        return response

                except OpenAIError as e:
                    if not await self._handle_api_error(e, attempt):
                        break
                except Exception as e:
                    log_error(
                        f"Unexpected error during API request for {func_name}: {e}"
                    )
                    if attempt == self.config.max_retries - 1:
                        raise

            log_warning(
                f"Failed to generate docstring for function {func_name} after {self.config.max_retries} attempts"
            )
            return None

        except Exception as e:
            log_error(f"Error in get_docstring for {func_name}: {e}")
            return None

    async def _make_api_request(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        """Makes an API request with proper configuration."""
        try:
            log_debug(f"Making API request with prompt: {prompt[:50]}...")
            response = await asyncio.to_thread(
                openai.Completion.create,
                model=self.config.deployment_name,  # Use your Azure deployment name
                prompt=prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                log_info("API response received successfully.")
                return {
                    "content": response["choices"][0]["text"],
                    "usage": response.get("usage", {}),
                }
            else:
                log_warning("API response is incomplete.")
                return None

        except OpenAIError as e:
            log_error(f"OpenAIError occurred during API request: {e}")
            raise
        except Exception as e:
            log_error(f"Unexpected error during API request: {e}")
            return None

    async def _handle_api_error(self, error: OpenAIError, attempt: int) -> bool:
        """Handles API errors and determines if retry is appropriate.

        Args:
            error (OpenAIError): The API error encountered.
            attempt (int): Current attempt number for retries.

        Returns:
            bool: True if should retry, False otherwise.
        """
        log_warning(f"Handling API error on attempt {attempt + 1}: {error}")
        if "rate limit" in str(error).lower():  # Check for rate limit error
            retry_after = self.config.retry_delay**attempt
            log_info(f"Rate limit hit. Waiting {retry_after}s before retry.")
            await asyncio.sleep(retry_after)
            return True
        elif attempt < self.config.max_retries - 1:
            await asyncio.sleep(self.config.retry_delay**attempt)
            return True
        return False

    def _create_prompt(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> str:
        """Creates the prompt for the API request.

        Args:
            func_name (str): Name of the function.
            params (List[Tuple[str, str]]): List of parameter names and types.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring if any.
            decorators (Optional[List[str]]): List of decorators.
            exceptions (Optional[List[str]]): List of exceptions.

        Returns:
            str: Formatted prompt for the API.
        """
        func_name = func_name.strip()
        param_details = (
            ", ".join([f"{name}: {ptype}" for name, ptype in params])
            if params
            else "None"
        )
        return_type = return_type.strip() if return_type else "Any"
        complexity_score = max(0, min(complexity_score, 100))
        existing_docstring = (
            existing_docstring.strip().replace('"', "'")
            if existing_docstring
            else "None"
        )
        decorators_info = ", ".join(decorators) if decorators else "None"
        exceptions_info = ", ".join(exceptions) if exceptions else "None"

        prompt = f"""
        Generate a JSON object with the following fields:
        {{
            "summary": "Brief function overview.",
            "changelog": "Change history or 'Initial documentation.'",
            "docstring": "Google-style docstring including a Complexity section and examples.",
            "complexity_score": {complexity_score}
        }}

        Function: {func_name}
        Parameters: {param_details}
        Returns: {return_type}
        Decorators: {decorators_info}
        Exceptions: {exceptions_info}
        Existing docstring: {existing_docstring}
        """
        log_debug(f"Created prompt for function {func_name}: {prompt[:50]}...")
        return prompt.strip()

    async def validate_connection(self) -> bool:
        """Validates the connection to Azure OpenAI service.

        Returns:
            bool: True if connection is successful.

        Raises:
            ConnectionError: If connection validation fails.
        """
        log_debug("Validating connection to Azure OpenAI service")
        try:
            response = await asyncio.wait_for(
                self._make_api_request(prompt="test", max_tokens=1, temperature=0.1, attempt=0),
                timeout=self.config.request_timeout,
            )
            log_info("Connection to Azure OpenAI API validated successfully")
            return True
        except Exception as e:
            log_error(f"Connection validation failed: {str(e)}")
            raise ConnectionError(f"Connection validation failed: {str(e)}")

    async def health_check(self) -> bool:
        """Performs a health check to verify the service is operational.

        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        log_debug("Performing health check on Azure OpenAI service")
        try:
            response = await self.get_docstring(
                func_name="test_function",
                params=[("test_param", "str")],
                return_type="None",
                complexity_score=1,
                existing_docstring="",
            )
            if response:
                log_info("Health check passed")
                return True
            else:
                log_warning("Health check failed")
                return False
        except Exception as e:
            log_error(f"Health check failed: {e}")
            return False

    def handle_rate_limits(self, retry_after: Optional[int] = None):
        """Handles rate limits by waiting for a specified duration before retrying.

        Args:
            retry_after (Optional[int]): The time to wait before retrying.

        Raises:
            TooManyRetriesError: If maximum retry attempts exceeded.
        """
        try:
            if self.current_retry >= self.config.max_retries:
                raise TooManyRetriesError(
                    f"Maximum retry attempts ({self.config.max_retries}) exceeded"
                )

            wait_time = (
                retry_after
                if retry_after
                else min(
                    self.config.retry_delay**self.current_retry,
                    self.config.request_timeout,
                )
            )
            log_info(
                f"Rate limit encountered. Waiting {wait_time}s "
                f"(attempt {self.current_retry + 1}/{self.config.max_retries})"
            )

            self.current_retry += 1
            time.sleep(wait_time)

        except TooManyRetriesError as e:
            log_error(f"Rate limit retry failed: {str(e)}")
            raise
        except Exception as e:
            log_error(f"Unexpected error in rate limit handling: {str(e)}")
            raise

    async def close(self):
        """Closes the API client and releases any resources."""
        log_debug("Closing API client")
        try:
            # Perform any necessary cleanup operations here
            log_info("API client closed successfully")
        except Exception as e:
            log_error(f"Error closing API client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        log_debug("Entering async context manager for API client")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        log_debug("Exiting async context manager for API client")
        await self.close()

    @property
    def is_ready(self) -> bool:
        """Checks if the client is ready to make API requests.

        Returns:
            bool: True if the client is properly configured.
        """
        is_ready = True  # Adjusted to reflect readiness without a client attribute
        log_debug(f"Client readiness status: {is_ready}")
        return is_ready

    def get_client_info(self) -> Dict[str, Any]:
        """Gets information about the API client configuration.

        Returns:
            Dict[str, Any]: Client configuration details.
        """
        client_info = {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "timeout": self.config.request_timeout,
            "is_ready": self.is_ready,
        }
        log_debug(f"Client configuration details: {client_info}")
        return client_info
