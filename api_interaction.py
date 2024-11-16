"""
API Interaction Module

This module handles interactions with the Azure OpenAI API, including making requests,
handling retries, managing rate limits, and validating connections.

Version: 1.1.0
Author: Development Team
"""

import asyncio
import time
from typing import List, Tuple, Optional, Dict, Any
from openai import AzureOpenAI, APIError
from logger import log_info, log_error, log_debug
from token_management import TokenManager
from cache import Cache
from response_parser import ResponseParser
from config import AzureOpenAIConfig
from exceptions import TooManyRetriesError

class APIInteraction:
    """Handles interactions with the Azure OpenAI API.

    This class manages direct communication with the API, including request handling,
    retries, rate limiting, and response processing.

    Attributes:
        client (AzureOpenAI): The Azure OpenAI client instance.
        token_manager (TokenManager): Instance for managing tokens.
        cache (Cache): Instance for caching responses.
        parser (ResponseParser): Instance for parsing API responses.
        config (AzureOpenAIConfig): Configuration instance for Azure OpenAI.
        current_retry (int): Counter for the current retry attempt.
    """

    def __init__(self, config: AzureOpenAIConfig, token_manager: TokenManager, cache: Cache):
        """Initializes the APIInteraction with necessary components.

        Args:
            config (AzureOpenAIConfig): Configuration instance for Azure OpenAI.
            token_manager (TokenManager): Instance for managing tokens.
            cache (Cache): Instance for caching responses.
        """
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        self.token_manager = token_manager
        self.cache = cache
        self.parser = ResponseParser()
        self.config = config
        self.current_retry = 0

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
        temperature: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Generates a docstring for a function using Azure OpenAI.

        Args:
            func_name (str): Name of the function.
            params (List[Tuple[str, str]]): List of parameter names and types.
            return_type (str): Return type of the function.
            complexity_score (int): Complexity score of the function.
            existing_docstring (str): Existing docstring if any.
            decorators (Optional[List[str]]): List of decorators.
            exceptions (Optional[List[str]]): List of exceptions.
            max_tokens (Optional[int]): Maximum tokens for response.
            temperature (Optional[float]): Sampling temperature.

        Returns:
            Optional[Dict[str, Any]]: Generated docstring and metadata, or None if failed.
        """
        cache_key = f"{func_name}:{hash(str(params))}{hash(return_type)}"
        
        try:
            # Check cache first
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                log_info(f"Cache hit for function: {func_name}")
                return cached_response

            # Create and optimize prompt
            prompt = self._create_prompt(
                func_name, params, return_type, complexity_score,
                existing_docstring, decorators, exceptions
            )

            # Validate token limits
            is_valid, metrics, message = self.token_manager.validate_request(
                prompt,
                max_completion_tokens=max_tokens or self.config.max_tokens
            )
            
            if not is_valid:
                log_error(f"Token validation failed: {message}")
                return None

            optimized_prompt, token_usage = self.token_manager.optimize_prompt(
                prompt,
                max_tokens=max_tokens or self.config.max_tokens
            )

            # Make API request with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = await self._make_api_request(
                        optimized_prompt,
                        max_tokens,
                        temperature,
                        attempt
                    )

                    if response:
                        # Cache successful generation
                        await self.cache.save_docstring(
                            cache_key,
                            response,
                            tags=[
                                f"func:{func_name}",
                                f"model:{self.config.deployment_name}"
                            ]
                        )
                        return response

                except APIError as e:
                    if not await self._handle_api_error(e, attempt):
                        break
                except Exception as e:
                    log_error(f"Unexpected error: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise

            return None

        except Exception as e:
            log_error(f"Error in get_docstring for {func_name}: {e}")
            return None

    async def _make_api_request(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        attempt: int
    ) -> Optional[Dict[str, Any]]:
        """Makes an API request with proper configuration.

        Args:
            prompt (str): The prompt for the API request.
            max_tokens (Optional[int]): Maximum tokens for response.
            temperature (Optional[float]): Sampling temperature.
            attempt (int): Current attempt number for retries.

        Returns:
            Optional[Dict[str, Any]]: Parsed response data or None if failed.
        """
        try:
            response = await asyncio.wait_for(
                self._execute_api_call(prompt, max_tokens, temperature),
                timeout=self.config.request_timeout
            )

            parsed_response = self.parser.parse_json_response(
                response.choices[0].message.content
            )

            if not parsed_response:
                log_error(f"Failed to parse response (attempt {attempt + 1})")
                return None

            return {
                "content": parsed_response,
                "usage": response.usage.model_dump()
            }

        except asyncio.TimeoutError:
            log_error(f"Request timeout (attempt {attempt + 1})")
            return None

    async def _execute_api_call(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float]
    ):
        """Executes the actual API call.

        Args:
            prompt (str): The prompt for the API request.
            max_tokens (Optional[int]): Maximum tokens for response.
            temperature (Optional[float]): Sampling temperature.

        Returns:
            The API response object.
        """
        return self.client.chat.completions.create(
            model=self.config.deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a documentation expert. Generate clear, "
                              "comprehensive docstrings following Google style guide."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            functions=[{
                "name": "generate_docstring",
                "description": "Generate a structured docstring",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docstring": {"type": "string"},
                        "summary": {"type": "string"},
                        "complexity_score": {"type": "integer"},
                        "changelog": {"type": "string"},
                    },
                    "required": ["docstring", "summary"],
                },
            }],
            function_call={"name": "generate_docstring"}
        )

    async def _handle_api_error(self, error: APIError, attempt: int) -> bool:
        """Handles API errors and determines if retry is appropriate.

        Args:
            error (APIError): The API error encountered.
            attempt (int): Current attempt number for retries.

        Returns:
            bool: True if should retry, False otherwise.
        """
        if error.status_code == 429:  # Rate limit error
            retry_after = int(error.headers.get('retry-after', self.config.retry_delay ** attempt))
            log_info(f"Rate limit hit. Waiting {retry_after}s before retry.")
            await asyncio.sleep(retry_after)
            return True
        elif attempt < self.config.max_retries - 1:
            await asyncio.sleep(self.config.retry_delay ** attempt)
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
        exceptions: Optional[List[str]] = None
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
        param_details = ", ".join([f"{name}: {ptype}" for name, ptype in params]) if params else "None"
        return_type = return_type.strip() if return_type else "Any"
        complexity_score = max(0, min(complexity_score, 100))
        existing_docstring = existing_docstring.strip().replace('"', "'") if existing_docstring else "None"
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
        return prompt.strip()

    async def validate_connection(self) -> bool:
        """Validates the connection to Azure OpenAI service.

        Returns:
            bool: True if connection is successful.

        Raises:
            ConnectionError: If connection validation fails.
        """
        try:
            response = await asyncio.wait_for(
                self._execute_api_call(
                    prompt="test",
                    max_tokens=1,
                    temperature=0.1
                ),
                timeout=self.config.request_timeout
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
        try:
            response = await self.get_docstring(
                func_name="test_function",
                params=[("test_param", "str")],
                return_type="None",
                complexity_score=1,
                existing_docstring=""
            )
            return response is not None
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
                
            wait_time = retry_after if retry_after else min(
                self.config.retry_delay ** self.current_retry,
                self.config.request_timeout
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
        try:
            # Close any open connections or resources
            if hasattr(self.client, 'close'):
                await self.client.close()
            log_info("API client closed successfully")
        except Exception as e:
            log_error(f"Error closing API client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type:
            log_error(f"Error in context manager: {exc_val}")
        await self.close()

    @property
    def is_ready(self) -> bool:
        """Checks if the client is ready to make API requests.

        Returns:
            bool: True if the client is properly configured.
        """
        return bool(self.client)

    def get_client_info(self) -> Dict[str, Any]:
        """Gets information about the API client configuration.

        Returns:
            Dict[str, Any]: Client configuration details.
        """
        return {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "timeout": self.config.request_timeout,
            "is_ready": self.is_ready
        }