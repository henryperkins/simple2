"""
Azure OpenAI API Client Module

This module provides a client for interacting with the Azure OpenAI API to generate
docstrings for Python functions. It handles API requests, retries, and error management,
and constructs prompts based on extracted function metadata.

Version: 1.2.0
Author: Development Team
"""

import asyncio
import json
import os
import time
from typing import Optional, Dict, Any, List, Tuple
from openai import AzureOpenAI, OpenAIError
from logger import log_info, log_error, log_debug, log_exception
from monitoring import SystemMonitor
from token_management import optimize_prompt
from cache import Cache

class TooManyRetriesError(Exception):
    """Raised when maximum retry attempts are exceeded."""
    pass
class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI to generate docstrings.

    This class manages the communication with the Azure OpenAI API, including
    constructing prompts, handling responses, and managing retries and errors.
    """
    MAX_WAIT_TIME = 60
    BASE_WAIT_TIME = 2
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = os.getenv('API_VERSION', '2024-08-01-preview'),
        model: str = os.getenv('MODEL', 'gpt-4'),
        max_retries: int = 3,
        cache: Optional[Cache] = None
    ):
        """Initialize the AzureOpenAIClient with necessary configuration."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model = model
        self.max_retries = max_retries
        self.monitor = SystemMonitor()
        self.cache = cache or Cache()
        self.current_retry = 0  # Add this line

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        log_info("Azure OpenAI client initialized successfully")

    def create_enhanced_json_schema_prompt(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: List[str] = None,
        exceptions: List[str] = None,
    ) -> str:
        """
        Create a prompt for generating a JSON schema for a function's docstring.

        Args:
            func_name (str): The name of the function.
            params (List[Tuple[str, str]]): A list of parameter names and types.
            return_type (str): The return type of the function.
            complexity_score (int): The complexity score of the function.
            existing_docstring (str): The existing docstring of the function.
            decorators (List[str]): A list of decorators applied to the function.
            exceptions (List[str]): A list of exceptions that the function may raise.

        Returns:
            str: The constructed prompt for the API.
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

    async def get_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        max_tokens: int = 500,
        temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a docstring for a function using Azure OpenAI.

        Args:
            func_name (str): The name of the function.
            params (List[Tuple[str, str]]): A list of parameter names and types.
            return_type (str): The return type of the function.
            complexity_score (int): The complexity score of the function.
            existing_docstring (str): The existing docstring of the function.
            decorators (Optional[List[str]]): A list of decorators applied to the function.
            exceptions (Optional[List[str]]): A list of exceptions that the function may raise.
            max_tokens (int): Maximum number of tokens for the response.
            temperature (float): Sampling temperature for the API.

        Returns:
            Optional[Dict[str, Any]]: The generated docstring and related metadata, or None if failed.
        """
        # Generate cache key
        cache_key = f"{func_name}:{hash(str(params))}{hash(return_type)}"
        
        # Try cache first
        cached_response = await self.cache.get_cached_docstring(cache_key)
        if cached_response:
            self.monitor.log_request(
                func_name,
                status="cache_hit",
                response_time=0,
                tokens=cached_response.get('token_usage', {}).get('total_tokens', 0)
            )
            return cached_response

        # Create and optimize prompt
        prompt = self.create_enhanced_json_schema_prompt(
            func_name=func_name,
            params=params,
            return_type=return_type,
            complexity_score=complexity_score,
            existing_docstring=existing_docstring,
            decorators=decorators,
            exceptions=exceptions
        )

        optimized_prompt = optimize_prompt(prompt, max_tokens=max_tokens)
        log_debug(f"Optimized prompt: {optimized_prompt}")
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                log_debug(f"Attempt {attempt + 1} to generate docstring.")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a documentation expert. Generate clear, "
                            "comprehensive docstrings following Google style guide.",
                        },
                        {"role": "user", "content": optimized_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    functions=[
                        {
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
                        }
                    ],
                    function_call={"name": "generate_docstring"},
                )

                self.monitor.log_request(
                    func_name,
                    status="success",
                    response_time=time.time() - start_time,
                    tokens=response.usage.total_tokens,
                    endpoint="chat.completions"
                )
                log_info("Docstring generated successfully.")

                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                log_debug(f"Function arguments parsed: {function_args}")

                # Cache the response with tags for smart invalidation
                await self.cache.save_docstring(
                    cache_key,
                    {
                        'content': function_args,
                        'usage': response.usage._asdict(),
                        'metadata': {
                            'func_name': func_name,
                            'timestamp': time.time(),
                            'model': self.model,
                            'complexity_score': complexity_score
                        }
                    },
                    tags=[
                        f"func:{func_name}",
                        f"model:{self.model}",
                        f"complexity:{complexity_score//10}0"  # Group by complexity ranges
                    ]
                )

                return {"content": function_args, "usage": response.usage._asdict()}

            except OpenAIError as e:
                wait_time = 2**attempt
                self.monitor.log_request(
                    func_name,
                    status="error",
                    response_time=time.time() - start_time,
                    tokens=0,
                    endpoint="chat.completions",
                    error=str(e)
                )
                log_error(f"OpenAIError on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    log_info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
            except json.JSONDecodeError as e:
                log_error(f"Error decoding JSON response: {e}")
                return None
            except Exception as e:
                log_error(f"Unexpected error: {e}")
                return None

    async def invalidate_cache_for_function(self, func_name: str) -> bool:
        """Invalidate all cached responses for a specific function."""
        try:
            count = await self.cache.invalidate_by_tags([f"func:{func_name}"])
            log_info(f"Invalidated {count} cache entries for function: {func_name}")
            return True
        except Exception as e:
            log_error(f"Failed to invalidate cache for function {func_name}: {e}")
            return False

    async def invalidate_cache_by_model(self, model: str) -> bool:
        """Invalidate all cached responses for a specific model."""
        try:
            count = await self.cache.invalidate_by_tags([f"model:{model}"])
            log_info(f"Invalidated {count} cache entries for model: {model}")
            return True
        except Exception as e:
            log_error(f"Failed to invalidate cache for model {model}: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            cache_stats = await self.cache.get_cache_stats()
            return {
                'cache_stats': cache_stats,
                'client_info': self.get_client_info(),
                'monitor_stats': self.monitor.get_metrics_summary()
            }
        except Exception as e:
            log_error(f"Failed to get cache stats: {e}")
            return {}

    async def _get_completion(self, func_name: str, params: List[Tuple[str, str]],
                            return_type: str, complexity_score: int,
                            existing_docstring: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Internal method for getting completion from Azure OpenAI."""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                prompt = self.create_enhanced_json_schema_prompt(
                    func_name=func_name,
                    params=params,
                    return_type=return_type,
                    complexity_score=complexity_score,
                    existing_docstring=existing_docstring,
                    decorators=kwargs.get('decorators'),
                    exceptions=kwargs.get('exceptions')
                )

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a documentation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=kwargs.get('max_tokens', 500),
                    temperature=kwargs.get('temperature', 0.2)
                )

                self.monitor.log_request(
                    func_name,
                    status="success",
                    response_time=time.time() - start_time,
                    tokens=response.usage.total_tokens,
                    endpoint="chat.completions"
                )

                return {
                    'content': response.choices[0].message.content,
                    'usage': response.usage._asdict()
                }

            except Exception as e:
                log_error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.monitor.log_request(
                        func_name,
                        status="error",
                        response_time=time.time() - start_time,
                        tokens=0,
                        endpoint="chat.completions",
                        error=str(e)
                    )
                    return None
                await asyncio.sleep(2 ** attempt)

    async def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from the API to ensure it contains required fields.

        Args:
            response (Dict[str, Any]): The response from the API.

        Returns:
            bool: True if the response is valid, False otherwise.
        """
        try:
            required_fields = ["docstring", "summary"]
            if not all(field in response["content"] for field in required_fields):
                log_error("Response missing required fields")
                return False

            if not response["content"]["docstring"].strip():
                log_error("Empty docstring in response")
                return False

            if not response["content"]["summary"].strip():
                log_error("Empty summary in response")
                return False

            log_info("Response validation successful")
            return True

        except KeyError as e:
            log_error(f"KeyError during response validation: {e}")
            return False
        except Exception as e:
            log_error(f"Error validating response: {e}")
            return False
    def reset_retry_counter(self) -> None:
        """Reset the retry counter after successful operation."""
        self.current_retry = 0
        log_debug("Retry counter reset")

    async def handle_rate_limits(self, retry_after: Optional[int] = None):
        """
        Handle rate limits by waiting for a specified duration before retrying.

        Args:
            retry_after (Optional[int]): The time to wait before retrying.
        
        Raises:
            TooManyRetriesError: If maximum retry attempts exceeded
            
        Note:
            Uses exponential backoff with a maximum wait time of 60 seconds
            when retry_after is not provided.
        """
        try:
            if self.current_retry >= self.max_retries:
                raise TooManyRetriesError(f"Maximum retry attempts ({self.max_retries}) exceeded")
                
            wait_time = retry_after if retry_after else min(
                self.BASE_WAIT_TIME ** self.current_retry, 
                self.MAX_WAIT_TIME
            )
            log_info(f"Rate limit encountered. Waiting {wait_time}s (attempt {self.current_retry + 1}/{self.max_retries})")
            
            self.current_retry += 1
            await asyncio.sleep(wait_time)
            
        except TooManyRetriesError as e:
            log_error(f"Rate limit retry failed: {str(e)}")
            raise
        except Exception as e:
            log_error(f"Unexpected error in rate limit handling: {str(e)}")
            raise

    async def close(self):
        """
        Close the Azure OpenAI client and release any resources.
        """
        try:
            log_info("Azure OpenAI client closed successfully")
        except Exception as e:
            log_error(f"Error closing Azure OpenAI client: {e}")

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            log_error(f"Error in context manager: {exc_val}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        if exc_type:
            log_error(f"Error in async context manager: {exc_val}")

    @property
    def is_ready(self) -> bool:
        """
        Check if the client is ready to make API requests.

        Returns:
            bool: True if the client is properly configured, False otherwise.
        """
        return bool(self.endpoint and self.api_key and self.client)

    def get_client_info(self) -> Dict[str, Any]:
        """
        Retrieve information about the client configuration.

        Returns:
            Dict[str, Any]: A dictionary containing client configuration details.
        """
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_version": self.api_version,
            "max_retries": self.max_retries,
            "is_ready": self.is_ready,
        }

    async def health_check(self) -> bool:
        """
        Perform a health check to verify the service is operational.

        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        try:
            response = await self.get_docstring(
                func_name="test_function",
                params=[("test_param", "str")],
                return_type="None",
                complexity_score=1,
                existing_docstring="",
            )
            return response is not None
        except Exception as e:
            log_error(f"Health check failed: {e}")
            return False

if __name__ == "__main__":

    async def test_client():
        """
        Test the AzureOpenAIClient by performing a health check and generating a docstring.
        """
        client = AzureOpenAIClient()

        if not client.is_ready:
            log_error("Client not properly configured")
            return

        is_healthy = await client.health_check()
        print(f"Service health check: {'Passed' if is_healthy else 'Failed'}")

        test_response = await client.get_docstring(
            func_name="example_function",
            params=[("param1", "str"), ("param2", "int")],
            return_type="bool",
            complexity_score=5,
            existing_docstring="",
        )

        if test_response:
            print("Test successful!")
            print(json.dumps(test_response, indent=2))
        else:
            print("Test failed!")

        await client.close()

    asyncio.run(test_client())