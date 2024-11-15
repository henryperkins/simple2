"""
Azure OpenAI API Client Module

This module provides a client for interacting with the Azure OpenAI API to generate
docstrings for Python functions. It handles API requests, retries, and error management,
and constructs prompts based on extracted function metadata.

Version: 1.2.0
Author: Development Team
"""

import json
import os
import asyncio
import time
from typing import Optional, Dict, Any, List, Tuple
from openai import AzureOpenAI, APIError
from logger import log_info, log_error, log_debug
from monitoring import SystemMonitor
from token_management import optimize_prompt
from cache import Cache
from config import AzureOpenAIConfig, default_config

class TooManyRetriesError(Exception):
    """Raised when maximum retry attempts are exceeded."""
    pass

class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI to generate docstrings.

    This class manages the communication with the Azure OpenAI API, including
    constructing prompts, handling responses, and managing retries and errors.
    """
    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        """
        Initialize the AzureOpenAIClient with necessary configuration.

        Args:
            config (Optional[AzureOpenAIConfig]): Configuration instance for Azure OpenAI.
        """
        self.config = config or default_config
        if not self.config.validate():
            raise ValueError("Invalid Azure OpenAI configuration")
        
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.endpoint
        )
        
        self.monitor = SystemMonitor()
        self.cache = Cache()
        self.current_retry = 0

        log_info("Azure OpenAI client initialized successfully")

    def validate_connection(self) -> bool:
        """
        Validate the connection to Azure OpenAI service.
        
        Returns:
            bool: True if connection is successful
            
        Raises:
            ConnectionError: If connection validation fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            log_info("Connection to Azure OpenAI API validated successfully.")
            return True
        except Exception as e:
            log_error(f"Connection validation failed: {str(e)}")
            raise ConnectionError(f"Connection validation failed: {str(e)}")

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
        cache_key = f"{func_name}:{hash(str(params))}{hash(return_type)}"
        
        # Ensure cache retrieval is awaited if it's an async operation
        cached_response = await self.cache.get_cached_docstring(cache_key)
        if cached_response:
            self.monitor.log_request(
                func_name,
                status="cache_hit",
                response_time=0,
                tokens=cached_response.get('token_usage', {}).get('total_tokens', 0)
            )
            return cached_response

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

        for attempt in range(self.config.max_retries):
            try:
                log_debug(f"Attempt {attempt + 1} to generate docstring.")
                response = self.client.chat.completions.create(
                    model=self.config.deployment_name,
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

                await self.cache.save_docstring(
                    cache_key,
                    {
                        'content': function_args,
                        'usage': response.usage.model_dump(),
                        'metadata': {
                            'func_name': func_name,
                            'timestamp': time.time(),
                            'model': self.config.deployment_name,
                            'complexity_score': complexity_score
                        }
                    },
                    tags=[
                        f"func:{func_name}",
                        f"model:{self.config.deployment_name}",
                        f"complexity:{complexity_score//10}0"
                    ]
                )

                return {"content": function_args, "usage": response.usage.model_dump()}

            except APIError as e:
                wait_time = self.config.retry_delay ** attempt
                self.monitor.log_request(
                    func_name,
                    status="error",
                    response_time=time.time() - start_time,
                    tokens=0,
                    endpoint="chat.completions",
                    error=str(e)
                )
                log_error(f"APIError on attempt {attempt + 1}: {e}")

                if attempt < self.config.max_retries - 1:
                    log_info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"Failed after {self.config.max_retries} attempts: {str(e)}")
                    return None
            except json.JSONDecodeError as e:
                log_error(f"Error decoding JSON response: {e}")
                return None
            except Exception as e:
                log_error(f"Unexpected error: {e}")
                return None
        
    def invalidate_cache_for_function(self, func_name: str) -> bool:
        """
        Invalidate all cached responses for a specific function.

        Args:
            func_name (str): The name of the function to invalidate cache for.

        Returns:
            bool: True if cache invalidation was successful, False otherwise.
        """
        try:
            count = self.cache.invalidate_by_tags([f"func:{func_name}"])
            log_info(f"Invalidated {count} cache entries for function: {func_name}")
            return True
        except Exception as e:
            log_error(f"Failed to invalidate cache for function {func_name}: {e}")
            return False

    def invalidate_cache_by_model(self, model: str) -> bool:
        """
        Invalidate all cached responses for a specific model.

        Args:
            model (str): The model name to invalidate cache for.

        Returns:
            bool: True if cache invalidation was successful, False otherwise.
        """
        try:
            count = self.cache.invalidate_by_tags([f"model:{model}"])
            log_info(f"Invalidated {count} cache entries for model: {model}")
            return True
        except Exception as e:
            log_error(f"Failed to invalidate cache for model {model}: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dict[str, Any]: A dictionary containing cache statistics and client information.
        """
        try:
            cache_stats = self.cache.get_cache_stats()
            return {
                'cache_stats': cache_stats,
                'client_info': self.get_client_info(),
                'monitor_stats': self.monitor.get_metrics_summary()
            }
        except Exception as e:
            log_error(f"Failed to get cache stats: {e}")
            return {}

    def _get_completion(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method for getting completion from Azure OpenAI.

        Args:
            func_name (str): The name of the function.
            params (List[Tuple[str, str]]): A list of parameter names and types.
            return_type (str): The return type of the function.
            complexity_score (int): The complexity score of the function.
            existing_docstring (str): The existing docstring of the function.

        Returns:
            Optional[Dict[str, Any]]: The generated completion and related metadata, or None if failed.
        """
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
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

                response = self.client.chat.completions.create(
                    model=self.config.deployment_name,
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
                    'usage': response.usage.model_dump()
                }

            except Exception as e:
                log_error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    self.monitor.log_request(
                        func_name,
                        status="error",
                        response_time=time.time() - start_time,
                        tokens=0,
                        endpoint="chat.completions",
                        error=str(e)
                    )
                    return None
                time.sleep(self.config.retry_delay ** attempt)

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from the API to ensure it contains required fields and proper content.

        Args:
            response (Dict[str, Any]): The response from the API containing content and usage information.
                Expected format:
                {
                    "content": {
                        "docstring": str,
                        "summary": str,
                        "complexity_score": int,
                        "changelog": str
                    },
                    "usage": {
                        "prompt_tokens": int,
                        "completion_tokens": int,
                        "total_tokens": int
                    }
                }

        Returns:
            bool: True if the response is valid and contains all required fields with proper content,
                False otherwise.

        Note:
            This method performs the following validations:
            1. Checks for presence of required fields
            2. Validates that docstring and summary are non-empty strings
            3. Verifies that complexity_score is a valid integer
            4. Ensures usage information is present and valid
        """
        try:
            # Check if response has the basic required structure
            if not isinstance(response, dict) or "content" not in response:
                log_error("Response missing basic structure")
                return False

            content = response["content"]

            # Validate required fields exist
            required_fields = ["docstring", "summary", "complexity_score", "changelog"]
            missing_fields = [field for field in required_fields if field not in content]
            if missing_fields:
                log_error(f"Response missing required fields: {missing_fields}")
                return False

            # Validate docstring
            if not isinstance(content["docstring"], str) or not content["docstring"].strip():
                log_error("Invalid or empty docstring")
                return False

            # Validate summary
            if not isinstance(content["summary"], str) or not content["summary"].strip():
                log_error("Invalid or empty summary")
                return False

            # Validate complexity score
            if not isinstance(content["complexity_score"], int) or not 0 <= content["complexity_score"] <= 100:
                log_error("Invalid complexity score")
                return False

            # Validate changelog
            if not isinstance(content["changelog"], str):
                log_error("Invalid changelog format")
                return False

            # Validate usage information if present
            if "usage" in response:
                usage = response["usage"]
                required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
                if not all(field in usage for field in required_usage_fields):
                    log_error("Missing usage information fields")
                    return False
                
                # Verify all token counts are non-negative integers
                if not all(isinstance(usage[field], int) and usage[field] >= 0 
                        for field in required_usage_fields):
                    log_error("Invalid token count in usage information")
                    return False

                # Verify total tokens is sum of prompt and completion tokens
                if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                    log_error("Inconsistent token counts in usage information")
                    return False

            log_info("Response validation successful")
            log_debug(f"Validated response content: {content}")
            return True

        except KeyError as e:
            log_error(f"KeyError during response validation: {e}")
            return False
        except TypeError as e:
            log_error(f"TypeError during response validation: {e}")
            return False
        except Exception as e:
            log_error(f"Unexpected error during response validation: {e}")
            return False

    def reset_retry_counter(self) -> None:
        """Reset the retry counter after successful operation."""
        self.current_retry = 0
        log_debug("Retry counter reset")

    def handle_rate_limits(self, retry_after: Optional[int] = None):
        """
        Handle rate limits by waiting for a specified duration before retrying.

        Args:
            retry_after (Optional[int]): The time to wait before retrying.
        
        Raises:
            TooManyRetriesError: If maximum retry attempts exceeded
            
        Note:
            Uses exponential backoff with a maximum wait time based on configuration
            when retry_after is not provided.
        """
        try:
            if self.current_retry >= self.config.max_retries:
                raise TooManyRetriesError(f"Maximum retry attempts ({self.config.max_retries}) exceeded")
                
            wait_time = retry_after if retry_after else min(
                self.config.retry_delay ** self.current_retry, 
                self.config.request_timeout
            )
            log_info(f"Rate limit encountered. Waiting {wait_time}s (attempt {self.current_retry + 1}/{self.config.max_retries})")
            
            self.current_retry += 1
            time.sleep(wait_time)
            
        except TooManyRetriesError as e:
            log_error(f"Rate limit retry failed: {str(e)}")
            raise
        except Exception as e:
            log_error(f"Unexpected error in rate limit handling: {str(e)}")
            raise

    def close(self):
        """
        Close the Azure OpenAI client and release any resources.
        """
        try:
            log_info("Azure OpenAI client closed successfully")
        except Exception as e:
            log_error(f"Error closing Azure OpenAI client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            log_error(f"Error in context manager: {exc_val}")
        self.close()

    @property
    def is_ready(self) -> bool:
        """
        Check if the client is ready to make API requests.

        Returns:
            bool: True if the client is properly configured, False otherwise.
        """
        return bool(self.client)

    def get_client_info(self) -> Dict[str, Any]:
        """
        Retrieve information about the client configuration.

        Returns:
            Dict[str, Any]: A dictionary containing client configuration details.
        """
        return {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "is_ready": self.is_ready,
        }

    def health_check(self) -> bool:
        """
        Perform a health check to verify the service is operational.

        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        try:
            response = self.get_docstring(
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
    def test_client():
        """
        Test the AzureOpenAIClient by performing a health check and generating a docstring.
        """
        client = AzureOpenAIClient()

        if not client.is_ready:
            log_error("Client not properly configured")
            return

        # Validate connection
        try:
            if client.validate_connection():
                log_info("Connection is valid. Proceeding with operations.")
            else:
                log_error("Connection validation failed.")
        except ConnectionError as e:
            log_error(f"Connection error: {e}")

        is_healthy = client.health_check()
        print(f"Service health check: {'Passed' if is_healthy else 'Failed'}")

        test_response = client.get_docstring(
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

        client.close()

    test_client()