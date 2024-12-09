import asyncio
import json
from typing import List, Tuple, Optional, Dict, Any, Union, Mapping, Iterable, TypeAlias
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from core.logger import log_info, log_error, log_debug, log_warning
from api.token_management import TokenManager, TokenUsage
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.exceptions import TooManyRetriesError
from docstring_utils import DocstringValidator
# Assuming this is where the monitor is defined
from core.monitoring import SystemMonitor

# Define HealthStatus as a type alias for better clarity
HealthStatus = Dict[str, Union[str, Optional[float],
                               Optional[Dict[str, int]], Optional[str]]]


class APIInteraction:
    """Handles interactions with the Azure OpenAI API."""

    def __init__(
        self, config: AzureOpenAIConfig, token_manager: TokenManager, cache: Cache, monitor: SystemMonitor
    ):
        """Initializes the APIInteraction with necessary components."""
        log_debug("Initializing APIInteraction with Azure OpenAI configuration")

        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )

        self.token_manager = token_manager
        self.cache = cache
        self.config = config
        self.monitor = monitor
        self.current_retry = 0
        self.validator = DocstringValidator()  # Add validator instance
        log_info("APIInteraction initialized successfully.")

    def _log_token_usage(self, func_name: str, token_usage: TokenUsage, response_time: float = 0.0,
                         error: Optional[str] = None):
        """Logs token usage for a function."""
        log_info(
            f"Token usage for {func_name}: {token_usage.total_tokens} tokens used, response time: {response_time}s")
        if error:
            log_error(
                f"Error during token usage logging for {func_name}: {error}")

    def _get_docstring_function(self) -> Dict[str, Any]:
        """Enhanced function schema for docstring generation."""
        return {
            "name": "generate_docstring",
            "description": "Generate a structured docstring for a function",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief description of the function"
                    },
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "optional": {"type": "boolean"},
                                "default": {"type": ["string", "number", "boolean", "null"]}
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "description"]
                    },
                    "raises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["exception", "description"]
                        }
                    },
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["code"]
                        }
                    }
                },
                "required": ["summary", "parameters", "returns"]
            }
        }

    async def get_docstring(
            self,
            func_name: str,
            params: List[Tuple[str, str]],
            return_type: str,
            complexity_score: int,
            existing_docstring: str,
            decorators: Optional[List[str]] = None,
            exceptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generates a docstring for a function using Azure OpenAI with token management."""
        log_debug(f"Generating docstring for function: {func_name}")

        # Convert lists to tuples for hashable contexts
        cache_key = f"docstring:{func_name}:{hash(tuple(params))}:{hash(return_type)}"

        try:
            # Check cache first
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                log_info(f"Cache hit for function: {func_name}")
                # Check if response needs to be deserialized
                if isinstance(cached_response, str):
                    return json.loads(cached_response)
                return cached_response

            # Create messages
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "You are a technical documentation expert. Generate comprehensive and accurate function documentation."
                },
                {
                    "role": "user",
                    "content": self._create_prompt(
                        func_name, params, return_type, complexity_score,
                        existing_docstring, decorators, exceptions
                    )
                }
            ]

            # Validate token limits before making request
            prompt_text = json.dumps(messages)
            is_valid, metrics, validation_message = self.token_manager.validate_request(
                prompt_text)

            if not is_valid:
                log_error(
                    f"Token validation failed for {func_name}: {validation_message}")
                return None

            # Optimize prompt if needed
            optimized_messages = await self._optimize_prompt(messages, {k: int(v) for k, v in metrics.items()})
            if optimized_messages is None:
                log_error(f"Failed to optimize prompt for {func_name}")
                return None

            # Make API request with retry logic and token tracking
            for attempt in range(self.config.max_retries):
                try:
                    response = await self._make_api_request(optimized_messages, attempt)
                    if response:
                        return await self._process_response(response, {"function": func_name})
                except TooManyRetriesError:
                    log_error(f"Max retries exceeded for {func_name}")
                    raise
                except Exception as e:
                    log_error(f"Error in get_docstring for {func_name}: {e}")
                    await self.handle_rate_limits()

            log_warning(
                f"Failed to generate docstring for {func_name} after {self.config.max_retries} attempts")
            return None

        except Exception as e:
            log_error(f"Error in get_docstring for {func_name}: {e}")
            return None

    async def _optimize_prompt(
            self,
            messages: List[ChatCompletionMessageParam],
            metrics: Mapping[str, int]
    ) -> Optional[List[ChatCompletionMessageParam]]:
        """Optimizes the prompt to fit within token limits."""
        try:
            optimized_messages, token_usage = self.token_manager.optimize_prompt(
                json.dumps(messages),
                max_tokens=self.config.max_tokens,
                preserve_sections=['parameters', 'returns']
            )

            log_info(f"Optimized prompt tokens: {token_usage.prompt_tokens}")
            return json.loads(optimized_messages)

        except Exception as e:
            log_error(f"Error optimizing prompt: {e}")
            return None

    async def _make_api_request(
            self,
            messages: List[ChatCompletionMessageParam],
            attempt: int
    ) -> Optional[ChatCompletion]:
        """Makes an API request with token tracking."""
        try:
            log_debug(f"Making API request, attempt {attempt + 1}")

            # Pre-request token check
            estimated_tokens = self.token_manager.estimate_tokens(
                json.dumps(messages))
            if estimated_tokens > self.config.max_tokens:
                log_error(
                    f"Estimated tokens ({estimated_tokens}) exceed max tokens ({self.config.max_tokens})")
                return None

            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Track token usage
            if response.usage:
                self.token_manager.track_request(
                    response.usage.prompt_tokens, response.usage.completion_tokens)

            log_debug("API request successful")
            return response

        except Exception as e:
            log_error(f"Error making API request: {e}")
            return None

    async def _process_response(self, response: ChatCompletion, error_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate API response."""
        try:
            if not response.choices:
                return None

            content = response.choices[0].message.content
            if not content:
                return None

            # Remove code block markers if present
            if content.startswith('```') and content.endswith('```'):
                content = content.strip('```')
                if content.startswith('json'):
                    content = content[len('json'):].strip()

            parsed_args = json.loads(content)

            # Ensure docstring field exists
            if 'summary' in parsed_args and 'docstring' not in parsed_args:
                # Convert summary to docstring if missing
                parsed_args['docstring'] = parsed_args['summary']

            # Validate response content
            is_valid, validation_errors = self.validator.validate_docstring(
                parsed_args)

            if not is_valid:
                log_error(
                    f"Response validation failed for {error_context['function']}: "
                    f"{validation_errors}"
                )
                return None

            return {
                "content": parsed_args,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as e:
            log_error(f"Response processing error: {e}")
            log_debug(f"Raw API response: {response}")
            error_context['last_error'] = str(e)
            return None

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
        exceptions_list = ""
        if exceptions:
            exceptions_list = "\n".join(f"- {ex}" for ex in exceptions)

        # Adjust return type for functions that don't return a value
        if return_type.lower() in {"none", "null"}:
            return_type = "NoneType"

        return f"""
    Generate a docstring in JSON format that must include the following fields:

    Function Details:
    Name: {func_name}
    Parameters: {', '.join(f'{name}: {type_}' for name, type_ in params)}
    Return Type: {return_type}
    Decorators: {', '.join(decorators) if decorators else 'None'}
    Exceptions: {exceptions_list if exceptions_list else 'None'}
    Complexity Score: {complexity_score}
    Existing Docstring: {existing_docstring if existing_docstring else 'None'}

    Required JSON Format:
    {{
        "docstring": "Complete docstring text that fully describes the function",
        "summary": "Brief summary of function purpose",
        "parameters": [
            {{
                "name": "parameter_name",
                "type": "parameter_type",
                "description": "Description of the parameter"
            }},
            ...
        ],
        "returns": {{
            "type": "{return_type}",
            "description": "Description of the return value"
        }},
        "raises": [
            {{
                "exception": "ExceptionName",
                "description": "Description of when this exception is raised"
            }},
            ...
        ],
        "examples": [
            {{
                "code": "Example code snippet",
                "description": "Description of what the example demonstrates"
            }},
            ...
        ]
    }}

    Ensure the 'summary' field is included and the 'returns' field is correctly formatted.
    """

    async def validate_connection(self) -> bool:
        """Validates the connection to Azure OpenAI service."""
        log_debug("Validating connection to Azure OpenAI service")
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "system", "content": "ping"}],
                max_tokens=1
            )
            return response is not None
        except Exception as e:
            log_error(f"Connection validation error: {e}")
            return False

    async def health_check(self) -> HealthStatus:
        """
        Performs a health check of the API service.

        Returns:
            HealthStatus: Dictionary containing health check results including:
                - status: "healthy" or "unhealthy"
                - latency: Response time in seconds (if healthy)
                - token_usage: Token usage statistics (if healthy)
                - error: Error message (if unhealthy)
        """
        health_status: HealthStatus = {
            "status": "unhealthy",
            "latency": None,
            "error": None,
            "token_usage": None
        }

        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[{"role": "system", "content": "ping"}],
                max_tokens=1
            )
            latency = time.time() - start_time

            if response:
                health_status.update({
                    "status": "healthy",
                    "latency": latency,
                    "token_usage": self.get_token_usage_stats()
                })
            else:
                health_status["error"] = "No response from API"

        except Exception as e:
            health_status["error"] = str(e)

        return health_status

    async def close(self):
        """Closes the API client and releases resources."""
        log_debug("Closing API client")
        try:
            await self.client.close()
        except Exception as e:
            log_error(f"Error closing API client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def get_client_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Gets information about the API client configuration.

        Returns:
            Dict[str, Union[str, int, float]]: Dictionary containing client configuration
            with string and numeric values.
        """
        return {
            "endpoint": self.config.endpoint,
            "model": self.config.deployment_name,
            "api_version": self.config.api_version,
            "max_retries": self.config.max_retries,
            "timeout": self.config.request_timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

    async def handle_rate_limits(self, retry_after: Optional[int] = None):
        """Handles rate limits by implementing exponential backoff."""
        if self.current_retry >= self.config.max_retries:
            raise TooManyRetriesError(
                f"Maximum retry attempts ({self.config.max_retries}) exceeded"
            )

        wait_time = retry_after or min(
            self.config.retry_delay * (2 ** self.current_retry),
            self.config.request_timeout
        )

        log_info(
            f"Rate limit encountered. Waiting {wait_time}s "
            f"(attempt {self.current_retry + 1}/{self.config.max_retries})"
        )

        self.current_retry += 1
        await asyncio.sleep(wait_time)

    def get_token_usage_stats(self) -> Dict[str, int]:
        """Returns current token usage statistics."""
        return {
            "total_prompt_tokens": self.token_manager.total_prompt_tokens,
            "total_completion_tokens": self.token_manager.total_completion_tokens
        }

    async def batch_process(
        self,
        functions: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process multiple functions in batches with rate limiting.

        Args:
            functions: List of function metadata dictionaries, each containing:
                - func_name: Name of the function
                - params: List of parameter tuples (name, type)
                - return_type: Function return type
                - complexity_score: Function complexity score
                - existing_docstring: Existing docstring if any
                - decorators: Optional list of decorators
                - exceptions: Optional list of exceptions
            batch_size: Number of functions to process concurrently.
                       Defaults to config setting if not specified.

        Returns:
            List[Optional[Dict[str, Any]]]: List of generated docstrings and metadata
        """
        batch_size = batch_size or getattr(self.config, 'batch_size', 10)
        results = []

        for i in range(0, len(functions), batch_size):
            batch = functions[i:i + batch_size]
            try:
                log_debug(
                    f"Processing batch {i//batch_size + 1}, size {len(batch)}")

                batch_results = await asyncio.gather(*[
                    self.get_docstring(**func) for func in batch
                ], return_exceptions=True)

                for func, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        log_error(
                            f"Error processing {func.get('func_name', 'unknown')}: {result}")
                        results.append(None)
                    else:
                        results.append(result)

            except Exception as e:
                log_error(f"Batch processing error: {str(e)}")
                results.extend([None] * len(batch))

        return results
