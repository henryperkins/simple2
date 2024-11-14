# api_client.py
import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from openai import AzureOpenAI, OpenAIError
from logger import log_info, log_error, log_debug
from monitoring import SystemMonitor
from token_management import optimize_prompt


class AzureOpenAIClient:
    """
    Enhanced Azure OpenAI client with integrated monitoring, caching, and error handling.
    Implements best practices from Azure OpenAI Strategy Guide.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        model: str = "gpt-4",
        max_retries: int = 3,
    ):
        """Initialize Azure OpenAI client with configuration."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model = model
        self.max_retries = max_retries
        self.monitor = SystemMonitor()

        # Initialize Azure OpenAI client
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
        Generate an enhanced JSON schema prompt for OpenAI to create or update a docstring.

        Args:
            func_name (str): The name of the function
            params (List[Tuple[str, str]]): List of (parameter_name, parameter_type) tuples
            return_type (str): The return type of the function
            complexity_score (int): The complexity score of the function
            existing_docstring (str): The existing docstring, if any
            decorators (List[str]): List of decorators applied to the function
            exceptions (List[str]): List of exceptions the function might raise

        Returns:
            str: The enhanced JSON schema prompt
        """
        # Validate and sanitize function name
        func_name = func_name.strip()

        # Handle empty parameters
        if not params:
            param_details = "None"
        else:
            param_details = ", ".join([f"{name}: {ptype}" for name, ptype in params])

        # Handle missing return type
        return_type = return_type.strip() if return_type else "Any"

        # Validate complexity score
        complexity_score = max(
            0, min(complexity_score, 100)
        )  # Ensure score is between 0 and 100

        # Sanitize existing docstring
        existing_docstring = (
            existing_docstring.strip().replace('"', "'")
            if existing_docstring
            else "None"
        )

        # Include decorators and exceptions if available
        decorators_info = ", ".join(decorators) if decorators else "None"
        exceptions_info = ", ".join(exceptions) if exceptions else "None"

        # Construct the enhanced JSON schema prompt
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
        Generate a docstring using OpenAI with a structured JSON schema prompt.

        Args:
            func_name (str): The name of the function
            params (List[Tuple[str, str]]): List of (parameter_name, parameter_type) tuples
            return_type (str): The return type of the function
            complexity_score (int): The complexity score of the function
            existing_docstring (str): The existing docstring, if any
            decorators (Optional[List[str]]): List of decorators applied to the function
            exceptions (Optional[List[str]]): List of exceptions the function might raise
            max_tokens (int): Maximum tokens for the response
            temperature (float): Temperature for response generation

        Returns:
            Optional[Dict[str, Any]]: The generated docstring and metadata
        """
        # Generate the prompt using the JSON schema
        prompt = self.create_enhanced_json_schema_prompt(
            func_name=func_name,
            params=params,
            return_type=return_type,
            complexity_score=complexity_score,
            existing_docstring=existing_docstring,
            decorators=decorators,
            exceptions=exceptions
        )

        optimized_prompt = optimize_prompt(prompt)
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

                # Log successful request
                self.monitor.log_request(
                    endpoint="chat.completions",
                    tokens=response.usage.total_tokens,
                    response_time=time.time() - start_time,
                    status="success",
                )
                log_info("Docstring generated successfully.")

                # Parse function call response
                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                log_debug(f"Function arguments parsed: {function_args}")

                return {"content": function_args, "usage": response.usage._asdict()}

            except OpenAIError as e:
                wait_time = 2**attempt
                self.monitor.log_request(
                    endpoint="chat.completions",
                    tokens=0,
                    response_time=time.time() - start_time,
                    status="error",
                    error=str(e),
                )
                log_error(f"OpenAIError on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    log_info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None

    async def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from Azure OpenAI.

        Args:
            response (Dict[str, Any]): The response to validate

        Returns:
            bool: True if the response is valid, False otherwise
        """
        try:
            required_fields = ["docstring", "summary"]
            if not all(field in response["content"] for field in required_fields):
                log_error("Response missing required fields")
                return False

            # Validate docstring is not empty
            if not response["content"]["docstring"].strip():
                log_error("Empty docstring in response")
                return False

            # Validate summary is not empty
            if not response["content"]["summary"].strip():
                log_error("Empty summary in response")
                return False

            log_info("Response validation successful")
            return True

        except Exception as e:
            log_error(f"Error validating response: {e}")
            return False

    async def handle_rate_limits(self, retry_after: Optional[int] = None):
        """
        Handle rate limiting with exponential backoff.

        Args:
            retry_after (Optional[int]): Suggested retry time from API response
        """
        if retry_after:
            wait_time = retry_after
        else:
            wait_time = min(2 ** (self.current_retry), 60)  # Max 60 seconds

        log_info(f"Rate limit encountered. Waiting {wait_time} seconds before retry.")
        await asyncio.sleep(wait_time)

    async def close(self):
        """
        Cleanup method to properly close the client session.
        """
        try:
            # Add any cleanup operations here
            log_info("Azure OpenAI client closed successfully")
        except Exception as e:
            log_error(f"Error closing Azure OpenAI client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            log_error(f"Error in context manager: {exc_val}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        if exc_type:
            log_error(f"Error in async context manager: {exc_val}")

    @property
    def is_ready(self) -> bool:
        """
        Check if the client is properly configured and ready to use.

        Returns:
            bool: True if the client is ready, False otherwise
        """
        return bool(self.endpoint and self.api_key and self.client)

    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the current client configuration.

        Returns:
            Dict[str, Any]: Dictionary containing client configuration details
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
        Perform a health check on the Azure OpenAI service.

        Returns:
            bool: True if the service is healthy, False otherwise
        """
        try:
            # Simple test request
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
        """Test the Azure OpenAI client functionality."""
        client = AzureOpenAIClient()

        if not client.is_ready:
            log_error("Client not properly configured")
            return

        # Test health check
        is_healthy = await client.health_check()
        print(f"Service health check: {'Passed' if is_healthy else 'Failed'}")

        # Test docstring generation
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

    # Run the test
    asyncio.run(test_client())
