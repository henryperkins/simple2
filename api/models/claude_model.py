# claude_client.py
from typing import Dict, Any, Optional, List
from anthropic import AsyncAnthropic, AsyncMessages
from base_client import BaseAIClient
from config import ClaudeConfig
from logger import log_info, log_error, log_debug
from tenacity import retry, stop_after_attempt, wait_exponential

class ClaudeClient(BaseAIClient):
    """
    Claude AI client implementation using Anthropic's API.
    
    Supports advanced features like:
    - Streaming responses
    - JSON mode
    - Contextual embeddings
    - Robust error handling
    """

    def __init__(self, config: ClaudeConfig):
        """Initialize Claude client with configuration."""
        self.config = config
        self.client = AsyncAnthropic(api_key=config.api_key)
        log_info(f"Initialized Claude client with model: {config.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate docstring using Claude with advanced error handling and retries.
        """
        try:
            # Create system message for context
            system_message = {
                "role": "system",
                "content": "You are a technical documentation expert. Generate comprehensive and accurate function documentation."
            }

            # Create user message with function details
            user_message = {
                "role": "user",
                "content": self._create_prompt(
                    func_name=func_name,
                    params=params,
                    return_type=return_type,
                    complexity_score=complexity_score,
                    existing_docstring=existing_docstring,
                    decorators=decorators,
                    exceptions=exceptions
                )
            }

            # Request structured JSON response
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[system_message, user_message],
                response_format={"type": "json_object"}
            )

            # Parse and validate response
            if response and response.content:
                parsed_response = self._parse_response(response)
                if self._validate_response(parsed_response):
                    log_info(f"Successfully generated docstring for {func_name}")
                    return parsed_response
                else:
                    log_error(f"Invalid response format for {func_name}")
                    return None

            log_error(f"Empty response from Claude for {func_name}")
            return None

        except Exception as e:
            log_error(f"Error generating docstring with Claude: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Claude service.
        """
        try:
            start_time = time.time()
            response = await self.client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=10
            )
            latency = time.time() - start_time

            return {
                "status": "healthy" if response else "unhealthy",
                "latency": round(latency, 3),
                "model": self.config.model_name,
                "error": None
            }

        except Exception as e:
            log_error(f"Claude health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "latency": None,
                "model": self.config.model_name,
                "error": str(e)
            }

    def _create_prompt(self, **kwargs) -> str:
        """
        Create a detailed prompt for Claude.
        """
        return f"""
Generate documentation for the following Python function:

Function Name: {kwargs['func_name']}
Parameters: {', '.join(f'{name}: {type_}' for name, type_ in kwargs['params'])}
Return Type: {kwargs['return_type']}
Decorators: {', '.join(kwargs['decorators']) if kwargs['decorators'] else 'None'}
Exceptions: {', '.join(kwargs['exceptions']) if kwargs['exceptions'] else 'None'}
Complexity Score: {kwargs['complexity_score']}
Existing Docstring: {kwargs['existing_docstring'] if kwargs['existing_docstring'] else 'None'}

Please provide a JSON response with the following structure:
{{
    "docstring": "The complete docstring text",
    "summary": "A brief summary of the function's purpose",
    "complexity_analysis": "Time and space complexity analysis",
    "examples": ["Usage example 1", "Usage example 2"],
    "changelog": "Documentation changelog"
}}

Ensure the docstring follows Google style format and includes:
1. Clear description of functionality
2. Detailed parameter descriptions with types
3. Return value description
4. Usage examples
5. Time and space complexity analysis
6. Exception documentation
"""

    def _parse_response(self, response: AsyncMessages) -> Optional[Dict[str, Any]]:
        """
        Parse Claude's response into structured format.
        """
        try:
            if not response.content:
                return None

            # Parse JSON content
            content = json.loads(response.content)
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }

        except Exception as e:
            log_error(f"Error parsing Claude response: {str(e)}")
            return None

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate Claude's response format and content.
        """
        required_fields = ["docstring", "summary", "complexity_analysis", "examples"]
        
        try:
            if not response or "content" not in response:
                return False

            content = response["content"]
            return all(field in content for field in required_fields)

        except Exception as e:
            log_error(f"Error validating Claude response: {str(e)}")
            return False

    async def close(self):
        """Close the Claude client connection."""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
            log_info("Claude client closed successfully")
        except Exception as e:
            log_error(f"Error closing Claude client: {str(e)}")