# api_client.py
import asyncio
import json
import os
import time
from typing import Optional, Dict, Any
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
        model: str = "gpt-4o-2024-08-06",
        max_retries: int = 3
    ):
        log_debug("Initializing AzureOpenAIClient.")
        self.setup_client(endpoint, api_key, api_version)
        self.model = model
        self.max_retries = max_retries
        self.monitor = SystemMonitor()

    def setup_client(self, endpoint: Optional[str], api_key: Optional[str], api_version: str):
        """Initialize the Azure OpenAI client with proper error handling."""
        log_debug("Setting up Azure OpenAI client.")
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_KEY')
        
        if not self.endpoint or not self.api_key:
            log_error("Azure OpenAI endpoint and API key must be provided")
            raise ValueError("Azure OpenAI endpoint and API key must be provided")
            
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=api_version
        )
        log_info("Azure OpenAI client initialized successfully")

    async def get_docstring(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a docstring with integrated monitoring and error handling.
        Implements retry logic and token optimization from the strategy guide.
        """
        optimized_prompt = optimize_prompt(prompt)
        log_debug(f"Optimized prompt: {optimized_prompt}")
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                log_debug(f"Attempt {attempt + 1} to generate docstring.")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": "You are a documentation expert. Generate clear, "
                                   "comprehensive docstrings following Google style guide."
                    },
                    {
                        "role": "user",
                        "content": optimized_prompt
                    }],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    functions=[{
                        "name": "generate_docstring",
                        "description": "Generate a structured docstring",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docstring": {"type": "string"},
                                "summary": {"type": "string"},
                                "complexity_score": {"type": "integer"},
                                "changelog": {"type": "string"}
                            },
                            "required": ["docstring", "summary"]
                        }
                    }],
                    function_call={"name": "generate_docstring"}
                )
                
                # Log successful request
                self.monitor.log_request(
                    endpoint="chat.completions",
                    tokens=response.usage.total_tokens,
                    response_time=time.time() - start_time,
                    status="success"
                )
                log_info("Docstring generated successfully.")
                
                # Parse function call response
                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                log_debug(f"Function arguments parsed: {function_args}")
                
                return {
                    'content': function_args,
                    'usage': response.usage._asdict()
                }
                
            except OpenAIError as e:
                wait_time = 2 ** attempt
                self.monitor.log_request(
                    endpoint="chat.completions",
                    tokens=0,
                    response_time=time.time() - start_time,
                    status="error",
                    error=str(e)
                )
                log_error(f"OpenAIError on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    log_info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None

    async def check_content_safety(self, content: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Implement content safety checks with retry logic.

        Args:
            content (str): Content to check
            max_retries (int): Maximum number of retry attempts

        Returns:
            Dict[str, Any]: Safety check results
        """
        log_debug("Starting content safety check.")
        for attempt in range(max_retries):
            try:
                log_debug(f"Attempt {attempt + 1} for content safety check.")
                response = await self.client.moderations.create(input=content)
                log_info("Content safety check completed successfully.")
                return {
                    'safe': not any(response.results[0].flagged),
                    'categories': response.results[0].categories._asdict()
                }
            except Exception as e:
                log_error(f"Error during content safety check attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    log_error(f"Content safety check failed after {max_retries} attempts: {e}")
                    return {'safe': False, 'error': str(e)}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff