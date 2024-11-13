# api_client.py
from typing import Optional, Dict, Any
from openai import AzureOpenAI, OpenAIError
from logger import log_info, log_error
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
        self.setup_client(endpoint, api_key, api_version)
        self.model = model
        self.max_retries = max_retries
        self.monitor = SystemMonitor()

    def setup_client(self, endpoint: Optional[str], api_key: Optional[str], api_version: str):
        """Initialize the Azure OpenAI client with proper error handling."""
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_KEY')
        
        if not self.endpoint or not self.api_key:
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
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
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
                
                # Parse function call response
                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                
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
                
                if attempt < self.max_retries - 1:
                    log_info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None

    async def check_content_safety(self, content: str) -> Dict[str, Any]:
        """
        Implement content safety checks using Azure OpenAI's moderation endpoint.
        """
        try:
            response = await self.client.moderations.create(input=content)
            return {
                'safe': not any(response.results[0].flagged),
                'categories': response.results[0].categories._asdict()
            }
        except OpenAIError as e:
            log_error(f"Content safety check failed: {str(e)}")
            return {'safe': False, 'error': str(e)}