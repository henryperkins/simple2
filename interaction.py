import asyncio
import hashlib
import ast
import time
from typing import Dict, Tuple, Optional, List
from api_client import AzureOpenAIClient
from docs import DocStringManager
from cache import Cache
from logger import log_info, log_error, log_debug
from documentation_analyzer import DocumentationAnalyzer
from response_parser import ResponseParser
from monitoring import SystemMonitor


class InteractionHandler:
    """
    Enhanced interaction handler implementing Azure OpenAI best practices.
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5
    ):
        self.api_client = AzureOpenAIClient(endpoint=endpoint, api_key=api_key)
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        log_info("Interaction Handler initialized with batch processing capability")

    async def process_function(
        self,
        source_code: str,
        function_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code: The complete source code containing the function
            function_info: Dictionary containing function metadata

        Returns:
            Tuple[Optional[str], Optional[Dict]]: Generated docstring and metadata
        """
        async with self.semaphore:  # Implement rate limiting
            func_name = function_info.get('name', 'unknown')
            try:
                start_time = time.time()
                function_node = function_info['node']

                # Generate cache key
                function_id = self._generate_cache_key(function_node)

                # Try cache first
                cached_result = await self.cache.get_cached_docstring(function_id)
                if cached_result:
                    self.monitor.log_cache_hit(func_name)
                    return cached_result['docstring'], cached_result['metadata']

                # Check if docstring needs updating
                analyzer = DocumentationAnalyzer()
                if not analyzer.is_docstring_incomplete(function_node):
                    self.monitor.log_docstring_changes('retained', func_name)
                    return ast.get_docstring(function_node), None

                # Generate docstring
                prompt = self._generate_prompt(function_node)

                # Perform content safety check
                safety_check = await self.api_client.check_content_safety(prompt)
                if 'error' in safety_check:
                    log_error(f"Content safety check error for {func_name}: {safety_check['error']}")
                    return None, None
                elif not safety_check['safe']:
                    log_error(f"Content flagged for {func_name}: {safety_check['annotations']}")
                    return None, None

                # Get docstring from API
                response = await self.api_client.get_docstring(prompt)
                if not response:
                    return None, None

                # Process response
                docstring_data = response['content']

                # Cache the result
                await self.cache.save_docstring(
                    function_id,
                    {
                        'docstring': docstring_data['docstring'],
                        'metadata': {
                            'summary': docstring_data['summary'],
                            'complexity_score': docstring_data['complexity_score'],
                            'changelog': docstring_data.get('changelog', 'Initial documentation')
                        }
                    }
                )

                # Log metrics
                self.monitor.log_operation_complete(
                    func_name,
                    time.time() - start_time,
                    response['usage']['total_tokens']
                )

                return docstring_data['docstring'], docstring_data

            except Exception as e:
                log_error(f"Error processing function {func_name}: {str(e)}")
                return None, None

    async def process_all_functions(
        self, source_code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions in the source code with batch processing and rate limiting.

        Args:
            source_code: The complete source code to process

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation
        """
        log_debug("Starting batch processing of all functions.")
        try:
            functions = self._extract_functions(source_code)
            log_info(f"Extracted {len(functions)} functions from source code.")

            # Process functions in batches
            results = []
            for i in range(0, len(functions), self.batch_size):
                batch = functions[i: i + self.batch_size]
                log_debug(
                    f"Processing batch of functions: {[func['name'] for func in batch]}"
                )
                batch_tasks = [
                    self.process_function(source_code, func_info) for func_info in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

            # Update source code and generate documentation
            manager = DocStringManager(source_code)
            documentation_entries = []

            for function_info, (docstring, metadata) in zip(functions, results):
                if docstring:
                    manager.insert_docstring(function_info["node"], docstring)
                    if metadata:
                        documentation_entries.append(
                            {
                                "function_name": function_info["name"],
                                "complexity_score": metadata.get("complexity_score", 0),
                                "docstring": docstring,
                                "summary": metadata.get("summary", ""),
                                "changelog": metadata.get("changelog", ""),
                            }
                        )

            updated_code = manager.update_source_code(documentation_entries)
            documentation = manager.generate_markdown_documentation(
                documentation_entries
            )

            # Log final metrics
            self.monitor.log_batch_completion(len(functions))
            log_info("Batch processing completed successfully.")

            return updated_code, documentation

        except Exception as e:
            log_error(f"Error in batch processing: {str(e)}")
            return None, None

    def _generate_cache_key(self, function_node) -> str:
        """Generate a unique cache key for a function."""
        func_signature = self._get_function_signature(function_node)
        cache_key = hashlib.md5(func_signature.encode()).hexdigest()
        log_debug(f"Generated cache key: {cache_key}")
        return cache_key

    def _get_function_signature(self, function_node) -> str:
        """Generate a unique signature for a function."""
        func_name = function_node.name
        args = [arg.arg for arg in function_node.args.args]
        signature = f"{func_name}({', '.join(args)})"
        log_debug(f"Function signature: {signature}")
        return signature

    def _generate_prompt(self, function_node: ast.FunctionDef) -> str:
        """Generate a prompt for the API based on the function node."""
        prompt = f"Generate a docstring for the function {function_node.name} with parameters {', '.join(arg.arg for arg in function_node.args.args)}."
        log_debug(f"Generated prompt: {prompt}")
        return prompt

    @staticmethod
    def _extract_functions(source_code: str) -> List[Dict]:
        """Extract all functions from the source code."""
        log_debug("Extracting functions from source code.")
        try:
            tree = ast.parse(source_code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(
                        {"node": node, "name": node.name, "lineno": node.lineno}
                    )
            log_info(f"Extracted {len(functions)} functions.")
            return functions
        except Exception as e:
            log_error(f"Error extracting functions: {str(e)}")
            return []