# interaction.py
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
from metrics import MetricsError

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
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            endpoint: The Azure OpenAI endpoint.
            api_key: The API key for Azure OpenAI.
            cache_config: Configuration for the cache.
            batch_size: Number of functions to process concurrently.
        """
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
        """
        async with self.semaphore:
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

                # Extract function parameters and return type
                params = [(arg.arg, self._get_arg_type(arg)) for arg in function_node.args.args]
                return_type = self._get_return_type(function_node)
                
                # Calculate complexity score
                complexity_score = self._calculate_complexity(function_node)
                
                # Get existing docstring
                existing_docstring = ast.get_docstring(function_node) or ""

                # Extract decorators
                decorators = [ast.unparse(dec) for dec in function_node.decorator_list]

                # Extract potential exceptions
                exceptions = self._extract_exceptions(function_node)

                # Get docstring from API
                response = await self.api_client.get_docstring(
                    func_name=func_name,
                    params=params,
                    return_type=return_type,
                    complexity_score=complexity_score,
                    existing_docstring=existing_docstring,
                    decorators=decorators,
                    exceptions=exceptions
                )

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
                    func_name=func_name,
                    duration=time.time() - start_time,
                    tokens=response['usage']['total_tokens']
                )

                log_info(f"Processed function '{func_name}' successfully.")
                return docstring_data['docstring'], docstring_data

            except Exception as e:
                self.monitor.log_error_event(f"Error processing function {func_name}: {str(e)}")
                return None, None
    def _extract_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """
        Extract potential exceptions that could be raised by the function.

        Args:
            node: The function node to analyze

        Returns:
            List[str]: List of exception names that could be raised
        """
        exceptions = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Name):
                    exceptions.add(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.add(child.exc.func.id)
        return list(exceptions)

    def _get_arg_type(self, arg: ast.arg) -> str:
        """
        Extract type annotation from argument.

        Args:
            arg: The argument node

        Returns:
            str: The type annotation as a string
        """
        if arg.annotation:
            try:
                return ast.unparse(arg.annotation)
            except Exception as e:
                log_error(f"Error unparsing argument type: {e}")
                return "Any"
        return "Any"

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from function.

        Args:
            node: The function node

        Returns:
            str: The return type annotation as a string
        """
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except Exception as e:
                log_error(f"Error unparsing return type: {e}")
                return "Any"
        return "Any"

    def _calculate_complexity(self, function_node: ast.FunctionDef) -> int:
        """
        Calculate complexity score for the function.

        Args:
            function_node: The function node

        Returns:
            int: The calculated complexity score
        """
        try:
            from metrics import Metrics
            metrics = Metrics()
            complexity = metrics.calculate_cyclomatic_complexity(function_node)
            if complexity != 5:
                raise MetricsError(f"Expected 5, got {complexity}")
            return complexity
        except MetricsError as me:
            log_error(str(me))
            return complexity  # Or handle accordingly
        except Exception as e:
            log_error(f"Error calculating complexity: {e}")
            return 1  # Default complexity score

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
                log_debug(f"Processing batch of functions: {[func['name'] for func in batch]}")
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
            documentation = manager.generate_markdown_documentation(documentation_entries)

            # Log final metrics
            self.monitor.log_batch_completion(len(functions))
            log_info("Batch processing completed successfully.")

            return updated_code, documentation

        except Exception as e:
            log_error(f"Error in batch processing: {str(e)}")
            return None, None

    def _generate_cache_key(self, function_node: ast.FunctionDef) -> str:
        """Generate a unique cache key for a function."""
        func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
        return hashlib.md5(func_signature.encode()).hexdigest()

    def _get_function_signature(self, function_node: ast.FunctionDef) -> str:
        """Generate a unique signature for a function."""
        func_name = function_node.name
        args = [arg.arg for arg in function_node.args.args]
        signature = f"{func_name}({', '.join(args)})"
        log_debug(f"Function signature: {signature}")
        return signature

    def _generate_prompt(self, function_node: ast.FunctionDef) -> str:
        """Generate a prompt for the API based on the function node."""
        return (
            f"Generate a docstring for the function {function_node.name} with parameters "
            f"{', '.join(arg.arg for arg in function_node.args.args)}."
        )

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