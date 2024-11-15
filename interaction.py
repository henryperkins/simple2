"""
Interaction Handler Module

This module manages the orchestration of docstring generation with integrated caching,
monitoring, and error handling. It processes functions in batches and interacts with
the Azure OpenAI API to generate documentation.

Version: 1.1.0
Author: Development Team
"""

import asyncio
import hashlib
import os
import time
from typing import Dict, Tuple, Optional, List
import ast
from dotenv import load_dotenv
from api_client import AzureOpenAIClient
from docs import DocStringManager
from cache import Cache
from logger import log_info, log_error, log_debug
from monitoring import SystemMonitor
from extract.extraction_manager import ExtractionManager

# Load environment variables from .env file
load_dotenv()

class InteractionHandler:
    """
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
            endpoint (Optional[str]): The Azure OpenAI endpoint.
            api_key (Optional[str]): The API key for Azure OpenAI.
            cache_config (Optional[Dict]): Configuration for the cache.
            batch_size (int): Number of functions to process concurrently.
        """
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_client = AzureOpenAIClient(endpoint=endpoint, api_key=api_key)
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        log_info("Interaction Handler initialized with batch processing capability")

    async def process_all_functions(self, source_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions.")
        try:
            # Extract metadata using the centralized manager
            metadata = self.extraction_manager.extract_metadata(source_code)
            functions = metadata['functions']
            log_info(f"Extracted {len(functions)} functions from source code.")

            # Process functions in batches
            results = []
            for i in range(0, len(functions), self.batch_size):
                batch = functions[i: i + self.batch_size]
                log_debug(f"Processing batch of functions: {[func['name'] for func in batch]}")
                batch_tasks = [self.process_function(source_code, func_info) for func_info in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

            # Update source code and generate documentation
            manager = DocStringManager(source_code)
            documentation_entries = []

            for function_info, (docstring, metadata) in zip(functions, results):
                if docstring:
                    manager.insert_docstring(function_info["node"], docstring)
                    if metadata:
                        documentation_entries.append({
                            "function_name": function_info["name"],
                            "complexity_score": metadata.get("complexity_score", 0),
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "changelog": metadata.get("changelog", ""),
                        })

            updated_code = manager.update_source_code(documentation_entries)
            documentation = manager.generate_markdown_documentation(documentation_entries)

            # Save the generated markdown documentation
            output_file = "generated_docs/documentation.md"
            manager.save_documentation(documentation, output_file)

            # Log final metrics
            self.monitor.log_batch_completion(len(functions))
            log_info("Batch processing completed successfully.")

            return updated_code, documentation

        except Exception as e:
            log_error(f"Error in batch processing: {str(e)}")
            return None, None

    async def process_function(self, source_code: str, function_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function.
            function_info (Dict): Metadata about the function to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata.
        """
        async with self.semaphore:
            func_name = function_info.get('name', 'unknown')
            try:
                start_time = time.time()
                function_node = function_info['node']

                # Use extracted metadata for docstring generation
                response = await self.api_client.get_docstring(
                    func_name=function_info['name'],
                    params=function_info['args'],
                    return_type=function_info['returns'],
                    complexity_score=function_info.get('complexity_score', 0),
                    existing_docstring=function_info['docstring'],
                    decorators=function_info['decorators'],
                    exceptions=function_info.get('exceptions', []),
                    max_tokens=4000,
                    temperature=0.2
                )

                if not response:
                    return None, None

                # Process response
                docstring_data = response['content']

                # Cache the result
                await self.cache.save_docstring(
                    self._generate_cache_key(function_node),
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

            except TimeoutError:
                log_error(f"Timeout occurred while processing function {func_name}.")
                return None, None
            except Exception as e:
                self.monitor.log_error_event(f"Error processing function {func_name}: {str(e)}")
                return None, None

    def _generate_cache_key(self, function_node: ast.FunctionDef) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_node (ast.FunctionDef): The function node to generate a key for.

        Returns:
            str: The generated cache key.
        """
        func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
        return hashlib.md5(func_signature.encode()).hexdigest()