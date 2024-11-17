"""
Interaction Handler Module

This module manages the orchestration of docstring generation with integrated caching,
monitoring, and error handling. It processes functions and classes in batches and interacts with
the Azure OpenAI API to generate documentation.

Version: 1.2.1
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
from logger import log_info, log_error, log_debug, log_warning
from monitoring import SystemMonitor
from extract.extraction_manager import ExtractionManager
from docstring_utils import (
    analyze_code_element_docstring,
    parse_docstring,
    validate_docstring,
)
from response_parser import ResponseParser
from utils import handle_exceptions  # Import the decorator

# Load environment variables from .env file
load_dotenv()


class InteractionHandler:
    """
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling. This class processes functions and classes
    in batches and interacts with the Azure OpenAI API to generate documentation.
    """

    def __init__(
        self,
        client: Optional[AzureOpenAIClient] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5,
    ):
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            client (Optional[AzureOpenAIClient]): The Azure OpenAI client instance.
            endpoint (Optional[str]): The Azure OpenAI endpoint.
            api_key (Optional[str]): The API key for Azure OpenAI.
            cache_config (Optional[Dict]): Configuration for the cache.
            batch_size (int): Number of functions to process concurrently.
        """
        if client is None:
            if not endpoint or not api_key:
                raise ValueError(
                    "Azure OpenAI endpoint and API key must be provided if client is not supplied."
                )
            self.client = AzureOpenAIClient(endpoint=endpoint, api_key=api_key)
        else:
            self.client = client

        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        self.response_parser = ResponseParser()
        log_info("Interaction Handler initialized with batch processing capability")

    @handle_exceptions(log_error)
    async def process_all_functions(
        self, source_code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions and classes in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions and classes.")

        # Extract metadata using the centralized manager
        metadata = self.extraction_manager.extract_metadata(source_code)
        functions = metadata["functions"]
        classes = metadata.get("classes", [])

        log_info(
            f"Extracted {len(functions)} functions and {len(classes)} classes from source code."
        )

        # Process functions in batches
        function_results = []
        for i in range(0, len(functions), self.batch_size):
            batch = functions[i : i + self.batch_size]
            batch_tasks = [
                self.process_function(source_code, func_info) for func_info in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            function_results.extend(batch_results)

        # Process classes
        class_results = []
        for class_info in classes:
            class_result = await self.process_class(source_code, class_info)
            if class_result:
                class_results.append(class_result)

        # Update source code and generate documentation
        manager = DocStringManager(source_code)
        documentation_entries = []

        # Add function documentation
        for function_info, (docstring, metadata) in zip(functions, function_results):
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

        # Add class documentation
        for class_info, (docstring, metadata) in zip(classes, class_results):
            if docstring:
                manager.insert_docstring(class_info["node"], docstring)
                if metadata:
                    documentation_entries.append(
                        {
                            "class_name": class_info["name"],
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "methods": class_info.get("methods", []),
                        }
                    )

        updated_code = manager.update_source_code(documentation_entries)
        documentation = manager.generate_markdown_documentation(documentation_entries)

        # Save the generated markdown documentation using DocumentationManager
        doc_manager = DocumentationManager(output_dir="generated_docs")
        output_file = "generated_docs/documentation.md"
        doc_manager.save_documentation(documentation, output_file)

        # Log final metrics
        total_items = len(functions) + len(classes)
        self.monitor.log_batch_completion(total_items)
        log_info("Batch processing completed successfully.")

        return updated_code, documentation

    @handle_exceptions(log_error)
    async def process_function(
        self, source_code: str, function_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function.
            function_info (Dict): Metadata about the function to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        async with self.semaphore:
            func_name = function_info.get("name", "unknown")
            start_time = time.time()

            # Check cache first
            cache_key = self._generate_cache_key(function_info["node"])
            cached_response = await self.cache.get_cached_docstring(cache_key)

            if cached_response:
                # Validate cached response
                parsed_cached = self.response_parser.parse_docstring_response(
                    cached_response
                )
                if parsed_cached and validate_docstring(parsed_cached):
                    log_info(f"Using valid cached docstring for {func_name}")
                    self.monitor.log_cache_hit(func_name)
                    return parsed_cached["docstring"], cached_response
                else:
                    log_warning(
                        f"Invalid cached docstring found for {func_name}, will regenerate"
                    )
                    await self.cache.invalidate_by_tags([cache_key])

            # Check existing docstring
            existing_docstring = function_info.get("docstring")
            if existing_docstring and validate_docstring(
                parse_docstring(existing_docstring)
            ):
                log_info(f"Existing complete docstring found for {func_name}")
                return existing_docstring, None

            # Attempt to generate new docstring
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    response = await self.client.get_docstring(
                        func_name=function_info["name"],
                        params=function_info["args"],
                        return_type=function_info["returns"],
                        complexity_score=function_info.get("complexity_score", 0),
                        existing_docstring=function_info["docstring"],
                        decorators=function_info["decorators"],
                        exceptions=function_info.get("exceptions", []),
                    )

                    if not response:
                        log_error(
                            f"Failed to generate docstring for {func_name} (attempt {attempt + 1}/{max_attempts})"
                        )
                        continue

                    # Parse and validate the response
                    parsed_response = self.response_parser.parse_json_response(
                        response["content"]
                    )

                    if not parsed_response:
                        log_error(
                            f"Failed to parse response for {func_name} (attempt {attempt + 1}/{max_attempts})"
                        )
                        continue

                    # Validate the generated docstring
                    if validate_docstring(parsed_response["docstring"]):
                        # Cache successful generation
                        await self.cache.save_docstring(
                            cache_key,
                            {
                                "docstring": parsed_response["docstring"],
                                "metadata": {
                                    "timestamp": time.time(),
                                    "function_name": func_name,
                                    "summary": parsed_response.get("summary", ""),
                                    "complexity_score": parsed_response.get(
                                        "complexity_score", 0
                                    ),
                                },
                            },
                            tags=[f"func:{func_name}"],
                        )

                        # Log success metrics
                        self.monitor.log_operation_complete(
                            func_name=func_name,
                            duration=time.time() - start_time,
                            tokens=response["usage"]["total_tokens"],
                        )

                        log_info(
                            f"Successfully generated and cached docstring for {func_name}"
                        )
                        return parsed_response["docstring"], parsed_response

                    else:
                        log_warning(
                            f"Generated docstring incomplete for {func_name} (attempt {attempt + 1}/{max_attempts})"
                        )
                        self.monitor.log_docstring_issue(
                            func_name, "incomplete_generated"
                        )

                except asyncio.TimeoutError:
                    log_error(
                        f"Timeout generating docstring for {func_name} (attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                except Exception as e:
                    log_error(
                        f"Error generating docstring for {func_name} (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    await asyncio.sleep(2**attempt)

            # If all attempts fail
            log_error(
                f"Failed to generate valid docstring for {func_name} after {max_attempts} attempts"
            )
            self.monitor.log_docstring_failure(func_name, "max_attempts_exceeded")
            return None, None

    @handle_exceptions(log_error)
    async def process_class(
        self, source_code: str, class_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single class with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the class.
            class_info (Dict): Metadata about the class to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        class_name = class_info.get("name", "unknown")
        start_time = time.time()

        # Generate docstring for class
        response = await self.client.get_docstring(
            func_name=class_name,
            params=[],
            return_type="None",
            complexity_score=0,
            existing_docstring=class_info.get("docstring", ""),
            decorators=class_info.get("decorators", []),
            exceptions=[],
        )

        if response and response.get("content"):
            return response["content"].get("docstring"), response["content"]
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
