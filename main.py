"""
Main Module for Docstring Workflow System

This module serves as the entry point for the docstring generation workflow using Azure OpenAI.
It handles command-line arguments, initializes necessary components, and orchestrates the
processing of Python source files to generate and update docstrings.

Version: 1.0.1
Author: Development Team
"""

import argparse
import asynciogit
import os
import shutil
import tempfile
import subprocess
import time
import signal
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv
from interaction import InteractionHandler
from logger import log_info, log_error, log_debug, log_warning
from monitoring import SystemMonitor
from utils import ensure_directory
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from docs import MarkdownGenerator
from api_client import AzureOpenAIClient
from config import AzureOpenAIConfig
from docstring_utils import parse_docstring, validate_docstring, analyze_code_element_docstring

# Load environment variables from .env file
load_dotenv()

monitor = SystemMonitor()

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    log_info("Received shutdown signal, cleaning up...")
    raise KeyboardInterrupt

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

@contextmanager
def cleanup_context(temp_dir: Optional[str] = None):
    """Context manager for cleanup operations."""
    try:
        yield
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                log_debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                log_info("Cleanup completed successfully")
            except Exception as e:
                log_error(f"Error during cleanup: {e}")

async def initialize_client(api_key: str, endpoint: str) -> AzureOpenAIClient:
    """
    Initialize the Azure OpenAI client with proper retry logic.
    
    Args:
        api_key (str): The API key for Azure OpenAI.
        endpoint (str): The Azure OpenAI endpoint.
    
    Returns:
        AzureOpenAIClient: Configured client instance
    """
    config = AzureOpenAIConfig.from_env()
    return AzureOpenAIClient(config)

def load_source_file(file_path: str) -> str:
    """
    Load the source code from a specified file path.

    Args:
        file_path (str): The path to the source file.

    Returns:
        str: The content of the source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    try:
        log_debug(f"Attempting to load source file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
            log_info(f"Successfully loaded source code from '{file_path}'")
            return source_code
    except FileNotFoundError:
        log_error(f"File '{file_path}' not found.")
        raise
    except IOError as e:
        log_error(f"Failed to read file '{file_path}': {e}")
        raise

def save_updated_source(file_path: str, updated_code: str) -> None:
    """
    Save the updated source code to a specified file path.

    Args:
        file_path (str): The path to save the updated source code.
        updated_code (str): The updated source code to save.

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        log_debug(f"Attempting to save updated source code to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_code)
            log_info(f"Successfully saved updated source code to '{file_path}'")
    except IOError as e:
        log_error(f"Failed to save updated source code to '{file_path}': {e}")
        raise

async def process_file(file_path: str, args: argparse.Namespace, client: AzureOpenAIClient) -> None:
    """
    Process a single Python file to extract and update docstrings.

    Args:
        file_path (str): The path to the Python file to process
        args (argparse.Namespace): The command-line arguments
        client (AzureOpenAIClient): The Azure OpenAI client for generating docstrings

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If there is an error reading or writing the file
    """
    log_debug(f"Processing file: {file_path}")
    start_time = time.time()
    interaction_handler = InteractionHandler(client=client)
    failed_items = []

    try:
        source_code = load_source_file(file_path)
        
        # Extract classes and functions
        class_extractor = ClassExtractor(source_code)
        class_info = class_extractor.extract_classes()
        
        function_extractor = FunctionExtractor(source_code)
        function_info = function_extractor.extract_functions()
        
        # Process classes
        for class_data in class_info:
            try:
                issues = analyze_code_element_docstring(class_data['node'])
                if issues:
                    docstring = await client.get_docstring(
                        func_name=class_data['name'],
                        params=[(method['name'], 'Unknown') for method in class_data['methods']],
                        return_type='Unknown',
                        complexity_score=0,
                        existing_docstring=class_data['docstring'],
                        decorators=[],
                        exceptions=[]
                    )
                    if docstring and docstring['content']:
                        class_data['docstring'] = docstring['content']['docstring']
                        log_info(f"Updated docstring for class: {class_data['name']}")
                    else:
                        failed_items.append(('class', class_data['name']))
                        log_warning(f"Failed to generate docstring for class: {class_data['name']}")
            except Exception as e:
                failed_items.append(('class', class_data['name']))
                log_error(f"Error processing class {class_data['name']}: {e}")
        
        # Process functions
        for function_data in function_info:
            try:
                issues = analyze_code_element_docstring(function_data['node'])
                if issues:
                    docstring = await client.get_docstring(
                        func_name=function_data['name'],
                        params=function_data['args'],
                        return_type=function_data['return_type'],
                        complexity_score=0,
                        existing_docstring=function_data['docstring'],
                        decorators=function_data['decorators'],
                        exceptions=function_data['exceptions']
                    )
                    if docstring and docstring['content']:
                        function_data['docstring'] = docstring['content']['docstring']
                        log_info(f"Updated docstring for function: {function_data['name']}")
                    else:
                        failed_items.append(('function', function_data['name']))
                        log_warning(f"Failed to generate docstring for function: {function_data['name']}")
            except Exception as e:
                failed_items.append(('function', function_data['name']))
                log_error(f"Error processing function {function_data['name']}: {e}")

        # Generate documentation
        markdown_generator = MarkdownGenerator()
        for class_data in class_info:
            markdown_generator.add_header(class_data['name'])
            markdown_generator.add_code_block(class_data['docstring'], language="python")
        for function_data in function_info:
            markdown_generator.add_header(function_data['name'])
            markdown_generator.add_code_block(function_data['docstring'], language="python")
        
        markdown_content = markdown_generator.generate_markdown()
        
        # Save documentation and updated source
        ensure_directory(args.output_dir)
        markdown_file_path = os.path.join(args.output_dir, f"{os.path.basename(file_path)}.md")
        with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
            log_info(f"Markdown documentation saved to '{markdown_file_path}'")
        
        # Update source code with new docstrings
        updated_code = source_code  # You might need to implement the actual source code update logic
        output_file_path = os.path.join(args.output_dir, os.path.basename(file_path))
        save_updated_source(output_file_path, updated_code)
        
        # Report results
        monitor.log_operation_complete(file_path, time.time() - start_time, len(failed_items))
        
        if failed_items:
            log_warning(f"Failed to process {len(failed_items)} items in {file_path}")
            for item_type, item_name in failed_items:
                log_warning(f"Failed {item_type}: {item_name}")

    except Exception as e:
        log_error(f"Error processing file {file_path}: {e}")
        monitor.log_request(file_path, "error", time.time() - start_time, error=str(e))
        raise

async def run_workflow(args: argparse.Namespace) -> None:
    """Run the docstring generation workflow for the specified source path."""
    source_path = args.source_path
    temp_dir = None

    try:
        # Initialize client with timeout
        client = await asyncio.wait_for(initialize_client(args.api_key, args.endpoint), timeout=30)
        
        if source_path.startswith(('http://', 'https://')):
            temp_dir = tempfile.mkdtemp()
            try:
                log_debug(f"Cloning repository from URL: {source_path} to temp directory: {temp_dir}")
                # Use async subprocess
                process = await asyncio.create_subprocess_exec(
                    'git', 'clone', source_path, temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minute timeout
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, ['git', 'clone'])
                    source_path = temp_dir
                except asyncio.TimeoutError:
                    log_error("Repository cloning timed out")
                    if process:
                        process.terminate()
                    return
            except Exception as e:
                log_error(f"Failed to clone repository: {e}")
                return

        with cleanup_context(temp_dir):
            if os.path.isdir(source_path):
                log_debug(f"Processing directory: {source_path}")
                tasks = []
                for root, _, files in os.walk(source_path):
                    for file in files:
                        if file.endswith('.py'):
                            task = asyncio.create_task(
                                process_file(os.path.join(root, file), args, client)
                            )
                            tasks.append(task)
                
                # Process tasks with timeout and proper cancellation handling
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    log_error("Tasks cancelled - attempting graceful shutdown")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for tasks to complete cancellation
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
            else:
                log_debug(f"Processing single file: {source_path}")
                await process_file(source_path, args, client)

    except asyncio.CancelledError:
        log_error("Workflow cancelled - performing cleanup")
        raise
    except Exception as e:
        log_error(f"Workflow error: {str(e)}")
        raise
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DocStrings Workflow System with Azure OpenAI and Redis Caching')
    parser.add_argument('source_path', help='Path to the Python source file, directory, or Git repository to be processed.')
    parser.add_argument('api_key', help='Your Azure OpenAI API key.')
    parser.add_argument('--endpoint', default='https://your-azure-openai-endpoint.openai.azure.com/', help='Your Azure OpenAI endpoint.')
    parser.add_argument('--output-dir', default='output', help='Directory to save the updated source code and markdown documentation.')
    parser.add_argument('--documentation-file', help='Path to save the generated documentation.')
    parser.add_argument('--redis-host', default='localhost', help='Redis server host.')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port.')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database number.')
    parser.add_argument('--redis-password', help='Redis server password if required.')
    parser.add_argument('--cache-ttl', type=int, default=86400, help='Default TTL for cache entries in seconds.')
    args = parser.parse_args()

    try:
        # Run with proper task and event loop management
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_workflow(args))
        except (asyncio.CancelledError, KeyboardInterrupt):
            log_error("Program interrupted")
        except Exception as e:
            log_error(f"Program error: {e}")
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for task cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    finally:
        log_info("Program shutdown complete")