import argparse
import asyncio
import os
import shutil
import tempfile
import subprocess
from api_client import AzureOpenAIClient
from logger import log_info, log_error, log_debug
from utils import ensure_directory
from interaction import InteractionHandler

async def process_prompt(prompt: str, client: AzureOpenAIClient):
    """
    Process a single prompt to generate a response.

    Args:
        prompt (str): The input prompt for the AI model.
        client (AzureOpenAIClient): The Azure OpenAI client instance.

    Returns:
        str: The AI-generated response.
    """
    try:
        response = await client.get_docstring(prompt)
        if response:
            print(response['content']['docstring'])
        else:
            log_error("Failed to generate a response.")
    except Exception as e:
        log_error(f"Error processing prompt: {e}")

def load_source_file(file_path):
    """
    Load the Python source code from a given file path.
    
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

def save_updated_source(file_path, updated_code):
    """
    Save the updated source code to the file.
    
    Args:
        file_path (str): The path to the source file.
        updated_code (str): The updated source code to be saved.
    
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

async def process_file(file_path, args, client):
    """
    Process a single file to generate/update docstrings.
    
    Args:
        file_path (str): The path to the source file.
        args (argparse.Namespace): Parsed command-line arguments.
        client (AzureOpenAIClient): The Azure OpenAI client instance.
    """
    log_debug(f"Processing file: {file_path}")
    try:
        source_code = load_source_file(file_path)

        cache_config = {
            'host': args.redis_host,
            'port': args.redis_port,
            'db': args.redis_db,
            'password': args.redis_password,
            'default_ttl': args.cache_ttl
        }
        interaction_handler = InteractionHandler(
            endpoint=args.endpoint,
            api_key=args.api_key,
            cache_config=cache_config
        )
        updated_code, documentation = await interaction_handler.process_all_functions(source_code)

        if updated_code:
            ensure_directory(args.output_dir)  # Ensure the output directory exists
            output_file_path = os.path.join(args.output_dir, os.path.basename(file_path))
            save_updated_source(output_file_path, updated_code)
            if args.documentation_file and documentation:
                with open(args.documentation_file, 'a', encoding='utf-8') as doc_file:
                    doc_file.write(documentation)
                    log_info(f"Documentation appended to '{args.documentation_file}'")
        else:
            log_error(f"No updated code to save for '{file_path}'.")
    except Exception as e:
        log_error(f"An error occurred while processing '{file_path}': {e}")

async def run_workflow(args):
    """
    Main function to handle the workflow of generating/updating docstrings.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    source_path = args.source_path
    temp_dir = None

    log_debug(f"Starting workflow for source path: {source_path}")

    # Initialize the Azure OpenAI client
    client = AzureOpenAIClient(endpoint=args.endpoint, api_key=args.api_key)

    # Check if the source path is a Git URL
    if source_path.startswith('http://') or source_path.startswith('https://'):
        temp_dir = tempfile.mkdtemp()
        try:
            log_debug(f"Cloning repository from URL: {source_path} to temp directory: {temp_dir}")
            subprocess.run(['git', 'clone', source_path, temp_dir], check=True)
            source_path = temp_dir
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to clone repository: {e}")
            return

    try:
        if os.path.isdir(source_path):
            log_debug(f"Processing directory: {source_path}")
            # Process all Python files in the directory
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.endswith('.py'):
                        await process_file(os.path.join(root, file), args, client)
        else:
            log_debug(f"Processing single file: {source_path}")
            # Process a single file
            await process_file(source_path, args, client)
    finally:
        if temp_dir:
            log_debug(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DocStrings Workflow System with Azure OpenAI and Redis Caching')
    parser.add_argument('source_path', help='Path to the Python source file, directory, or Git repository to be processed.')
    parser.add_argument('api_key', help='Your Azure OpenAI API key.')
    parser.add_argument('--endpoint', default='https://your-azure-openai-endpoint.openai.azure.com/', help='Your Azure OpenAI endpoint.')
    parser.add_argument('--output-dir', default='output', help='Directory to save the updated source code.')
    parser.add_argument('--documentation-file', help='Path to save the generated documentation.')
    parser.add_argument('--redis-host', default='localhost', help='Redis server host.')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port.')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database number.')
    parser.add_argument('--redis-password', help='Redis server password if required.')
    parser.add_argument('--cache-ttl', type=int, default=86400, help='Default TTL for cache entries in seconds.')
    args = parser.parse_args()

    asyncio.run(run_workflow(args))