import argparse
import asyncio
import os
import shutil
import tempfile
import subprocess
from interaction import InteractionHandler
from logger import log_info, log_error, log_debug
from utils import ensure_directory
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor  # Import FunctionExtractor
from docs import MarkdownGenerator
from api_client import AzureOpenAIClient

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
    Process a single file to generate/update docstrings and markdown documentation.
    
    Args:
        file_path (str): The path to the source file.
        args (argparse.Namespace): Parsed command-line arguments.
        client (AzureOpenAIClient): The client to interact with the language model.
    """
    log_debug(f"Processing file: {file_path}")
    try:
        source_code = load_source_file(file_path)
        
        # Extract class information
        class_extractor = ClassExtractor(source_code)
        class_info = class_extractor.extract_classes()
        
        # Extract function information
        function_extractor = FunctionExtractor(source_code)
        function_info = function_extractor.extract_functions()
        
        # Generate docstrings using the language model for classes
        for class_data in class_info:
            prompt = f"Generate a docstring for the class {class_data['name']} with methods {class_data['methods']}."
            docstring = await client.get_docstring(
                func_name=class_data['name'],
                params=[(method['name'], 'Unknown') for method in class_data['methods']],  # Placeholder for actual types
                return_type='Unknown',  # Placeholder for actual return type
                complexity_score=0,  # Placeholder for actual complexity score
                existing_docstring=class_data['docstring'],
                decorators=[],
                exceptions=[]
            )
            class_data['docstring'] = docstring['content']['docstring'] if docstring else class_data['docstring']
        
        # Generate docstrings using the language model for functions
        for function_data in function_info:
            prompt = f"Generate a docstring for the function {function_data['name']} with arguments {function_data['args']}."
            docstring = await client.get_docstring(
                func_name=function_data['name'],
                params=function_data['args'],
                return_type=function_data['returns'],
                complexity_score=0,  # Placeholder for actual complexity score
                existing_docstring=function_data['docstring'],
                decorators=function_data['decorators'],
                exceptions=[]
            )
            function_data['docstring'] = docstring['content']['docstring'] if docstring else function_data['docstring']
        
        # Generate markdown documentation
        markdown_generator = MarkdownGenerator()
        for class_data in class_info:
            markdown_generator.add_header(class_data['name'])
            markdown_generator.add_code_block(class_data['docstring'], language="python")
        for function_data in function_info:
            markdown_generator.add_header(function_data['name'])
            markdown_generator.add_code_block(function_data['docstring'], language="python")
        
        markdown_content = markdown_generator.generate_markdown()
        
        # Save markdown documentation
        ensure_directory(args.output_dir)
        markdown_file_path = os.path.join(args.output_dir, f"{os.path.basename(file_path)}.md")
        with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
            log_info(f"Markdown documentation saved to '{markdown_file_path}'")
        
        # Save updated source code with docstrings
        updated_code = source_code  # Placeholder for actual updated code logic
        output_file_path = os.path.join(args.output_dir, os.path.basename(file_path))
        save_updated_source(output_file_path, updated_code)
        
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

    # Initialize the AzureOpenAIClient
    client = AzureOpenAIClient(api_key=args.api_key, endpoint=args.endpoint)

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
    parser.add_argument('--output-dir', default='output', help='Directory to save the updated source code and markdown documentation.')
    parser.add_argument('--documentation-file', help='Path to save the generated documentation.')
    parser.add_argument('--redis-host', default='localhost', help='Redis server host.')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis server port.')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database number.')
    parser.add_argument('--redis-password', help='Redis server password if required.')
    parser.add_argument('--cache-ttl', type=int, default=86400, help='Default TTL for cache entries in seconds.')
    args = parser.parse_args()

    asyncio.run(run_workflow(args))
