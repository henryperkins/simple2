import asyncio
import ast
import os
import shutil
import subprocess
import aiofiles
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.logger import LoggerSetup
import sentry_sdk
from utils import create_error_result, ensure_directory, filter_files, add_parent_info, is_python_file
from extract.code import extract_classes_and_functions_from_ast
from api_interaction import analyze_function_with_openai
from cache import CacheManager, create_cache_manager
logger = LoggerSetup.get_logger('files')

class RepositoryManager:
    """Manages repository operations."""

    @staticmethod
    async def clone_repo(repo_url: str, clone_dir: str) -> None:
        """Clone a GitHub repository."""
        logger.info(f'Cloning repository {repo_url}')
        try:
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir)
            result = subprocess.run(['git', 'clone', '--depth', '1', repo_url, clone_dir], capture_output=True, text=True, timeout=60, check=True)
            if result.stderr:
                logger.warning(f'Git clone warning: {result.stderr}')
            os.chmod(clone_dir, 493)
            logger.info('Repository cloned successfully')
        except Exception as e:
            logger.error(f'Failed to clone repository: {e}')
            raise

class FileProcessor:
    """Processes Python files for analysis."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def read_file_content(self, filepath: str) -> str:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f'Read {len(content)} characters from {filepath}')
            return content
        except Exception as e:
            logger.error(f'Failed to read file {filepath}: {e}')
            raise

    async def process_file(self, filepath: str, service: str) -> Dict[str, Any]:
        """Process a Python file for analysis."""
        logger.debug(f'Processing file: {filepath}')
        try:
            cache_key = f'{filepath}:{service}'
            cached_result = self.cache_manager.get_cached_response(cache_key)
            if cached_result:
                logger.debug(f'Using cached result for {filepath}')
                return cached_result
            content = await self.read_file_content(filepath)
            result = await self._analyze_file_content(content, service)
            self.cache_manager.cache_response(cache_key, result)
            return result
        except Exception as e:
            logger.error(f'Error processing file {filepath}: {e}')
            sentry_sdk.capture_exception(e)
            return create_error_result('Processing Error', str(e))

    async def _analyze_file_content(self, content: str, service: str) -> Dict[str, Any]:
        """Analyze file content and extract information."""
        logger.debug('Analyzing file content')
        try:
            tree = ast.parse(content)
            add_parent_info(tree)
            extracted_data = extract_classes_and_functions_from_ast(tree, content)
            if 'classes' not in extracted_data:
                extracted_data['classes'] = []
            tasks = [analyze_function_with_openai(func, service) for func in extracted_data.get('functions', [])]
            analyzed_functions = await asyncio.gather(*tasks, return_exceptions=True)
            for func, analysis in zip(extracted_data.get('functions', []), analyzed_functions):
                if isinstance(analysis, Exception):
                    logger.error(f'Error analyzing function {func.get('name', 'unknown')}: {analysis}')
                    sentry_sdk.capture_exception(analysis)
                else:
                    func.update(analysis)
            return extracted_data
        except Exception as e:
            logger.error(f'Error analyzing content: {e}')
            return create_error_result('Analysis Error', str(e))

class CodeAnalyzer:
    """Main class for code analysis operations."""

    def __init__(self):
        self.cache_manager = create_cache_manager()
        self.repo_manager = RepositoryManager()
        self.file_processor = FileProcessor(self.cache_manager)

    async def analyze_repository(self, repo_url: str, output_dir: str, service: str) -> Dict[str, Any]:
        """Analyze a complete repository."""
        clone_dir = 'temp_repo'
        try:
            await self.repo_manager.clone_repo(repo_url, clone_dir)
            return await self.analyze_directory(clone_dir, service)
        finally:
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir)

    async def analyze_directory(self, directory: str, service: str) -> Dict[str, Any]:
        """Analyze a local directory."""
        logger.debug(f'Analyzing directory: {directory}')
        try:
            python_files = filter_files(directory, '*.py')
            results = {}
            tasks = [self.file_processor.process_file(filepath, service) for filepath in python_files if is_python_file(filepath)]
            analyzed_files = await asyncio.gather(*tasks, return_exceptions=True)
            for filepath, result in zip(python_files, analyzed_files):
                if isinstance(result, Exception):
                    logger.error(f'Error analyzing file {filepath}: {result}')
                    sentry_sdk.capture_exception(result)
                    results[filepath] = create_error_result('Processing Error', str(result))
                else:
                    results[filepath] = result
            return results
        except Exception as e:
            logger.error(f'Error analyzing directory {directory}: {e}')
            sentry_sdk.capture_exception(e)
            raise

def create_analyzer() -> CodeAnalyzer:
    """Create a CodeAnalyzer instance."""
    return CodeAnalyzer()