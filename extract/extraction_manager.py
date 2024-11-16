"""
Extraction Manager Module

This module provides a manager class for extracting metadata from Python source code.
It coordinates the extraction of class and function metadata using dedicated extractors.

Version: 1.1.0
Author: Development Team
"""

import ast
from typing import Dict, List, Any, Optional
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor
from metrics import Metrics
from logger import log_info, log_error, log_debug

class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass

class ExtractionManager:
    """
    A manager class that coordinates the extraction of metadata from Python source code.

    This class uses ClassExtractor and FunctionExtractor to analyze the Abstract Syntax Tree (AST)
    of the source code and extract detailed metadata about classes and functions.
    """

    def __init__(self):
        """Initialize the ExtractionManager without source code."""
        self.class_extractor = None
        self.function_extractor = None
        self.tree = None
        self.source_code = None

    def validate_source_code(self, source_code: str) -> bool:
        """
        Validate the provided source code to ensure it can be parsed into an AST.

        Args:
            source_code (str): The Python source code to validate.

        Returns:
            bool: True if the source code is valid, False otherwise.
        """
        try:
            self.tree = ast.parse(source_code)
            return True
        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return False

    def process_node(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Process an AST node to extract metadata.

        Args:
            node (ast.AST): The AST node to process.

        Returns:
            Optional[Dict[str, Any]]: Extracted metadata or None if processing fails.
        """
        if isinstance(node, ast.ClassDef):
            return self.class_extractor.extract_details(node)
        elif isinstance(node, ast.FunctionDef):
            return self.function_extractor.extract_details(node)
        else:
            log_error(f"Unsupported node type: {type(node).__name__}")
            return None

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from the provided source code.

        This method parses the source code into an Abstract Syntax Tree (AST) and uses
        dedicated extractors to gather detailed metadata about classes and functions.
        The metadata includes information such as function names, parameters, return types,
        and complexity metrics.

        Args:
            source_code (str): The Python source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing extracted metadata for classes and functions.
                - 'classes': A list of dictionaries, each containing metadata for a class.
                - 'functions': A list of dictionaries, each containing metadata for a function.

        Raises:
            ExtractionError: If there is an error during extraction, such as invalid syntax or failure to parse.
        """
        try:
            log_debug("Starting metadata extraction")
            if not self.validate_source_code(source_code):
                raise ExtractionError("Source code validation failed")
            self.source_code = source_code
            self.class_extractor = ClassExtractor(source_code)
            self.function_extractor = FunctionExtractor(source_code)
            classes = []
            functions = []
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    try:
                        metadata = self.process_node(node)
                        if metadata:
                            if isinstance(node, ast.ClassDef):
                                classes.append(metadata)
                                log_debug(f"Extracted class: {node.name}")
                            else:
                                functions.append(metadata)
                                log_debug(f"Extracted function: {node.name}")
                    except Exception as e:
                        log_error(f"Error extracting metadata for {type(node).__name__}: {str(e)}")
                        continue
            log_info(f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {'classes': classes, 'functions': functions}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            raise ExtractionError(f"Failed to extract metadata: {str(e)}")

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """
        Extract the docstring from an AST node.

        Args:
            node (ast.AST): The AST node to extract the docstring from.

        Returns:
            Optional[str]: The extracted docstring or None if no docstring is found.
        """
        try:
            docstring = ast.get_docstring(node)
            if docstring:
                log_info(f"Docstring extracted from node: {type(node).__name__}")
                return docstring
            else:
                log_debug(f"No docstring found in node: {type(node).__name__}")
                return None
        except Exception as e:
            log_error(f"Error extracting docstring from node: {str(e)}")
            return None