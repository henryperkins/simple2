"""
Function Extraction Module

This module provides functionality to extract function definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as parameters, return types, and docstrings.

Version: 1.0.1
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor

class FunctionExtractor(BaseExtractor):
    """
    Extract function definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about functions, including parameters, return types, and docstrings.
    """

    def extract_functions(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all function definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing function metadata.
        """
        if source_code:
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of function definitions.")
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    log_debug(f"Found function definition: {node.name} at line {node.lineno}")
                    function_info = self.extract_details(node)
                    functions.append(function_info)
                    log_info(f"Extracted function '{node.name}' with metadata.")
                except Exception as e:
                    log_error(f"Error extracting function '{node.name}': {e}")
        
        log_debug(f"Total functions extracted: {len(functions)}")
        return functions

    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a function definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            dict: A dictionary containing function details.
        """
        if not isinstance(node, ast.FunctionDef):
            log_error(f"Expected FunctionDef node, got {type(node).__name__}")
            raise ValueError(f"Expected FunctionDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for function: {node.name}")

        try:
            details = self._extract_common_details(node)
            details.update({
                'args': self.extract_parameters(node),
                'return_type': self.extract_return_type(node),
                'decorators': self._extract_decorators(node),
                'exceptions': self._detect_exceptions(node),
                'body_summary': self.get_body_summary(node)
            })
            log_debug(f"Successfully extracted details for function {node.name}")
            return details

        except Exception as e:
            log_error(f"Failed to extract details for function {node.name}: {e}")
            raise

    def extract_parameters(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of tuples containing parameter names and types.
        """
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_type_annotation(arg)
            parameters.append((param_name, param_type))
            log_debug(f"Extracted parameter: {param_name} with type: {param_type}")
        return parameters

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        return_type = ast.unparse(node.returns) if node.returns else "Any"
        log_debug(f"Extracted return type for function {node.name}: {return_type}")
        return return_type

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        body_summary = " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."
        log_debug(f"Generated body summary for function {node.name}: {body_summary}")
        return body_summary