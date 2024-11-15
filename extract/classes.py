"""
Class Extraction Module

This module provides functionality to extract class definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as methods, attributes, and docstrings.

Version: 1.0.0
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from logger import log_info, log_error, log_debug
from schema import DocstringSchema, DocstringParameter

class BaseExtractor(ABC):
    """Abstract base class for code extractors."""
    
    def __init__(self, source_code: Optional[str] = None):
        """Initialize the extractor with optional source code."""
        self.source_code = source_code
        self.tree = None
        if source_code:
            self.parse_source(source_code)

    def parse_source(self, source_code: str) -> None:
        """Parse source code into AST."""
        try:
            self.tree = ast.parse(source_code)
            self.source_code = source_code
        except SyntaxError as e:
            log_error(f"Failed to parse source code: {e}")
            raise

    @abstractmethod
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """Extract details from an AST node."""
        pass
    
class ClassExtractor(BaseExtractor):
    """
    Extract class definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about classes, including methods, attributes, and docstrings.
    """

    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing class metadata.
        """
        if source_code:
            log_debug("Initializing ClassExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of class definitions.")
        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                try:
                    class_info = self.extract_details(node)
                    classes.append(class_info)
                    log_info(f"Extracted class '{node.name}' with metadata.")
                except Exception as e:
                    log_error(f"Error extracting class '{node.name}': {e}")
        
        log_debug(f"Total classes extracted: {len(classes)}")
        return classes

    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a class definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            dict: A dictionary containing class details.
        """
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Expected ClassDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for class: {node.name}")
        
        try:
            details = {
                'name': node.name,
                'bases': [ast.unparse(base) for base in node.bases],
                'docstring': ast.get_docstring(node),
                'methods': [self.extract_method_details(n) for n in node.body if isinstance(n, ast.FunctionDef)],
                'attributes': self.extract_class_attributes(node),
                'lineno': node.lineno
            }
            log_debug(f"Successfully extracted details for class {node.name}")
            return details
            
        except Exception as e:
            log_error(f"Failed to extract details for class {node.name}: {e}")
            raise

    def extract_method_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract detailed information from a method node.

        Args:
            node (ast.FunctionDef): The method node to extract details from.

        Returns:
            dict: A dictionary containing method details.
        """
        return {
            'name': node.name,
            'parameters': self.extract_parameters(node),
            'return_type': self.extract_return_type(node),
            'docstring': ast.get_docstring(node),
            'decorators': self.extract_decorators(node),
            'exceptions': self.detect_exceptions(node),
            'lineno': node.lineno,
            'body_summary': self.get_body_summary(node)
        }

    def extract_parameters(self, node: ast.FunctionDef) -> List[DocstringParameter]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of DocstringParameter objects.
        """
        parameters = []
        for arg, default in zip(node.args.args, node.args.defaults):
            param_info = DocstringParameter(
                name=arg.arg,
                type=self._get_type_annotation(arg),
                description="",  # To be filled by AI or further processing
                optional=self._has_default(arg, node),
                default_value=self._get_default_value(arg, node)
            )
            parameters.append(param_info)
        return parameters

    def extract_class_attributes(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extract attributes from a class node.

        Args:
            class_node (ast.ClassDef): The class node to extract attributes from.

        Returns:
            list: A list of dictionaries containing attribute details.
        """
        attributes = []
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attr_info = {
                            'name': target.id,
                            'type': self._infer_type(node.value),
                            'lineno': node.lineno
                        }
                        attributes.append(attr_info)
        return attributes

    def _infer_type(self, value_node: ast.AST) -> str:
        """
        Infer the type of a value node.

        Args:
            value_node (ast.AST): The value node to infer type from.

        Returns:
            str: The inferred type as a string.
        """
        if isinstance(value_node, ast.Constant):
            return type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            return "List"
        elif isinstance(value_node, ast.Dict):
            return "Dict"
        elif isinstance(value_node, ast.Set):
            return "Set"
        elif isinstance(value_node, ast.Tuple):
            return "Tuple"
        else:
            return "Unknown"

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        return ast.unparse(node.returns) if node.returns else "Any"

    def extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """
        Extract decorators from a function node.

        Args:
            node (ast.FunctionDef): The function node to extract decorators from.

        Returns:
            list: A list of decorator names.
        """
        return [ast.unparse(decorator) for decorator in node.decorator_list]

    def detect_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """
        Detect exceptions that could be raised by the function.

        Args:
            node (ast.FunctionDef): The function node to analyze.

        Returns:
            list: A list of exception names that could be raised.
        """
        exceptions = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Name):
                    exceptions.add(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.add(child.exc.func.id)
        return list(exceptions)

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        return " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """
        Get type annotation as a string.

        Args:
            arg (ast.arg): The argument node to extract type annotation from.

        Returns:
            str: The type annotation.
        """
        annotation = "Any"
        if arg.annotation:
            try:
                annotation = ast.unparse(arg.annotation)
            except Exception as e:
                log_error(f"Error unparsing annotation for argument '{arg.arg}': {e}")
        log_debug(f"Type annotation for '{arg.arg}': {annotation}")
        return annotation

    def _has_default(self, arg: ast.arg, node: ast.FunctionDef) -> bool:
        """
        Check if argument has a default value.

        Args:
            arg (ast.arg): The argument node to check.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            bool: True if the argument has a default value, False otherwise.
        """
        return arg in node.args.defaults

    def _get_default_value(self, arg: ast.arg, node: ast.FunctionDef) -> Optional[str]:
        """
        Get default value as a string if it exists.

        Args:
            arg (ast.arg): The argument node to extract default value from.
            node (ast.FunctionDef): The function node containing the argument.

        Returns:
            Optional[str]: The default value or None if not present.
        """
        try:
            index = node.args.args.index(arg) - (len(node.args.args) - len(node.args.defaults))
            if index >= 0:
                return ast.unparse(node.args.defaults[index])
        except Exception as e:
            log_error(f"Error extracting default value for argument '{arg.arg}': {e}")
        return None