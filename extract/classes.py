"""
Class Extraction Module

This module provides functionality to extract class definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as methods, attributes, and docstrings.

Version: 1.0.1
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor

class ClassExtractor(BaseExtractor):
    """
    Extract class definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about classes, including methods, attributes, and docstrings.
    """

    def extract_classes(self, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract all class definitions and their metadata from the source code.

        Args:
            source_code (Optional[str]): The source code to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing class metadata.
        """
        if source_code:
            log_debug("Initializing ClassExtractor with new source code.")
            self.parse_source(source_code)

        if not self.tree:
            log_error("No AST available for extraction")
            return []

        log_debug("Starting extraction of class definitions.")
        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                try:
                    log_debug(f"Found class definition: {node.name} at line {node.lineno}")
                    class_info = self.extract_details(node)
                    if class_info:
                        classes.append(class_info)
                        log_info(f"Extracted class '{node.name}' with metadata")
                except Exception as e:
                    log_error(f"Error extracting class '{getattr(node, 'name', '<unknown>')}' at line {node.lineno}: {e}")
                    continue

        log_debug(f"Total classes extracted: {len(classes)}")
        return classes

    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a class definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            Dict[str, Any]: A dictionary containing class details.
        """
        if not isinstance(node, ast.ClassDef):
            log_error(f"Expected ClassDef node, got {type(node).__name__}")
            return {}

        log_debug(f"Extracting details for class: {node.name}")

        try:
            details = self._extract_common_details(node)
            details.update({
                'bases': [ast.unparse(base) for base in node.bases],
                'methods': [self.extract_method_details(n) for n in node.body if isinstance(n, ast.FunctionDef)],
                'attributes': self.extract_class_attributes(node)
            })
            log_debug(f"Successfully extracted details for class {node.name}")
            return details

        except Exception as e:
            log_error(f"Failed to extract details for class {node.name}: {e}")
            return {}

    def extract_method_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract detailed information from a method node.

        Args:
            node (ast.FunctionDef): The method node to extract details from.

        Returns:
            dict: A dictionary containing method details.
        """
        log_debug(f"Extracting method details for method: {node.name}")
        return {
            'name': node.name,
            'parameters': self.extract_parameters(node),
            'return_type': self.extract_return_type(node),
            'docstring': self.extract_docstring(node),
            'decorators': self._extract_decorators(node),
            'exceptions': self._detect_exceptions(node),
            'lineno': node.lineno,
            'body_summary': self.get_body_summary(node)
        }

    def extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of dictionaries containing parameter details.
        """
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg),
                'optional': self._has_default(arg, node),
                'default_value': self._get_default_value(arg, node)
            }
            parameters.append(param_info)
            log_debug(f"Extracted parameter: {param_info}")
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
                        log_debug(f"Extracted attribute: {attr_info}")
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
            inferred_type = type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            inferred_type = "List"
        elif isinstance(value_node, ast.Dict):
            inferred_type = "Dict"
        elif isinstance(value_node, ast.Set):
            inferred_type = "Set"
        elif isinstance(value_node, ast.Tuple):
            inferred_type = "Tuple"
        else:
            inferred_type = "Unknown"
        
        log_debug(f"Inferred type for node: {inferred_type}")
        return inferred_type

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        return_type = ast.unparse(node.returns) if node.returns else "Any"
        log_debug(f"Extracted return type for method {node.name}: {return_type}")
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
        log_debug(f"Generated body summary for method {node.name}: {body_summary}")
        return body_summary