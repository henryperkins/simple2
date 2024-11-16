"""
Docstring Utilities Module

This module provides utilities for parsing and validating docstrings, including schema validation
and completeness checks. It ensures that docstrings conform to a specified structure and contain
necessary information about function parameters, return values, and exceptions.

Version: 1.0.0
Author: Development Team
"""

from typing import Dict, List, Any
from jsonschema import validate, ValidationError
import ast
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)

# JSON schema for validating docstrings
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "description": "The complete generated docstring."
        },
        "summary": {
            "type": "string",
            "description": "A concise summary of the function/class/module."
        },
        "changelog": {
            "type": "string",
            "description": "A brief description of changes (if applicable)."
        },
        "complexity_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "A score representing the code complexity (0-100)."
        }
    },
    "required": ["docstring", "summary"],
    "additionalProperties": False
}

def parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a docstring into structured sections.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        Dict[str, Any]: A dictionary with parsed sections.
    """
    if not docstring:
        return {}

    sections = {}
    current_section = 'Description'
    sections[current_section] = []

    for line in docstring.split('\n'):
        line = line.strip()
        if line.endswith(':') and line[:-1] in ['Args', 'Returns', 'Raises', 'Yields', 'Examples', 'Attributes']:
            current_section = line[:-1]
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    for key in sections:
        sections[key] = '\n'.join(sections[key]).strip()

    return sections

def validate_docstring(docstring_data: Dict[str, Any]) -> bool:
    """Validate a docstring against the JSON schema.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        validate(instance=docstring_data, schema=JSON_SCHEMA)
        return True
    except ValidationError as e:
        logger.error(f"Docstring validation error: {e.message}")
        return False

def analyze_code_element_docstring(node: ast.AST) -> List[str]:
    """Analyze the docstring of a code element for completeness.

    Args:
        node (ast.AST): The AST node representing the code element.

    Returns:
        List[str]: A list of issues found in the docstring.
    """
    issues = []
    docstring = ast.get_docstring(node)
    if not docstring:
        issues.append("Missing docstring.")
        return issues

    parsed_docstring = parse_docstring(docstring)
    if not validate_docstring(parsed_docstring):
        issues.append("Docstring does not conform to schema.")

    # Additional checks for completeness
    issues.extend(check_parameter_descriptions(parsed_docstring, node))
    issues.extend(check_return_description(parsed_docstring, node))
    issues.extend(check_exception_details(parsed_docstring, node))

    return issues

def _extract_documented_args(args_section: str) -> List[str]:
    """Extract parameter names from the Args section.

    Args:
        args_section (str): The Args section of the docstring.

    Returns:
        List[str]: A list of documented argument names.
    """
    documented_args = []
    for line in args_section.split('\n'):
        if ':' in line:
            arg_name = line.split(':')[0].strip()
            documented_args.append(arg_name)
    return documented_args

def check_parameter_descriptions(docstring_data: Dict[str, Any], function_node: ast.FunctionDef) -> List[str]:
    """Check for the presence and quality of parameter descriptions.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.

    Returns:
        List[str]: A list of issues found with parameter descriptions.
    """
    issues = []
    args_section = docstring_data.get('Args', '')
    
    if isinstance(args_section, str):
        documented_params = _extract_documented_args(args_section)
    else:
        documented_params = args_section.keys()

    function_params = [arg.arg for arg in function_node.args.args]

    for param in function_params:
        if param not in documented_params:
            issues.append(f"Parameter '{param}' is not documented.")
        elif isinstance(args_section, dict):
            description = args_section.get(param, '')
            if len(description) < 10:  # Example threshold for quality
                issues.append(f"Description for parameter '{param}' is too short.")

    return issues

def check_return_description(docstring_data: Dict[str, Any], function_node: ast.FunctionDef) -> List[str]:
    """Check for the presence and quality of return value descriptions.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.

    Returns:
        List[str]: A list of issues found with return descriptions.
    """
    issues = []
    if function_node.returns and 'Returns' not in docstring_data:
        issues.append("Missing 'Returns' section in docstring.")
    elif 'Returns' in docstring_data:
        description = docstring_data['Returns']
        if len(description) < 10:  # Example threshold for quality
            issues.append("Return description is too short.")

    return issues

def check_exception_details(docstring_data: Dict[str, Any], function_node: ast.FunctionDef) -> List[str]:
    """Check for the presence and quality of exception details.

    Args:
        docstring_data (Dict[str, Any]): The structured docstring data.
        function_node (ast.FunctionDef): The function node to analyze.

    Returns:
        List[str]: A list of issues found with exception details.
    """
    issues = []
    raises_exceptions = any(isinstance(node, ast.Raise) for node in ast.walk(function_node))
    if raises_exceptions and 'Raises' not in docstring_data:
        issues.append("Missing 'Raises' section in docstring.")
    elif 'Raises' in docstring_data:
        for exception, description in docstring_data['Raises'].items():
            if len(description) < 10:  # Example threshold for quality
                issues.append(f"Description for exception '{exception}' is too short.")

    return issues