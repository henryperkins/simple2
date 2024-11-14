from schema import DocstringSchema, JSON_SCHEMA
from jsonschema import validate, ValidationError
import ast
from typing import Optional
from logger import log_info, log_error, log_debug

class DocumentationAnalyzer:
    """
    Analyzes existing docstrings to determine if they are complete and correct.
    """

    def is_docstring_complete(self, docstring_data: Optional[DocstringSchema]) -> bool:
        """Check if docstring data is complete according to schema."""
        log_debug("Checking if docstring is complete.")
        if not docstring_data:
            log_debug("Docstring data is None or empty.")
            return False
            
        try:
            validate(instance=docstring_data, schema=JSON_SCHEMA)
            log_info("Docstring is complete and valid according to schema.")
            return True
        except ValidationError as e:
            log_error(f"Docstring validation error: {e}")
            return False

    def analyze_node(self, node: ast.AST) -> Optional[DocstringSchema]:
        """Analyze AST node and return schema-compliant docstring data."""
        log_debug(f"Analyzing AST node: {ast.dump(node)}")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                try:
                    log_debug(f"Docstring found: {docstring}")
                    docstring_data = self._parse_existing_docstring(docstring)
                    log_info(f"Successfully parsed docstring for node: {node.name}")
                    return docstring_data
                except Exception as e:
                    log_error(f"Failed to parse docstring for node '{node.name}': {e}")
                    return None
        log_debug("No docstring found or node is not a function/class definition.")
        return None

    def is_docstring_incomplete(self, function_node: ast.FunctionDef) -> bool:
        """
        Determine if the existing docstring for a function is incomplete.

        Args:
            function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
            bool: True if the docstring is incomplete, False otherwise.
        """
        log_debug(f"Checking if docstring is incomplete for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Node is not a function definition, cannot evaluate docstring.")
            return True

        existing_docstring = ast.get_docstring(function_node)
        if not existing_docstring:
            log_info(f"Function '{function_node.name}' has no docstring.")
            return True

        docstring_sections = self._parse_docstring_sections(existing_docstring)
        issues = []

        arg_issues = self._verify_args_section(function_node, docstring_sections.get('Args', ''))
        if arg_issues:
            issues.extend(arg_issues)

        return_issues = self._verify_returns_section(function_node, docstring_sections.get('Returns', ''))
        if return_issues:
            issues.extend(return_issues)

        raises_issues = self._verify_raises_section(function_node, docstring_sections.get('Raises', ''))
        if raises_issues:
            issues.extend(raises_issues)

        if issues:
            log_info(f"Function '{function_node.name}' has an incomplete docstring: {issues}")
            return True
        else:
            log_info(f"Function '{function_node.name}' has a complete docstring.")
            return False

    def is_class_docstring_incomplete(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class has an incomplete docstring.

        Args:
            class_node (ast.ClassDef): The class node to check.

        Returns:
            bool: True if the docstring is incomplete, False otherwise.
        """
        log_debug(f"Checking if class docstring is incomplete for class: {class_node.name}")
        existing_docstring = ast.get_docstring(class_node)
        if not existing_docstring:
            log_info(f"Class '{class_node.name}' has no docstring.")
            return True
        
        docstring_sections = self._parse_docstring_sections(existing_docstring)
        issues = []

        if not docstring_sections.get('Description', '').strip():
            issues.append("Missing Description section.")

        class_attributes = [node for node in class_node.body if isinstance(node, ast.Assign)]
        if class_attributes and not docstring_sections.get('Attributes', '').strip():
            issues.append("Missing Attributes section.")

        if issues:
            log_info(f"Class '{class_node.name}' has an incomplete docstring: {issues}")
            return True

        log_info(f"Class '{class_node.name}' has a complete docstring.")
        return False

    def _parse_docstring_sections(self, docstring: str) -> dict:
        """
        Parse the docstring into sections based on Google style.

        Args:
            docstring (str): The full docstring to parse.

        Returns:
            dict: A dictionary of docstring sections.
        """
        log_debug("Parsing docstring into sections.")
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
            log_debug(f"Section '{key}': {sections[key]}")
        
        return sections

    def _verify_args_section(self, function_node: ast.FunctionDef, args_section: str) -> list:
        """
        Verify that all parameters are documented in the Args section.

        Args:
            function_node (ast.FunctionDef): The function node to check.
            args_section (str): The Args section of the docstring.

        Returns:
            list: A list of issues found with the Args section.
        """
        log_debug(f"Verifying Args section for function: {function_node.name}")
        issues = []
        documented_args = self._extract_documented_args(args_section)
        function_args = [arg.arg for arg in function_node.args.args]

        for arg in function_args:
            if arg not in documented_args and arg != 'self':
                issues.append(f"Parameter '{arg}' not documented in Args section.")
        
        log_debug(f"Args section issues: {issues}")
        return issues

    def _extract_documented_args(self, args_section: str) -> list:
        """
        Extract parameter names from the Args section.

        Args:
            args_section (str): The Args section of the docstring.

        Returns:
            list: A list of documented argument names.
        """
        log_debug("Extracting documented arguments from Args section.")
        documented_args = []
        for line in args_section.split('\n'):
            if ':' in line:
                arg_name = line.split(':')[0].strip()
                documented_args.append(arg_name)
        log_debug(f"Documented arguments: {documented_args}")
        return documented_args

    def _verify_returns_section(self, function_node: ast.FunctionDef, returns_section: str) -> list:
        """
        Verify that the Returns section exists and is correctly documented.

        Args:
            function_node (ast.FunctionDef): The function node to check.
            returns_section (str): The Returns section of the docstring.

        Returns:
            list: A list of issues found with the Returns section.
        """
        log_debug(f"Verifying Returns section for function: {function_node.name}")
        issues = []
        if not returns_section and function_node.returns:
            issues.append("Missing Returns section.")
        
        log_debug(f"Returns section issues: {issues}")
        return issues

    def _verify_raises_section(self, function_node: ast.FunctionDef, raises_section: str) -> list:
        """
        Verify that the Raises section exists if the function raises exceptions.

        Args:
            function_node (ast.FunctionDef): The function node to check.
            raises_section (str): The Raises section of the docstring.

        Returns:
            list: A list of issues found with the Raises section.
        """
        log_debug(f"Verifying Raises section for function: {function_node.name}")
        issues = []
        raises_exception = any(isinstance(node, ast.Raise) for node in ast.walk(function_node))
        
        if raises_exception and not raises_section:
            issues.append("Missing Raises section for exceptions.")
        
        log_debug(f"Raises section issues: {issues}")
        return issues

    def _parse_existing_docstring(self, docstring: str) -> DocstringSchema:
        """
        Placeholder method to parse an existing docstring into a DocstringSchema.

        Args:
            docstring (str): The docstring to parse.

        Returns:
            DocstringSchema: The parsed docstring schema.
        """
        log_debug("Parsing existing docstring into DocstringSchema.")
        # Implement the actual parsing logic here
        return DocstringSchema(description=docstring, parameters=[], returns={})