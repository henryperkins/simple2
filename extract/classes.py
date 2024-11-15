import ast
from schema import DocstringSchema
from logger import log_info, log_error, log_debug
from extract.base import BaseExtractor

class ClassExtractor(BaseExtractor):
    """
    Extract class definitions and their metadata from Python source code.
    """

    def extract_classes(self, source_code=None):
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
        for node in self.walk_tree():
            if isinstance(node, ast.ClassDef):
                log_debug(f"Found class definition: {node.name}")
                class_info = self.extract_class_details(node)
                classes.append(class_info)
                log_info(f"Extracted class '{node.name}' with metadata.")
        
        log_info("Completed extraction of class definitions.")
        return classes

    def extract_class_details(self, class_node):
        """
        Extract details about a class AST node.

        Args:
            class_node (ast.ClassDef): The class node to extract details from.

        Returns:
            dict: A dictionary containing class details.
        """
        log_debug(f"Extracting details for class: {class_node.name}")
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                log_debug(f"Found method definition: {node.name} in class: {class_node.name}")
                annotations = self.extract_annotations(node)
                method_info = {
                    'name': node.name,
                    'args': annotations['args'],
                    'returns': annotations['returns'],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'docstring': self.extract_docstring(node)
                }
                methods.append(method_info)
                log_info(f"Extracted method '{node.name}' in class '{class_node.name}'.")

        class_details = {
            'name': class_node.name,
            'bases': [ast.unparse(base) for base in class_node.bases],
            'methods': methods,
            'docstring': self.extract_docstring(class_node)
        }
        log_debug(f"Class details extracted for '{class_node.name}': {class_details}")
        return class_details

    def extract_class_info(self, node) -> DocstringSchema:
        """
        Convert class information to schema format.

        Args:
            node (ast.ClassDef): The class node to extract information from.

        Returns:
            DocstringSchema: The schema representing class information.
        """
        log_debug(f"Extracting class info for schema conversion: {node.name}")
        schema = DocstringSchema(
            description="",  # To be filled by AI
            parameters=self._extract_init_parameters(node),
            returns={"type": "None", "description": ""},
            metadata={
                "author": self._extract_author(node),
                "since_version": self._extract_version(node)
            }
        )
        log_debug(f"Class info extracted for schema: {schema}")
        return schema

    def _extract_init_parameters(self, node):
        """
        Extract parameters from the __init__ method of a class.

        Args:
            node (ast.ClassDef): The class node to extract parameters from.

        Returns:
            list: A list of parameters for the __init__ method.
        """
        log_debug(f"Extracting __init__ parameters for class: {node.name}")
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                parameters = self.extract_annotations(item)['args']
                log_debug(f"__init__ parameters extracted for class '{node.name}': {parameters}")
                return parameters
        log_debug(f"No __init__ method found for class '{node.name}'.")
        return []

    def _extract_author(self, node):
        """
        Extract the author information from the class docstring.

        Args:
            node (ast.ClassDef): The class node to extract author information from.

        Returns:
            str: The author information, if available.
        """
        log_debug(f"Extracting author from class docstring: {node.name}")
        docstring = self.extract_docstring(node)
        # Implement logic to extract author from docstring
        author = ""  # Placeholder for actual extraction logic
        log_debug(f"Author extracted for class '{node.name}': {author}")
        return author

    def _extract_version(self, node):
        """
        Extract the version information from the class docstring.

        Args:
            node (ast.ClassDef): The class node to extract version information from.

        Returns:
            str: The version information, if available.
        """
        log_debug(f"Extracting version from class docstring: {node.name}")
        docstring = self.extract_docstring(node)
        # Implement logic to extract version from docstring
        version = ""  # Placeholder for actual extraction logic
        log_debug(f"Version extracted for class '{node.name}': {version}")
        return version