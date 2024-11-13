import ast
from schema import DocstringSchema
from extract.base import BaseExtractor
from logger import log_info

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
            self.__init__(source_code)

        classes = []
        for node in self.walk_tree():
            if isinstance(node, ast.ClassDef):
                class_info = self.extract_class_details(node)
                classes.append(class_info)
                log_info(f"Extracted class '{node.name}' with metadata.")
        
        return classes

    def extract_class_details(self, class_node):
        """
        Extract details about a class AST node.

        Args:
            class_node (ast.ClassDef): The class node to extract details from.

        Returns:
            dict: A dictionary containing class details.
        """
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                annotations = self.extract_annotations(node)
                method_info = {
                    'name': node.name,
                    'args': annotations['args'],
                    'returns': annotations['returns'],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'docstring': self.extract_docstring(node)
                }
                methods.append(method_info)

        return {
            'name': class_node.name,
            'bases': [ast.unparse(base) for base in class_node.bases],
            'methods': methods,
            'docstring': self.extract_docstring(class_node)
        }

    def extract_class_info(self, node) -> DocstringSchema:
        """
        Convert class information to schema format.

        Args:
            node (ast.ClassDef): The class node to extract information from.

        Returns:
            DocstringSchema: The schema representing class information.
        """
        return DocstringSchema(
            description="",  # To be filled by AI
            parameters=self._extract_init_parameters(node),
            returns={"type": "None", "description": ""},
            metadata={
                "author": self._extract_author(node),
                "since_version": self._extract_version(node)
            }
        )

    def _extract_init_parameters(self, node):
        """
        Extract parameters from the __init__ method of a class.

        Args:
            node (ast.ClassDef): The class node to extract parameters from.

        Returns:
            list: A list of parameters for the __init__ method.
        """
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return self.extract_annotations(item)['args']
        return []

    def _extract_author(self, node):
        """
        Extract the author information from the class docstring.

        Args:
            node (ast.ClassDef): The class node to extract author information from.

        Returns:
            str: The author information, if available.
        """
        docstring = self.extract_docstring(node)
        # Implement logic to extract author from docstring
        return ""

    def _extract_version(self, node):
        """
        Extract the version information from the class docstring.

        Args:
            node (ast.ClassDef): The class node to extract version information from.

        Returns:
            str: The version information, if available.
        """
        docstring = self.extract_docstring(node)
        # Implement logic to extract version from docstring
        return ""