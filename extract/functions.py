from schema import DocstringSchema, DocstringParameter
import ast
from typing import Optional
from extract.base import BaseExtractor
from logger import log_info

class FunctionExtractor(BaseExtractor):
    """
    A class to extract function definitions and their metadata from Python source code.
    """

    def extract_functions(self, source_code=None):
        """
        Extract all function definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing function metadata.
        """
        if source_code:
            self.__init__(source_code)
        
        functions = []
        for node in self.walk_tree():
            if isinstance(node, ast.FunctionDef):
                annotations = self.extract_annotations(node)
                function_info = {
                    'node': node,
                    'name': node.name,
                    'args': annotations['args'],
                    'returns': annotations['returns'],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'docstring': self.extract_docstring(node),
                    'comments': ast.get_docstring(node, clean=False)
                }
                functions.append(function_info)
                log_info(f"Extracted function '{node.name}' with metadata.")
        
        return functions

    def extract_function_info(self, node: ast.FunctionDef) -> DocstringSchema:
        """Extract function information into schema format."""
        parameters = [
            DocstringParameter(
                name=arg.arg,
                type=self._get_type_annotation(arg),
                description="",  # To be filled by AI
                optional=self._has_default(arg),
                default_value=self._get_default_value(arg)
            )
            for arg in node.args.args
        ]
        
        return DocstringSchema(
            description="",  # To be filled by AI
            parameters=parameters,
            returns={
                "type": self._get_return_type(node),
                "description": ""  # To be filled by AI
            }
        )

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Get type annotation as string."""
        if arg.annotation:
            return ast.unparse(arg.annotation)
        return "Any"

    def _has_default(self, arg: ast.arg) -> bool:
        """Check if argument has default value."""
        return hasattr(arg, 'default') and arg.default is not None

    def _get_default_value(self, arg: ast.arg) -> Optional[str]:
        """Get default value as string if it exists."""
        if hasattr(arg, 'default') and arg.default is not None:
            return ast.unparse(arg.default)
        return None

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Get return type annotation as string."""
        if node.returns:
            return ast.unparse(node.returns)
        return "Any"

# Example usage:
if __name__ == "__main__":
    source_code = """
@decorator
def foo(x: int, y: int) -> int:
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def bar(z):
    return z * 2
"""
    extractor = FunctionExtractor()
    functions = extractor.extract_functions(source_code)  # Pass source_code here
    for func in functions:
        print(func)