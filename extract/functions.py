from schema import DocstringSchema, DocstringParameter
import ast
from typing import Optional
from extract.base import BaseExtractor
from logger import log_info, log_error, log_debug

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
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)
        
        functions = []
        for node in self.walk_tree():
            if isinstance(node, ast.FunctionDef):
                try:
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
                except Exception as e:
                    log_error(f"Error extracting function '{node.name}': {e}")
        
        log_debug(f"Total functions extracted: {len(functions)}")
        return functions

    def extract_function_info(self, node: ast.FunctionDef) -> DocstringSchema:
        """Extract function information into schema format."""
        log_debug(f"Extracting function info for: {node.name}")
        try:
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
            
            schema = DocstringSchema(
                description="",  # To be filled by AI
                parameters=parameters,
                returns={
                    "type": self._get_return_type(node),
                    "description": ""  # To be filled by AI
                }
            )
            log_info(f"Extracted schema for function '{node.name}'.")
            return schema
        except Exception as e:
            log_error(f"Error extracting function info for '{node.name}': {e}")
            return DocstringSchema(description="", parameters=[], returns={"type": "Any", "description": ""})

    def _get_type_annotation(self, arg: ast.arg) -> str:
        """Get type annotation as string."""
        annotation = "Any"
        if arg.annotation:
            try:
                annotation = ast.unparse(arg.annotation)
            except Exception as e:
                log_error(f"Error unparsing annotation for argument '{arg.arg}': {e}")
        log_debug(f"Type annotation for '{arg.arg}': {annotation}")
        return annotation

    def _has_default(self, arg: ast.arg) -> bool:
        """Check if argument has default value."""
        has_default = hasattr(arg, 'default') and arg.default is not None
        log_debug(f"Argument '{arg.arg}' has default: {has_default}")
        return has_default

    def _get_default_value(self, arg: ast.arg) -> Optional[str]:
        """Get default value as string if it exists."""
        default_value = None
        if hasattr(arg, 'default') and arg.default is not None:
            try:
                default_value = ast.unparse(arg.default)
            except Exception as e:
                log_error(f"Error unparsing default value for argument '{arg.arg}': {e}")
        log_debug(f"Default value for '{arg.arg}': {default_value}")
        return default_value

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Get return type annotation as string."""
        return_type = "Any"
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception as e:
                log_error(f"Error unparsing return type for function '{node.name}': {e}")
        log_debug(f"Return type for function '{node.name}': {return_type}")
        return return_type

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