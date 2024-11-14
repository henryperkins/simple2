import ast
from logger import log_info, log_error, log_debug

class BaseExtractor:
    """
    Base class for extracting information from AST nodes.
    """
    def __init__(self, source_code):
        try:
            log_debug("Attempting to parse source code into AST.")
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed for extraction.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code: {e}")
            raise e

    def walk_tree(self):
        """
        Walk the AST and yield all nodes.
        """
        log_debug("Walking through the AST nodes.")
        for node in ast.walk(self.tree):
            log_debug(f"Yielding AST node: {type(node).__name__}")
            yield node

    def extract_docstring(self, node):
        """
        Extract docstring from an AST node.
        """
        try:
            log_debug(f"Extracting docstring from node: {type(node).__name__}")
            docstring = ast.get_docstring(node)
            if docstring:
                log_info(f"Docstring extracted from node: {type(node).__name__}")
            else:
                log_info(f"No docstring found in node: {type(node).__name__}")
            return docstring
        except Exception as e:
            log_error(f"Failed to extract docstring: {e}")
            return None

    def extract_annotations(self, node):
        """
        Extract type annotations from an AST node.
        """
        try:
            log_debug(f"Extracting annotations from node: {type(node).__name__}")
            if isinstance(node, ast.FunctionDef):
                annotations = {
                    'returns': ast.unparse(node.returns) if node.returns else "Any",
                    'args': [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") 
                             for arg in node.args.args]
                }
                log_info(f"Annotations extracted for function '{node.name}': {annotations}")
                return annotations
            log_info(f"No annotations found for node: {type(node).__name__}")
            return {}
        except Exception as e:
            log_error(f"Failed to extract annotations: {e}")
            return {}