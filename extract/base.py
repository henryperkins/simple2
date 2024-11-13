import ast
from logger import log_info, log_error

class BaseExtractor:
    """
    Base class for extracting information from AST nodes.
    """
    def __init__(self, source_code):
        try:
            self.tree = ast.parse(source_code)
            log_info("AST successfully parsed for extraction.")
        except SyntaxError as e:
            log_error(f"Failed to parse source code: {e}")
            raise e

    def walk_tree(self):
        """
        Walk the AST and yield all nodes.
        """
        for node in ast.walk(self.tree):
            yield node

    def extract_docstring(self, node):
        """
        Extract docstring from an AST node.
        """
        try:
            return ast.get_docstring(node)
        except Exception as e:
            log_error(f"Failed to extract docstring: {e}")
            return None

    def extract_annotations(self, node):
        """
        Extract type annotations from an AST node.
        """
        try:
            if isinstance(node, ast.FunctionDef):
                return {
                    'returns': ast.unparse(node.returns) if node.returns else "Any",
                    'args': [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else "Any") 
                            for arg in node.args.args]
                }
            return {}
        except Exception as e:
            log_error(f"Failed to extract annotations: {e}")
            return {}