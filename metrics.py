import ast
from logger import log_info, log_error

class Metrics:
    """
    Provides methods to calculate different complexity metrics for Python functions.
    """

    @staticmethod
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cyclomatic complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cyclomatic complexity of the function.
        """
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        complexity = 1  # Start with 1 for the function itself
        for node in ast.walk(function_node):
            if Metrics._is_decision_point(node):
                complexity += 1

        log_info(f"Calculated cyclomatic complexity for function '{function_node.name}' is {complexity}")
        return complexity

    @staticmethod
    def calculate_cognitive_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cognitive complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cognitive complexity of the function.
        """
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        cognitive_complexity = 0
        nesting_depth = 0
        prev_node = None

        for node in ast.walk(function_node):
            if Metrics._is_nesting_construct(node):
                nesting_depth += 1
                cognitive_complexity += nesting_depth
            elif Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
            prev_node = node

        log_info(f"Calculated cognitive complexity for function '{function_node.name}' is {cognitive_complexity}")
        return cognitive_complexity

    @staticmethod
    def _is_decision_point(node: ast.AST) -> bool:
        """
        Determine if a node represents a decision point for cyclomatic complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a decision point, False otherwise.
        """
        return isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler))

    @staticmethod
    def _is_nesting_construct(node: ast.AST) -> bool:
        """
        Determine if a node represents a nesting construct for cognitive complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a nesting construct, False otherwise.
        """
        return isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))

    @staticmethod
    def _is_complexity_increment(node: ast.AST, prev_node: ast.AST) -> bool:
        """
        Determine if a node should increment cognitive complexity.

        Parameters:
        node (ast.AST): The current AST node.
        prev_node (ast.AST): The previous AST node.

        Returns:
        bool: True if the node should increment complexity, False otherwise.
        """
        return isinstance(node, (ast.BoolOp, ast.Compare)) and not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or isinstance(node, (ast.Continue, ast.Break, ast.Raise, ast.Return))

# Suggested Test Cases
def test_metrics():
    source_code = """
def example_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            else:
                continue
    else:
        return -1
    return 0
"""
    tree = ast.parse(source_code)
    function_node = tree.body[0]  # Assuming the first node is the function definition

    # Test cyclomatic complexity
    cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)
    assert cyclomatic_complexity == 5, f"Expected 5, got {cyclomatic_complexity}"

    # Test cognitive complexity
    cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)
    assert cognitive_complexity == 6, f"Expected 6, got {cognitive_complexity}"

    print("All tests passed.")

# Run tests
test_metrics()