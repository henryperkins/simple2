
import pytest
import ast
from metrics import Metrics

# test_metrics.py
def test_metrics():
    log_info("Starting test_metrics.")
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

    if isinstance(function_node, ast.FunctionDef):
        # Test cyclomatic complexity
        cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)
        assert cyclomatic_complexity == 4, f"Expected 4, got {cyclomatic_complexity}"

        # Test cognitive complexity
        cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)
        assert cognitive_complexity == 6, f"Expected 6, got {cognitive_complexity}"

        log_info("All tests passed.")
    else:
        log_error("The node is not a function definition.")
def test_cyclomatic_complexity():
    """Test cyclomatic complexity calculation."""
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
    function_node = tree.body[0]
    complexity = Metrics.calculate_cyclomatic_complexity(function_node)
    assert complexity == 4, f"Expected 4, got {complexity}"
def test_cognitive_complexity():
    """Test cognitive complexity calculation."""
    source_code = """
def nested_function(x):
    if x > 0:
        while x > 5:
            for i in range(x):
                if i % 2 == 0:
                    return i
    return 0
"""
    tree = ast.parse(source_code)
    function_node = tree.body[0]
    complexity = Metrics.calculate_cognitive_complexity(function_node)
    assert complexity == 7

def test_invalid_node():
    """Test handling of invalid nodes."""
    tree = ast.parse("")
    module_node = tree
    cyclo = Metrics.calculate_cyclomatic_complexity(module_node)
    cogn = Metrics.calculate_cognitive_complexity(module_node)
    assert cyclo == 0
    assert cogn == 0