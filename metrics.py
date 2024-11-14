"""
Code Metrics Analysis Module

This module provides comprehensive code quality and complexity metrics for Python source code,
including cyclomatic complexity, cognitive complexity, Halstead metrics, and code quality analysis.

Version: 1.0.0
Author: Development Team
"""

import ast
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from logger import log_info, log_error, log_debug

class MetricsError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class Metrics:
    """
    Provides methods to calculate different complexity metrics for Python functions.
    """

    MAINTAINABILITY_THRESHOLDS = {
        'good': 80,
        'moderate': 60,
        'poor': 40
    }

    @staticmethod
    def calculate_cyclomatic_complexity(function_node: ast.FunctionDef) -> int:
        """
        Calculate the cyclomatic complexity of a function.

        Parameters:
        function_node (ast.FunctionDef): The AST node representing the function.

        Returns:
        int: The cyclomatic complexity of the function.
        """
        log_debug(f"Calculating cyclomatic complexity for function: {function_node.name}")
        if not isinstance(function_node, ast.FunctionDef):
            log_error("Provided node is not a function definition.")
            return 0

        complexity = 1  # Start with 1 for the function itself
        for node in ast.walk(function_node):
            if Metrics._is_decision_point(node):
                complexity += 1
                log_debug(f"Incremented complexity at node: {ast.dump(node)}")

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
        log_debug(f"Calculating cognitive complexity for function: {function_node.name}")
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
                log_debug(f"Nesting depth increased to {nesting_depth} at node: {ast.dump(node)}")
            elif Metrics._is_complexity_increment(node, prev_node):
                cognitive_complexity += 1
                log_debug(f"Incremented cognitive complexity at node: {ast.dump(node)}")
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
        decision_point = isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler))
        log_debug(f"Node {ast.dump(node)} is {'a' if decision_point else 'not a'} decision point.")
        return decision_point

    @staticmethod
    def _is_nesting_construct(node: ast.AST) -> bool:
        """
        Determine if a node represents a nesting construct for cognitive complexity.

        Parameters:
        node (ast.AST): The AST node to check.

        Returns:
        bool: True if the node is a nesting construct, False otherwise.
        """
        nesting_construct = isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
        log_debug(f"Node {ast.dump(node)} is {'a' if nesting_construct else 'not a'} nesting construct.")
        return nesting_construct

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
        increment = isinstance(node, (ast.BoolOp, ast.Compare)) and not isinstance(prev_node, (ast.BoolOp, ast.Compare)) or isinstance(node, (ast.Continue, ast.Break, ast.Raise, ast.Return))
        log_debug(f"Node {ast.dump(node)} {'increments' if increment else 'does not increment'} complexity.")
        return increment

    def calculate_maintainability_index(self, node: ast.AST) -> float:
        """
        Calculate maintainability index based on various metrics.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            float: Maintainability index score (0-100)
        """
        log_debug("Calculating maintainability index.")
        try:
            halstead = self.calculate_halstead_metrics(node)
            complexity = self.calculate_complexity(node)
            sloc = self._count_source_lines(node)
            
            # Calculate Maintainability Index
            volume = halstead['program_volume']
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
            mi = max(0, min(100, mi))  # Normalize to 0-100
            
            log_info(f"Calculated maintainability index is {mi}")
            return round(mi, 2)
            
        except Exception as e:
            log_error(f"Error calculating maintainability index: {e}")
            return 0.0

    def _count_source_lines(self, node: ast.AST) -> int:
        """
        Count source lines of code (excluding comments and blank lines).
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            int: Number of source code lines
        """
        log_debug("Counting source lines of code.")
        try:
            source = ast.unparse(node)
            lines = [line.strip() for line in source.splitlines()]
            count = len([line for line in lines if line and not line.startswith('#')])
            log_info(f"Counted {count} source lines of code.")
            return count
        except Exception as e:
            log_error(f"Error counting source lines: {e}")
            return 0

    def analyze_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """
        Analyze module dependencies and imports.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            Dict[str, Set[str]]: Dictionary of module dependencies
        """
        log_debug("Analyzing module dependencies.")
        deps = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }
        
        try:
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    self._process_import(subnode, deps)
            log_info(f"Analyzed dependencies: {deps}")
            return deps
        except Exception as e:
            log_error(f"Error analyzing dependencies: {e}")
            return deps

    def _process_import(self, node: ast.AST, deps: Dict[str, Set[str]]) -> None:
        """Process import statement and categorize dependency."""
        log_debug(f"Processing import: {ast.dump(node)}")
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._categorize_import(name.name, deps)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._categorize_import(node.module, deps)
        except Exception as e:
            log_error(f"Error processing import: {e}")

    def _categorize_import(self, module_name: str, deps: Dict[str, Set[str]]) -> None:
        """Categorize import as stdlib, third-party, or local."""
        log_debug(f"Categorizing import: {module_name}")
        try:
            if module_name in sys.stdlib_module_names:
                deps['stdlib'].add(module_name)
            elif '.' in module_name:
                deps['local'].add(module_name)
            else:
                deps['third_party'].add(module_name)
        except Exception as e:
            log_error(f"Error categorizing import {module_name}: {e}")

# Suggested Test Cases
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

    # Test cyclomatic complexity
    cyclomatic_complexity = Metrics.calculate_cyclomatic_complexity(function_node)
    assert cyclomatic_complexity == 5, f"Expected 5, got {cyclomatic_complexity}"

    # Test cognitive complexity
    cognitive_complexity = Metrics.calculate_cognitive_complexity(function_node)
    assert cognitive_complexity == 6, f"Expected 6, got {cognitive_complexity}"

    log_info("All tests passed.")

# Run tests
test_metrics()