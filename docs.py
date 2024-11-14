#!/usr/bin/env python3
"""
docs.py - Documentation Generation System

This module provides a comprehensive system for generating documentation from Python source code,
including docstring management, markdown generation, and documentation workflow automation.
"""

import os
import ast
import logging
import inspect
import importlib
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime

class DocStringManager:
    """Manages docstring operations for source code files."""

    def __init__(self, source_code: str):
        """
        Initialize with source code.

        Args:
            source_code (str): The source code to manage docstrings for
        """
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.docstring_parser = DocStringParser()
        logging.debug("DocStringManager initialized.")

    def insert_docstring(self, node: ast.FunctionDef, docstring: str) -> None:
        """
        Insert or update docstring for a function node.

        Args:
            node (ast.FunctionDef): The function node to update
            docstring (str): The new docstring to insert
        """
        logging.debug(f"Inserting docstring into function '{node.name}'.")
        node.body.insert(0, ast.Expr(value=ast.Str(s=docstring)))

    def update_source_code(self, documentation_entries: List[Dict]) -> str:
        """
        Update source code with new docstrings.

        Args:
            documentation_entries (List[Dict]): List of documentation updates

        Returns:
            str: Updated source code
        """
        logging.debug("Updating source code with new docstrings.")
        for entry in documentation_entries:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.FunctionDef) and node.name == entry['function_name']:
                    self.insert_docstring(node, entry['docstring'])

        updated_code = ast.unparse(self.tree)
        logging.info("Source code updated with new docstrings.")
        return updated_code

    def generate_markdown_documentation(
        self,
        documentation_entries: List[Dict],
        module_name: str = "",
        file_path: str = "",
        description: str = ""
    ) -> str:
        """
        Generate markdown documentation for the code.

        Args:
            documentation_entries (List[Dict]): List of documentation updates
            module_name (str): Name of the module
            file_path (str): Path to the source file 
            description (str): Module description

        Returns:
            str: Generated markdown documentation
        """
        logging.debug("Generating markdown documentation.")
        markdown_gen = MarkdownGenerator()
        if module_name:
            markdown_gen.add_header(f"Module: {module_name}")
        if description:
            markdown_gen.add_section("Description", description)
    
        for entry in documentation_entries:
            if 'function_name' in entry and 'docstring' in entry:
                markdown_gen.add_section(
                    f"Function: {entry['function_name']}",
                    entry['docstring']
                )

        markdown = markdown_gen.generate_markdown()
        logging.info("Markdown documentation generated.")
        return markdown
    
class DocStringParser:
    """Handles parsing and extraction of docstrings from Python source code."""
    
    @staticmethod
    def extract_docstring(source_code: str) -> Optional[str]:
        """
        Extract the module-level docstring from source code.

        Args:
            source_code (str): The source code to parse

        Returns:
            Optional[str]: The extracted docstring or None if not found
        """
        logging.debug("Extracting module-level docstring.")
        try:
            tree = ast.parse(source_code)
            docstring = ast.get_docstring(tree)
            logging.debug(f"Extracted docstring: {docstring}")
            return docstring
        except Exception as e:
            logging.error(f"Failed to parse source code: {e}")
            return None

    @staticmethod
    def parse_function_docstring(func) -> Dict[str, Any]:
        """
        Parse function docstring into structured format.

        Args:
            func: The function object to parse

        Returns:
            Dict[str, Any]: Structured docstring information
        """
        logging.debug(f"Parsing docstring for function: {func.__name__}")
        doc = inspect.getdoc(func)
        if not doc:
            logging.debug("No docstring found.")
            return {}

        sections = {
            'description': '',
            'args': {},
            'returns': '',
            'raises': [],
            'examples': []
        }

        current_section = 'description'
        lines = doc.split('\n')

        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:'):
                current_section = 'args'
                continue
            elif line.lower().startswith('returns:'):
                current_section = 'returns'
                continue
            elif line.lower().startswith('raises:'):
                current_section = 'raises'
                continue
            elif line.lower().startswith('example'):
                current_section = 'examples'
                continue

            if current_section == 'description' and line:
                sections['description'] += line + ' '
            elif current_section == 'args' and line:
                if ':' in line:
                    param, desc = line.split(':', 1)
                    sections['args'][param.strip()] = desc.strip()
            elif current_section == 'returns' and line:
                sections['returns'] += line + ' '
            elif current_section == 'raises' and line:
                sections['raises'].append(line)
            elif current_section == 'examples' and line:
                sections['examples'].append(line)

        logging.debug(f"Parsed docstring sections: {sections}")
        return sections

class DocStringGenerator:
    """Generates docstrings for Python code elements."""

    @staticmethod
    def generate_class_docstring(class_name: str, description: str) -> str:
        """
        Generate a docstring for a class.

        Args:
            class_name (str): Name of the class
            description (str): Description of the class purpose

        Returns:
            str: Generated docstring
        """
        logging.debug(f"Generating class docstring for: {class_name}")
        return f'"""{description}\n\nAttributes:\n    Add class attributes here\n"""'

    @staticmethod
    def generate_function_docstring(
        func_name: str,
        params: List[str],
        description: str,
        returns: Optional[str] = None
    ) -> str:
        """
        Generate a docstring for a function.

        Args:
            func_name (str): Name of the function
            params (List[str]): List of parameter names
            description (str): Description of the function
            returns (Optional[str]): Description of return value

        Returns:
            str: Generated docstring
        """
        logging.debug(f"Generating function docstring for: {func_name}")
        docstring = f'"""{description}\n\nArgs:\n'
        for param in params:
            docstring += f"    {param}: Description for {param}\n"
        
        if returns:
            docstring += f"\nReturns:\n    {returns}\n"
        
        docstring += '"""'
        logging.debug(f"Generated docstring: {docstring}")
        return docstring

class MarkdownGenerator:
    """Generates markdown documentation from Python code elements."""

    def __init__(self):
        """Initialize the MarkdownGenerator."""
        self.output = []
        logging.debug("MarkdownGenerator initialized.")

    def add_header(self, text: str, level: int = 1) -> None:
        """
        Add a header to the markdown document.

        Args:
            text (str): Header text
            level (int): Header level (1-6)
        """
        logging.debug(f"Adding header: {text}")
        self.output.append(f"{'#' * level} {text}\n")

    def add_code_block(self, code: str, language: str = "python") -> None:
        """
        Add a code block to the markdown document.

        Args:
            code (str): The code to include
            language (str): Programming language for syntax highlighting
        """
        logging.debug("Adding code block.")
        self.output.append(f"```{language}\n{code}\n```\n")

    def add_section(self, title: str, content: str) -> None:
        """
        Add a section with title and content.

        Args:
            title (str): Section title
            content (str): Section content
        """
        logging.debug(f"Adding section: {title}")
        self.output.append(f"### {title}\n\n{content}\n")

    def generate_markdown(self) -> str:
        """
        Generate the final markdown document.

        Returns:
            str: Complete markdown document
        """
        logging.debug("Generating final markdown document.")
        return "\n".join(self.output)

class DocumentationManager:
    """Manages the overall documentation generation process."""

    def __init__(self, output_dir: str = "docs"):
        """
        Initialize the DocumentationManager.

        Args:
            output_dir (str): Directory for output documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parser = DocStringParser()
        self.generator = DocStringGenerator()
        self.markdown = MarkdownGenerator()
        self.logger = self._setup_logging()
        logging.debug("DocumentationManager initialized.")

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('documentation_manager')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a single Python file for documentation.

        Args:
            file_path (Union[str, Path]): Path to the Python file

        Returns:
            Optional[str]: Generated markdown documentation
        """
        logging.debug(f"Processing file: {file_path}")
        try:
            file_path = Path(file_path)
            if not file_path.exists() or file_path.suffix != '.py':
                self.logger.error(f"Invalid Python file: {file_path}")
                return None

            with open(file_path, 'r') as f:
                source = f.read()

            module_doc = self.parser.extract_docstring(source)
            
            self.markdown.add_header(f"Documentation for {file_path.name}")
            if module_doc:
                self.markdown.add_section("Module Description", module_doc)

            # Parse the source code
            tree = ast.parse(source)
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._process_class(node)
                elif isinstance(node, ast.FunctionDef):
                    self._process_function(node)

            markdown = self.markdown.generate_markdown()
            logging.info(f"Generated markdown for file: {file_path}")
            return markdown

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _process_class(self, node: ast.ClassDef) -> None:
        """
        Process a class definition node.

        Args:
            node (ast.ClassDef): AST node representing a class definition
        """
        logging.debug(f"Processing class: {node.name}")
        try:
            class_doc = ast.get_docstring(node)
            self.markdown.add_section(f"Class: {node.name}", 
                                    class_doc if class_doc else "No documentation available")
            
            # Process class methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self._process_function(item, is_method=True, class_name=node.name)
        except Exception as e:
            self.logger.error(f"Error processing class {node.name}: {e}")

    def _process_function(self, node: ast.FunctionDef, is_method: bool = False, class_name: str = None) -> None:
        """
        Process a function definition node.

        Args:
            node (ast.FunctionDef): AST node representing a function definition
            is_method (bool): Whether the function is a class method
            class_name (str): Name of the containing class if is_method is True
        """
        logging.debug(f"Processing function: {node.name}")
        try:
            func_doc = ast.get_docstring(node)
            section_title = f"{'Method' if is_method else 'Function'}: {node.name}"
            if is_method:
                section_title = f"Method: {class_name}.{node.name}"

            # Extract function signature
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"

            content = [
                f"```python\n{signature}\n```\n",
                func_doc if func_doc else "No documentation available"
            ]
            
            self.markdown.add_section(section_title, "\n".join(content))
        except Exception as e:
            self.logger.error(f"Error processing function {node.name}: {e}")

    def process_directory(self, directory_path: Union[str, Path]) -> Dict[str, str]:
        """
        Process all Python files in a directory for documentation.

        Args:
            directory_path (Union[str, Path]): Path to the directory to process

        Returns:
            Dict[str, str]: Dictionary mapping file paths to their documentation
        """
        logging.debug(f"Processing directory: {directory_path}")
        directory_path = Path(directory_path)
        results = {}

        if not directory_path.is_dir():
            self.logger.error(f"Invalid directory path: {directory_path}")
            return results

        for file_path in directory_path.rglob("*.py"):
            try:
                doc_content = self.process_file(file_path)
                if doc_content:
                    results[str(file_path)] = doc_content
            except Exception as e:
                self.logger.error(f"Error processing directory {directory_path}: {e}")

        return results

    def save_documentation(self, content: str, output_file: Union[str, Path]) -> bool:
        """
        Save documentation content to a file.

        Args:
            content (str): Documentation content to save
            output_file (Union[str, Path]): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        logging.debug(f"Saving documentation to: {output_file}")
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Documentation saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving documentation: {e}")
            return False

    def generate_index(self, docs_map: Dict[str, str]) -> str:
        """
        Generate an index page for all documentation files.

        Args:
            docs_map (Dict[str, str]): Dictionary mapping file paths to their documentation

        Returns:
            str: Generated index page content
        """
        logging.debug("Generating documentation index.")
        index_content = [
            "# Documentation Index\n",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Files\n"
        ]

        for file_path in sorted(docs_map.keys()):
            rel_path = Path(file_path).name
            doc_path = Path(file_path).with_suffix('.md').name
            index_content.append(f"- [{rel_path}]({doc_path})")

        logging.info("Documentation index generated.")
        return "\n".join(index_content)

def main():
    """Main function to demonstrate usage of the documentation system."""
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize DocumentationManager
        doc_manager = DocumentationManager(output_dir="generated_docs")

        # Example usage
        source_dir = "."  # Current directory
        docs_map = doc_manager.process_directory(source_dir)

        # Generate and save documentation for each file
        for file_path, content in docs_map.items():
            output_file = Path("generated_docs") / Path(file_path).with_suffix('.md').name
            doc_manager.save_documentation(content, output_file)

        # Generate and save index
        index_content = doc_manager.generate_index(docs_map)
        doc_manager.save_documentation(index_content, "generated_docs/index.md")

        logger.info("Documentation generation completed successfully")

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise

if __name__ == "__main__":
    main()