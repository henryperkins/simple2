import jsonschema
from schema import (
    DocstringSchema, 
    DocstringParameter, 
    DocstringReturns, 
    DocstringException,
    JSON_SCHEMA
)
from jsonschema import validate
from typing import List, Dict, Optional
import ast
from logger import log_info, log_error
import os

class DocStringManager:
    """Manages docstring generation and documentation using the schema."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.schema = JSON_SCHEMA
        self.tree = ast.parse(source_code)

    def generate_markdown_documentation(self, docstring_data: List[DocstringSchema], module_name: str, file_path: str, description: str) -> str:
        """
        Generate markdown documentation from schema-validated docstring data.
        
        Args:
            docstring_data: List of DocstringSchema objects
            module_name: Name of the module
            file_path: Path to the module file
            description: Brief description of the module
            
        Returns:
            str: Generated markdown documentation
        """
        try:
            # Validate all docstring data against schema
            for entry in docstring_data:
                validate(instance=entry, schema=self.schema)

            docs = []
            
            # Module Overview
            docs.append(f"# Module: {module_name}\n")
            docs.append("## Overview")
            docs.append(f"**File:** `{file_path}`")
            docs.append(f"**Description:** {description}\n")
            
            # Classes
            docs.append("## Classes\n")
            docs.append("| Class | Inherits From | Complexity Score* |")
            docs.append("|-------|---------------|------------------|")
            for entry in docstring_data:
                if entry.get('type') == 'class':
                    docs.append(f"| `{entry.get('name', '')}` | `{entry.get('inherits', '')}` | - |")
            
            # Class Methods
            docs.append("\n### Class Methods\n")
            docs.append("| Class | Method | Parameters | Returns | Complexity Score* |")
            docs.append("|-------|--------|------------|---------|------------------|")
            for entry in docstring_data:
                if entry.get('type') == 'method':
                    params = ", ".join([f"{p.get('name', '')}: {p.get('type', '')}" for p in entry.get('parameters', [])])
                    docs.append(f"| `{entry.get('class', '')}` | `{entry.get('name', '')}` | `({params})` | `{entry.get('returns', {}).get('type', '')}` | - |")
            
            # Functions
            docs.append("\n## Functions\n")
            docs.append("| Function | Parameters | Returns | Complexity Score* |")
            docs.append("|----------|------------|---------|------------------|")
            for entry in docstring_data:
                if entry.get('type') == 'function':
                    params = ", ".join([f"{p.get('name', '')}: {p.get('type', '')}" for p in entry.get('parameters', [])])
                    docs.append(f"| `{entry.get('name', '')}` | `({params})` | `{entry.get('returns', {}).get('type', '')}` | - |")
            
            # Constants and Variables
            docs.append("\n## Constants and Variables\n")
            docs.append("| Name | Type | Value |")
            docs.append("|------|------|-------|")
            for entry in docstring_data:
                if entry.get('type') == 'constant':
                    docs.append(f"| `{entry.get('name', '')}` | `{entry.get('data_type', '')}` | `{entry.get('value', '')}` |")
            
            # Recent Changes
            docs.append("\n## Recent Changes\n")
            for change in entry.get('changes', []):
                docs.append(f"- [{change.get('date', '')}] {change.get('description', '')}")
            
            # Source Code
            docs.append("\n## Source Code\n")
            docs.append("```python")
            docs.append(self.source_code)
            docs.append("```\n")
            
            return "\n".join(docs)
            
        except Exception as e:
            log_error(f"Error generating documentation: {e}")
            return ""

    def _generate_function_section(self, docstring: DocstringSchema) -> str:
        """Generate markdown documentation for a single function."""
        sections = []
        
        # Function description
        sections.append(f"### {docstring.get('name', 'Unknown Function')}\n")
        sections.append(f"{docstring.get('description', '')}\n")
        
        # Parameters section
        parameters = docstring.get('parameters', [])
        if isinstance(parameters, list):
            sections.append("#### Parameters\n")
            for param in parameters:
                if isinstance(param, dict):
                    optional_str = " (Optional)" if param.get('optional', False) else ""
                    default_str = f", default: {param.get('default_value', '')}" if param.get('default_value') else ""
                    sections.append(f"- `{param.get('name', '')}: {param.get('type', '')}`{optional_str}{default_str}")
                    sections.append(f"  - {param.get('description', '')}\n")
        
        # Returns section
        returns = docstring.get('returns', {})
        if isinstance(returns, dict):
            sections.append("#### Returns\n")
            sections.append(f"- `{returns.get('type', '')}`: {returns.get('description', '')}\n")
        
        # Raises section
        raises = docstring.get('raises', [])
        if isinstance(raises, list):
            sections.append("#### Raises\n")
            for exception in raises:
                if isinstance(exception, dict):
                    sections.append(f"- `{exception.get('exception', '')}`: {exception.get('description', '')}\n")
        
        # Examples section
        examples = docstring.get('examples', [])
        if isinstance(examples, list):
            sections.append("#### Examples\n")
            for example in examples:
                if isinstance(example, dict):
                    if example.get('description'):
                        sections.append(f"{example['description']}\n")
                    sections.append("```python\n" + example.get('code', '') + "\n```\n")
        
        # Notes section
        notes = docstring.get('notes', [])
        if isinstance(notes, list):
            sections.append("#### Notes\n")
            for note in notes:
                if isinstance(note, dict):
                    note_type = note.get('type', '')
                    if isinstance(note_type, str):
                        sections.append(f"**{note_type.upper()}:** {note.get('content', '')}\n")
        
        # Metadata section
        metadata = docstring.get('metadata', {})
        if isinstance(metadata, dict):
            sections.append("#### Metadata\n")
            if metadata.get('author'):
                sections.append(f"- Author: {metadata['author']}\n")
            if metadata.get('since_version'):
                sections.append(f"- Since: {metadata['since_version']}\n")
            if metadata.get('deprecated'):
                dep = metadata['deprecated']
                if isinstance(dep, dict):
                    sections.append(f"- **DEPRECATED** in version {dep.get('version', '')}")
                    sections.append(f"  - Reason: {dep.get('reason', '')}")
                    if dep.get('alternative'):
                        sections.append(f"  - Use instead: {dep.get('alternative', '')}\n")
            if metadata.get('complexity'):
                comp = metadata['complexity']
                if isinstance(comp, dict):
                    sections.append("- Complexity:")
                    if comp.get('time'):
                        sections.append(f"  - Time: {comp.get('time', '')}")
                    if comp.get('space'):
                        sections.append(f"  - Space: {comp.get('space', '')}\n")
        
        return "\n".join(sections)

    def insert_docstring(self, node: ast.AST, docstring_data: DocstringSchema) -> None:
        """
        Insert a schema-validated docstring into an AST node.
        
        Args:
            node: The AST node to insert the docstring into
            docstring_data: The docstring data conforming to the schema
        """
        try:
            # Validate against schema
            validate(instance=docstring_data, schema=self.schema)
            
            # Convert schema format to Google-style docstring
            docstring = self._convert_schema_to_docstring(docstring_data)
            
            # Update the node's docstring
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                node.body.insert(0, ast.Expr(value=ast.Str(value=docstring)))
                log_info(f"Inserted docstring for {getattr(node, 'name', 'module')}")
            
        except Exception as e:
            log_error(f"Error inserting docstring: {e}")

    def _convert_schema_to_docstring(self, schema_data: DocstringSchema) -> str:
        """Convert schema format to Google-style docstring."""
        lines = []
        
        # Description
        lines.append(schema_data.get('description', ''))
        lines.append("")
        
        # Parameters
        parameters = schema_data.get('parameters', [])
        if isinstance(parameters, list):
            lines.append("Args:")
            for param in parameters:
                if isinstance(param, dict):
                    optional_str = " (Optional)" if param.get('optional', False) else ""
                    default_str = f", default: {param.get('default_value', '')}" if param.get('default_value') else ""
                    lines.append(f"    {param.get('name', '')} ({param.get('type', '')}){optional_str}{default_str}: {param.get('description', '')}")
            lines.append("")
        
        # Returns
        returns = schema_data.get('returns', {})
        if isinstance(returns, dict):
            lines.append("Returns:")
            lines.append(f"    {returns.get('type', '')}: {returns.get('description', '')}")
            lines.append("")
        
        # Raises
        raises = schema_data.get('raises', [])
        if isinstance(raises, list):
            lines.append("Raises:")
            for exception in raises:
                if isinstance(exception, dict):
                    lines.append(f"    {exception.get('exception', '')}: {exception.get('description', '')}")
            lines.append("")
        
        # Examples
        examples = schema_data.get('examples', [])
        if isinstance(examples, list):
            lines.append("Examples:")
            for example in examples:
                if isinstance(example, dict):
                    if example.get('description'):
                        lines.append(f"    {example['description']}")
                    lines.append("    >>> " + example.get('code', '').replace("\n", "\n    >>> "))
            lines.append("")
        
        # Notes
        notes = schema_data.get('notes', [])
        if isinstance(notes, list):
            lines.append("Notes:")
            for note in notes:
                if isinstance(note, dict):
                    note_type = note.get('type', '')
                    if isinstance(note_type, str):
                        lines.append(f"    {note_type.upper()}: {note.get('content', '')}")
            lines.append("")
        
        return "\n".join(lines)

    def update_source_code(self, docstring_data: List[DocstringSchema]) -> str:
        """
        Update the source code with schema-validated docstrings.
        
        Args:
            docstring_data: List of DocstringSchema objects
        
        Returns:
            str: The updated source code with inserted docstrings
        """
        try:
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Find corresponding docstring data
                    for entry in docstring_data:
                        if entry.get('name') == node.name:
                            self.insert_docstring(node, entry)
                            break
            
            # Unparse the AST back to source code
            updated_code = ast.unparse(self.tree)
            return updated_code
        
        except Exception as e:
            log_error(f"Error updating source code: {e}")
            return self.source_code

    def write_markdown_to_file(self, markdown_content: str, filename: str) -> None:
        """
        Write the generated markdown content to a file in the output directory.
        
        Args:
            markdown_content: The markdown content to write
            filename: The name of the file to write to
        """
        try:
            output_path = os.path.join('output', filename)
            with open(output_path, 'w') as file:
                file.write(markdown_content)
            log_info(f"Markdown documentation written to {output_path}")
        except Exception as e:
            log_error(f"Error writing markdown to file: {e}")
