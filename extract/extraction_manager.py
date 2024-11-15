"""
Extraction Manager Module

This module provides a manager class for extracting metadata from Python source code.
It coordinates the extraction of class and function metadata using dedicated extractors.

Version: 1.0.0
Author: Development Team
"""

from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor

class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass

class ExtractionManager:
    """
    A manager class that coordinates the extraction of metadata from Python source code.

    Attributes:
        class_extractor (ClassExtractor): An instance of ClassExtractor for extracting class metadata.
        function_extractor (FunctionExtractor): An instance of FunctionExtractor for extracting function metadata.
    """

    def __init__(self):
        """Initialize the ExtractionManager without source code."""
        self.class_extractor = None
        self.function_extractor = None

    def validate_source_code(self, source_code: str) -> bool:
        """
        Validate the provided source code.

        Args:
            source_code (str): The Python source code to validate.

        Returns:
            bool: True if the source code is valid, False otherwise.

        Raises:
            ValueError: If the source code is empty.
        """
        if not source_code:
            raise ValueError("Source code cannot be empty.")
        # Additional validation logic can be added here
        return True

    def set_source_code(self, source_code: str):
        """
        Set the source code for extraction.

        Args:
            source_code (str): The Python source code to set.

        Raises:
            ValueError: If the source code is invalid.
        """
        self.validate_source_code(source_code)
        self.class_extractor = ClassExtractor(source_code)
        self.function_extractor = FunctionExtractor(source_code)

    def extract_metadata(self, source_code: str) -> dict:
        """
        Extracts metadata from the provided source code.

        Args:
            source_code (str): The Python source code to analyze.

        Returns:
            dict: A dictionary containing extracted metadata with two keys:
                - 'classes': A list of dictionaries, each containing metadata for a class.
                - 'functions': A list of dictionaries, each containing metadata for a function.

        Raises:
            ExtractionError: If there is an error during extraction.
        """
        try:
            self.set_source_code(source_code)
            class_metadata = self.class_extractor.extract_classes()
            function_metadata = self.function_extractor.extract_functions()
            
            return {
                'classes': class_metadata,
                'functions': function_metadata
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract metadata: {e}")

# Example usage
if __name__ == "__main__":
    try:
        extraction_manager = ExtractionManager()
        source_code = "def example_function(): pass"
        metadata = extraction_manager.extract_metadata(source_code)
        print(metadata)
    except ValueError as ve:
        print(f"Validation error: {ve}")
    except ExtractionError as ee:
        print(f"Extraction error: {ee}")
    except Exception as e:
        print(f"Unexpected error: {e}")