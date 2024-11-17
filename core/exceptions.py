"""
Custom Exceptions Module

This module defines custom exceptions used in the project.

Version: 1.0.0
Author: Development Team
"""

class TooManyRetriesError(Exception):
    """Exception raised when the maximum number of retries is exceeded."""
    def __init__(self, message: str = "Maximum retry attempts exceeded"):
        self.message = message
        super().__init__(self.message)