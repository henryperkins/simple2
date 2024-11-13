from schema import DocstringSchema, JSON_SCHEMA
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json
from typing import Optional
from logger import log_info, log_error

class ResponseParser:
    """
    Parses the response from Azure OpenAI to extract docstring, summary, and other relevant metadata.
    """

    def parse_docstring_response(self, response: str) -> Optional[DocstringSchema]:
        """Parse and validate AI response against schema."""
        try:
            docstring_data = json.loads(response)
            validate(instance=docstring_data, schema=JSON_SCHEMA)
            log_info("Successfully validated docstring response against schema.")
            return DocstringSchema(**docstring_data)
        except (json.JSONDecodeError, ValidationError) as e:
            log_error(f"Invalid docstring format: {e}")
            return None

    @staticmethod
    def parse_json_response(response: str) -> dict:
        """
        Parse the Azure OpenAI response to extract the generated docstring and related details.
        """
        try:
            # Attempt to parse as JSON
            response_json = json.loads(response)
            log_info("Successfully parsed Azure OpenAI response.")
            return response_json
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse response as JSON: {e}")
            return {}

    @staticmethod
    def _parse_plain_text_response(text: str) -> dict:
        """
        Fallback parser for plain text responses from Azure OpenAI.
        """
        try:
            lines = text.strip().split('\n')
            result = {}
            current_key = None
            buffer = []

            for line in lines:
                line = line.strip()
                if line.endswith(':') and line[:-1] in ['summary', 'changelog', 'docstring', 'complexity_score']:
                    if current_key and buffer:
                        result[current_key] = '\n'.join(buffer).strip()
                        buffer = []
                    current_key = line[:-1]
                else:
                    buffer.append(line)
            if current_key and buffer:
                result[current_key] = '\n'.join(buffer).strip()
            log_info("Successfully parsed Azure OpenAI plain text response.")
            return result
        except Exception as e:
            log_error(f"Failed to parse plain text response: {e}")
            return {}