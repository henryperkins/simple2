import asyncio
from schema import DocstringSchema, JSON_SCHEMA
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json
from typing import Optional
from logger import log_info, log_error, log_debug

class ResponseParser:
    """
    Parses the response from Azure OpenAI to extract docstring, summary, and other relevant metadata.
    """

    def parse_docstring_response(self, response: str) -> Optional[DocstringSchema]:
        """Parse and validate AI response against schema."""
        log_debug("Parsing docstring response.")
        try:
            docstring_data = json.loads(response)
            log_debug(f"Docstring data loaded: {docstring_data}")
            validate(instance=docstring_data, schema=JSON_SCHEMA)
            log_info("Successfully validated docstring response against schema.")
            return DocstringSchema(**docstring_data)
        except json.JSONDecodeError as e:
            log_error(f"JSON decoding error: {e}")
        except ValidationError as e:
            log_error(f"Schema validation error: {e}")
        except Exception as e:
            log_error(f"Unexpected error during docstring parsing: {e}")
        return None

    @staticmethod
    def parse_json_response(response: str) -> dict:
        """
        Parse the Azure OpenAI response to extract the generated docstring and related details.
        """
        log_debug("Parsing JSON response.")
        try:
            response_json = json.loads(response)
            log_info("Successfully parsed Azure OpenAI response.")
            log_debug(f"Parsed JSON response: {response_json}")
            return response_json
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse response as JSON: {e}")
            return {}

    @staticmethod
    def _parse_plain_text_response(text: str) -> dict:
        """
        Fallback parser for plain text responses from Azure OpenAI.
        """
        log_debug("Parsing plain text response.")
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
                        log_debug(f"Extracted {current_key}: {result[current_key]}")
                        buffer = []
                    current_key = line[:-1]
                else:
                    buffer.append(line)
            if current_key and buffer:
                result[current_key] = '\n'.join(buffer).strip()
                log_debug(f"Extracted {current_key}: {result[current_key]}")
            log_info("Successfully parsed Azure OpenAI plain text response.")
            return result
        except Exception as e:
            log_error(f"Failed to parse plain text response: {e}")
            return {}