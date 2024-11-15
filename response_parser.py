"""
response_parser.py - AI Response Parsing System

This module provides functionality to parse and validate responses from Azure OpenAI,
focusing on extracting docstrings, summaries, and other metadata.

Classes:
    ResponseParser: Parses the response from Azure OpenAI to extract docstring, summary, and other relevant metadata.

Methods:
    parse_docstring_response(response: str) -> Optional[DocstringSchema]: Parses and validates AI response against schema.
    parse_json_response(response: str) -> Optional[Dict[str, Any]]: Parses the Azure OpenAI response to extract generated docstring and related details.
    _parse_plain_text_response(text: str) -> dict: Fallback parser for plain text responses from Azure OpenAI.
"""

import asyncio
from schema import DocstringSchema, JSON_SCHEMA
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json
from typing import Optional, Dict, Any
from logger import log_info, log_error, log_debug

class ResponseParser:
    """
    Parses the response from Azure OpenAI to extract docstring, summary, and other relevant metadata.

    Methods:
        parse_docstring_response(response: str) -> Optional[DocstringSchema]: Parses and validates AI response against schema.
        parse_json_response(response: str) -> Optional[Dict[str, Any]]: Parses the Azure OpenAI response to extract generated docstring and related details.
        _parse_plain_text_response(text: str) -> dict: Fallback parser for plain text responses from Azure OpenAI.
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
            # Log detailed information about the validation failure
            log_error(f"Docstring validation error: {e.message}")
            log_error(f"Failed docstring content: {docstring_data}")
            log_error(f"Schema path: {e.schema_path}")
            log_error(f"Validator: {e.validator} - Constraint: {e.validator_value}")
        except Exception as e:
            log_error(f"Unexpected error during docstring parsing: {e}")
        return None

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the Azure OpenAI response to extract the generated docstring and related details.
        """
        log_debug("Parsing JSON response.")
        try:
            response_json = json.loads(response)
            log_info("Successfully parsed Azure OpenAI response.")
            log_debug(f"Parsed JSON response: {response_json}")

            # Validate against JSON schema
            validate(instance=response_json, schema=JSON_SCHEMA)

            # Extract relevant fields
            docstring = response_json.get("docstring", "")
            summary = response_json.get("summary", "")
            changelog = response_json.get("changelog", "Initial documentation")
            complexity_score = response_json.get("complexity_score", 0)

            log_debug(f"Extracted docstring: {docstring}")
            log_debug(f"Extracted summary: {summary}")
            log_debug(f"Extracted changelog: {changelog}")
            log_debug(f"Extracted complexity score: {complexity_score}")

            return {
                "docstring": docstring,
                "summary": summary,
                "changelog": changelog,
                "complexity_score": complexity_score
            }
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse response as JSON: {e}")
        except ValidationError as e:
            # Log detailed information about the validation failure
            log_error(f"Response validation error: {e.message}")
            log_error(f"Failed response content: {response_json}")
            log_error(f"Schema path: {e.schema_path}")
            log_error(f"Validator: {e.validator} - Constraint: {e.validator_value}")
        except Exception as e:
            log_error(f"Unexpected error during JSON response parsing: {e}")
        return None

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