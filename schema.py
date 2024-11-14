# schema.py
import time
from typing import TypedDict, List, Optional, Union
from enum import Enum
import json
from pathlib import Path

class DocstringParameter(TypedDict):
    name: str
    type: str
    description: str
    optional: bool
    default_value: Optional[str]

class DocstringReturns(TypedDict):
    type: str
    description: str

class DocstringException(TypedDict):
    exception: str
    description: str

class NoteType(Enum):
    NOTE = "note"
    WARNING = "warning"
    TIP = "tip"
    IMPORTANT = "important"

class DocstringNote(TypedDict):
    type: NoteType
    content: str

class DocstringExample(TypedDict):
    code: str
    description: Optional[str]

class DocstringMetadata(TypedDict):
    author: Optional[str]
    since_version: Optional[str]
    deprecated: Optional[dict]
    complexity: Optional[dict]

class DocstringSchema(TypedDict):
    description: str
    parameters: List[DocstringParameter]
    returns: DocstringReturns
    raises: Optional[List[DocstringException]]
    examples: Optional[List[DocstringExample]]
    notes: Optional[List[DocstringNote]]
    metadata: Optional[DocstringMetadata]

# Load JSON schema
def load_schema() -> dict:
    schema_path = Path(__file__).parent / 'docstring_schema.json'
    with open(schema_path) as f:
        return json.load(f)

try:
    JSON_SCHEMA = load_schema()
except Exception as e:
    print(f"Warning: Could not load JSON schema: {e}")
    JSON_SCHEMA = {}