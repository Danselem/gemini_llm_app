from pathlib import Path
from google.genai import types
from google.genai.errors import ClientError as GoogleClientError
from google.genai.types import GenerateContentResponse

from src.llm.gemini_client import get_client
from src.handlers.error_handler import APIError, ClientError
from src.utils.logger import logger

import requests

html_url="https://arxiv.org/html/1706.03762v7"
MODEL_ID: str = 'gemini-2.0-flash-001'

client = get_client()

# Get the contents of the HTML URL using the Python Requests library
content = requests.get(html_url).content

# Create a multi-part prompt
prompt = [
    content,
    "Summarize the above content"
]

# Generate our response
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)