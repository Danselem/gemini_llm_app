from dotenv import load_dotenv
from google.genai import types
from google.genai.types import Tool, GoogleSearch
from src.llm.gemini_client import get_client
load_dotenv()

model = "gemini-2.0-flash-001"
client = get_client()

prompt = "What are the top 5 3-star top-rated hotels in Enugu city, Nigeria? Include 1 fact about each."

response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())], # <== Note the use of the GoogleSearch tool
    ),
)

print(response.text)