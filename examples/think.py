from dotenv import load_dotenv
from google.genai import types
from pathlib import Path
from src.llm.gemini_client import get_client
from src.utils.logger import logger

load_dotenv()
model = "gemini-2.5-flash-preview-04-17"

client = get_client()

prompt = "Explain the Occam's Razor concept and provide everyday examples of it"

response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024) # thinkingBudget must be an integer in the range 0 to 24576
    ),
)

data_path = Path("data/outputs")
data_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "think_output.txt", "w") as f:
    f.write(response.text)

logger.info(f"Think output: {response.model_dump_json(indent=2)}")