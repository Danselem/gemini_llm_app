from dotenv import load_dotenv
from google.genai import types
from pathlib import Path
from src.llm.gemini_client import get_client

load_dotenv()

client = get_client()

CUSTOM_SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

# Create a custom generation config to customize settings, such as temperature,
# response type, etc.
# - https://googleapis.github.io/python-genai/index.html#typed-config
MODEL_ID= 'gemini-2.0-flash-001' # @param ["gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-05-20", "gemini-2.0-flash-001", "gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-pro-exp-02-05","gemini-2.0-flash-thinking-exp-01-21", "gemini-1.5-flash-002", "gemini-1.5-pro-002", "meta/llama-3.2-90b-vision-instruct-maas", "meta/llama-3.1-405b-instruct-maas","meta/llama-3.1-70b-instruct-maas","meta/llama-3.1-8b-instruct-maas"]
TEMPERATURE= 1 # @param {type:"slider", min:0, max:2, step:0.01}
TOP_P = 0.95 # @param {type:"slider", min:0, max:1, step:0.05}
TOP_K = 40 # @param {type:"slider", min:1, max:40, step:1}
MAX_OUTPUT_TOKENS = 8192 # @param {type:"slider", min:64, max:8192, step:64}
RESPONSE_TYPE = 'text/plain' # @param ["text/plain", "application/json'"]
SYSTEM_INSTRUCTION = "Speak like Shakespeare. Format your responses in markdown. Use lots of emoji." # @param {type:"string"}
PROMPT = "Tell me 3 facts about Jupiter." # @param {type:"string"}
BUCKET = ""

CUSTOM_GENERATION_CONFIG = {
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "safety_settings": CUSTOM_SAFETY_SETTINGS,
    "responseMimeType": RESPONSE_TYPE,
    "system_instruction": SYSTEM_INSTRUCTION
}

response = client.models.generate_content(
    model=MODEL_ID,
    config=CUSTOM_GENERATION_CONFIG,
    contents=PROMPT
)

data_path = Path("data/outputs")
data_path.mkdir(parents=True, exist_ok=True)

with open(data_path / 'safety_output.txt', 'w') as f:
    f.write(response.text)