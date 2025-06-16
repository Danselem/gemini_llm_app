import os
# from dotenv import load_dotenv
from google import genai
from src.llm.utils import _set_env

_set_env("GOOGLE_API_KEY")

def get_client():
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("GOOGLE_API_KEY is not set")
    else:
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))