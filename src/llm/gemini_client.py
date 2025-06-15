import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def get_client():
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("GOOGLE_API_KEY is not set")
    else:
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))