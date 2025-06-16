import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.llm.utils import _set_env

_set_env("GOOGLE_API_KEY")

def get_langchain_llm(model: str):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.2,
        max_tokens=2048,
        timeout=None,
        max_retries=2,
        top_p=0.8,
        top_k=40,
        verbose=True,
        )
    
    return llm