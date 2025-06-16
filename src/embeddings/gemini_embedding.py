from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.llm.utils import _set_env

_set_env("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_NUM_BATCH = 5

def get_google_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        )
 
    return embeddings