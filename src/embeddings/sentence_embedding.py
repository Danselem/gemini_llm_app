import os
from langchain_huggingface import HuggingFaceEmbeddings
from redisvl.utils.vectorize import HFTextVectorizer
from src.utils.setvars import _set_env

_set_env("HF_EMBEDDING_MODEL")

model = "all-MiniLM-L6-v2"


def get_sentence_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})

    return embeddings

def get_hf_embed():
    hf_embed = HFTextVectorizer(model=os.getenv("HF_EMBEDDING_MODEL"))
    return hf_embed