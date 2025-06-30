from langchain_huggingface import HuggingFaceEmbeddings

model = "all-MiniLM-L6-v2"

def get_sentence_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
 
    return embeddings