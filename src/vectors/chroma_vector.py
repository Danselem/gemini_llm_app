# Initialize the vector store
from pathlib import Path
import hashlib
from uuid import uuid4
from langchain_chroma import Chroma
from src.utils.logger import logger

# Initialize the embeddings model with the default model
# embeddings = get_google_embeddings()


def generate_id_from_content(content: str) -> str:
    """
    Generate a deterministic hash-based ID for a document chunk.
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def get_chroma_ingest(embeddings, chunks, directory, collection_name):
    """
    Create a Chroma vector store from document chunks, avoiding duplicate ingestion.

    Args:
        embeddings: The embedding model to use.
        chunks (list): List of document chunks to be indexed.
        directory (str): Directory to persist the Chroma vector store.
        collection_name (str): Name of the vector collection.

    Returns:
        Chroma vector store instance.
    """
    if not chunks:
        raise ValueError("No document chunks provided to create the vector store.")
    if embeddings is None:
        raise ValueError("Embeddings model is not provided.")

    # Initialize the vector DB (load if already exists)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=directory,
        collection_name=collection_name
    )

    # Create content-based IDs
    chunk_ids = [generate_id_from_content(doc.page_content) for doc in chunks]

    # Check which IDs already exist
    existing_ids = set(vectordb.get()["ids"])
    
    # Filter out existing chunks
    new_chunks = []
    new_ids = []
    for doc, doc_id in zip(chunks, chunk_ids):
        if doc_id not in existing_ids:
            new_chunks.append(doc)
            new_ids.append(doc_id)

    # Add only new chunks
    if new_chunks:
        vectordb.add_documents(new_chunks, ids=new_ids)

    return vectordb




def get_chroma_load(collection_name: str, embeddings, directory: Path):

    # Initialize the vector store
    vector_store = Chroma(
        collection_name=collection_name,  # Use the collection name to load the specific collection
        embedding_function=embeddings,
        persist_directory=directory.as_posix(),
        )
    
    # Initialize the retriever
    retriever = vector_store.as_retriever()
    return retriever