# Initialize the vector store
import hashlib
from pathlib import Path
from typing import Any, List, Union

from langchain.schema import Document
from langchain_chroma import Chroma
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.logger import logger


def generate_id_from_content(content: str) -> str:
    """
    Generate a deterministic hash-based ID for a document chunk.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class DocumentChunk(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)


class ChromaIngestModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Any
    chunks: List[Union[Document, DocumentChunk, dict]]
    directory: Path
    collection_name: str

    @field_validator("chunks", mode="after")
    def chunks_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("No document chunks provided to create the vector store.")
        return v

    @field_validator("embeddings", mode="after")
    def embeddings_must_be_present(cls, v):
        if v is None:
            raise ValueError("Embeddings model is not provided.")
        return v

    @field_validator("directory", mode="before")
    def ensure_path(cls, v):
        return Path(v)


class ChromaLoadModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Any
    directory: Path
    collection_name: str

    @field_validator("embeddings", mode="after")
    def embeddings_present(cls, v):
        if v is None:
            raise ValueError("Embeddings model is not provided.")
        return v

    @field_validator("directory", mode="before")
    def ensure_path(cls, v):
        return Path(v)


def _normalize_chunks(
    chunks: List[Union[Document, DocumentChunk, dict]],
) -> List[Document]:
    """
    Normalize incoming chunk representations into langchain.schema.Document objects.
    Accepts existing Document instances, dicts with 'page_content'/'metadata', or DocumentChunk.
    """
    normalized: List[Document] = []
    for c in chunks:
        if isinstance(c, Document):
            normalized.append(c)
            continue
        if isinstance(c, DocumentChunk):
            normalized.append(
                Document(page_content=c.page_content, metadata=c.metadata)
            )
            continue
        if isinstance(c, dict):
            page_content = c.get("page_content") or c.get("text") or ""
            metadata = c.get("metadata", {})
            normalized.append(Document(page_content=page_content, metadata=metadata))
            continue
        # Fallback: duck-type objects with page_content attribute
        if hasattr(c, "page_content"):
            metadata = getattr(c, "metadata", {})
            normalized.append(
                Document(page_content=getattr(c, "page_content"), metadata=metadata)
            )
            continue
        raise ValueError(f"Unsupported chunk type: {type(c)}")
    return normalized


def get_chroma_ingest(
    embeddings: Any,
    chunks: List[Any],
    directory: Union[str, Path],
    collection_name: str,
) -> Chroma:
    """
    Create a Chroma vector store from document chunks, avoiding duplicate ingestion.
    Validation is performed via pydantic models to ensure inputs meet expectations.
    """
    config = ChromaIngestModel(
        embeddings=embeddings,
        chunks=chunks,
        directory=directory,
        collection_name=collection_name,
    )

    vectordb = Chroma(
        embedding_function=config.embeddings,
        persist_directory=config.directory.as_posix(),
        collection_name=config.collection_name,
    )

    normalized_chunks = _normalize_chunks(config.chunks)

    # Create deterministic content-based IDs
    chunk_ids = [
        generate_id_from_content(doc.page_content) for doc in normalized_chunks
    ]

    # Check which IDs already exist (vectordb.get() usually returns {"ids": [...]})
    try:
        existing = vectordb.get()
        existing_ids = set(existing.get("ids", []))
    except Exception as e:
        logger.debug("Could not read existing ids from Chroma vector store: %s", e)
        existing_ids = set()

    # Filter out existing chunks
    new_chunks = []
    new_ids = []
    for doc, doc_id in zip(normalized_chunks, chunk_ids):
        if doc_id not in existing_ids:
            new_chunks.append(doc)
            new_ids.append(doc_id)

    # Add only new chunks
    if new_chunks:
        vectordb.add_documents(new_chunks, ids=new_ids)
        try:
            vectordb.persist()
        except Exception:
            # Persist may not be required depending on how Chroma is configured
            logger.debug("Chroma persist() failed or is unnecessary.")


def get_chroma_load(collection_name: str, embeddings: Any, directory: Union[str, Path]):
    """
    Load an existing Chroma collection and return a retriever.
    Inputs validated with pydantic.
    """
    config = ChromaLoadModel(
        collection_name=collection_name, embeddings=embeddings, directory=directory
    )

    vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=config.embeddings,
        persist_directory=config.directory.as_posix(),
    )

    retriever = vector_store.as_retriever()
    return retriever
