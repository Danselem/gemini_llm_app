from pathlib import Path
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.utils.logger import logger

def load_documents_from_directory(docs_dir: Path) -> List[Document]:
    """
    Load all PDF documents from the specified directory in a reproducible order.

    Args:
        docs_dir (Path): Path to the directory containing PDF files.

    Returns:
        List[Document]: A list of loaded document chunks.
    """
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"The directory {docs_dir} does not exist or is not a directory.")
        raise ValueError(f"The directory {docs_dir} does not exist or is not a directory.")

    # Sort file paths to ensure consistent order across runs
    all_files = sorted(
        os.path.join(docs_dir, f) for f in os.listdir(docs_dir)
        if os.path.isfile(os.path.join(docs_dir, f))
    )

    documents = []
    for file_path in all_files:
        try:
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} chunks from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded total of {len(documents)} document chunks")
    return documents
