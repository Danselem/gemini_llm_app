import argparse
from pathlib import Path
from typing import Optional

from src.embeddings.gemini_embedding import get_google_embeddings
from src.embeddings.sentence_embedding import get_sentence_embeddings
from src.utils.doc_loader import load_documents_from_directory
from src.utils.doc_split import split_documents
from src.vectors.chroma_vector import get_chroma_ingest


def ingest(
    data_dir: Path = Path("data/pdfs"),
    persist_path: Path = Path("storage/chroma"),
    collection_name: str = "pdf",
    embeddings_model: Optional[object] = None,
):
    """
    Ingest documents into a Chroma vector database.

    Args:
        data_dir (Path): Directory containing documents to ingest.
        persist_path (Path): Directory to persist the Chroma DB.
        collection_name (str): Name of the Chroma collection/index.
        embeddings_model (Optional[object]): Preloaded embedding model. If None, loads Gemini embeddings.
    """
    embeddings = (
        embeddings_model or get_sentence_embeddings()
    )  # get_google_embeddings()
    persist_path.mkdir(parents=True, exist_ok=True)

    documents = load_documents_from_directory(docs_dir=data_dir)
    chunks = split_documents(documents)

    get_chroma_ingest(
        embeddings=embeddings,
        chunks=chunks,
        directory=str(persist_path),
        collection_name=collection_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into a Chroma vector store."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/pdfs",
        help="Directory containing documents.",
    )
    parser.add_argument(
        "--persist_path",
        type=str,
        default="storage/chroma",
        help="Directory to store the Chroma DB.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="pdf",
        help="Collection name for the vector store.",
    )

    args = parser.parse_args()

    ingest(
        data_dir=Path(args.data_dir),
        persist_path=Path(args.persist_path),
        collection_name=args.collection_name,
    )
