from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  # or langchain_core.documents.Document


def get_split_documents(index_path: Path) -> List[Document]:
    """
    Load and chunk documents from the specified directory.

    Args:
        index_path (Path): Path to the folder containing documents (.pdf or text).

    Returns:
        List[Document]: Chunked documents ready for embedding or indexing.
    """
    split_docs: List[Document] = []

    for file_path in index_path.iterdir():
        try:
            print(f"Processing: {file_path.name}")
            if file_path.suffix == ".pdf":
                loader = UnstructuredPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            text_splitter = CharacterTextSplitter(chunk_size=8192, chunk_overlap=128)
            split_docs.extend(text_splitter.split_documents(loader.load()))

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    return split_docs
