# examples/rag/app.py

from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

from src.embeddings.sentence_embedding import get_sentence_embeddings
from src.llm.gemini_client import get_client
from src.llm.lang_gemini import get_gemini_llm
from src.observability.arize_observability import init_langchain_observability
from src.prompt_engineering.templates import ai_assistant_template
from src.utils.doc_loader import load_documents_from_directory
from src.utils.doc_split import split_documents
from src.utils.download_file import download_file
from src.vectors.chroma_vector import get_chroma_ingest, get_chroma_load

init_langchain_observability()


def main() -> None:
    index_path: Path = Path("data/pdfs")
    index_path.mkdir(parents=True, exist_ok=True)

    persist_path: Path = Path("storage/chroma")
    persist_path.mkdir(parents=True, exist_ok=True)

    pdf_url: str = "https://www.irs.gov/pub/irs-pdf/p554.pdf"
    pdf_file_path: Path = index_path / "p554.pdf"

    if not pdf_file_path.exists():
        download_file(pdf_url, index_path)

    # Initialize components
    client = get_client()
    llm = get_gemini_llm(model="gemini-2.0-flash")
    # embeddings = get_google_embeddings()
    embeddings = get_sentence_embeddings()

    # Load documents
    documents = load_documents_from_directory(docs_dir=index_path)

    # split documents
    chunks = split_documents(documents)

    # Vector store
    get_chroma_ingest(
        embeddings=embeddings,
        chunks=chunks,
        directory=str(persist_path),
        collection_name="pdf",
    )

    # Retriever and chain setup
    vector_store = get_chroma_load(
        embeddings=embeddings, directory=persist_path, collection_name="pdf"
    )
    retriever = vector_store.as_retriever()

    prompt = PromptTemplate.from_template(ai_assistant_template)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Query and print result
    response = retrieval_chain.invoke({"input": "Tell me about Figuring the EIC."})
    print(response["answer"])


if __name__ == "__main__":
    main()
