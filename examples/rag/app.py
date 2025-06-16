# examples/rag/app.py

from pathlib import Path
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from src.utils.doc_split import get_split_documents
from src.embeddings.gemini_embedding import get_google_embeddings
from src.llm.gemini_client import get_client
from src.llm.lang_gemini import get_langchain_llm
from src.prompt_engineering.templates import ai_assistant_template
from src.utils.download_file import download_file


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
    llm = get_langchain_llm(model="gemini-2.0-flash")
    embeddings = get_google_embeddings()

    # Load and split documents
    split_docs = get_split_documents(index_path)

    # Vector store
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    db.persist()

    # Retriever and chain setup
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(ai_assistant_template)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Query and print result
    response = retrieval_chain.invoke({"input": "Tell me about Figuring the EIC."})
    print(response["answer"])


if __name__ == "__main__":
    main()
