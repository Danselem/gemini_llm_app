from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.embeddings.sentence_embedding import get_sentence_embeddings
from src.llm.lang_gemini import get_gemini_llm
from src.prompt_engineering.prompt import PROMPT_TEMPLATE
from src.utils.logger import logger
from src.vectors.chroma_vector import get_chroma_load

embeddings = get_sentence_embeddings()

llm = get_gemini_llm(model="gemini-2.0-flash")

retriever = get_chroma_load(
    embeddings=embeddings,
    directory=Path("storage/chroma"),
    collection_name="pdf",
)

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "Summarise the document in the knowledge base?"
res = rag_chain.invoke(query)
print(res)
