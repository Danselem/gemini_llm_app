import datetime
from pathlib import Path

from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from src.agent.tools import crawl, extract, search, vector_search
from src.embeddings.sentence_embedding import get_sentence_embeddings
from src.llm.lang_gemini import get_gemini_llm
from src.prompt_engineering.templates import ai_assistant_template
from src.retrievers.retriever import multi_query_retriever
from src.utils.logger import logger
from src.vectors.chroma_vector import get_chroma_load


def run_hybrid_agent(question: str, output_name: str):
    """
    Runs the hybrid agent with the provided question and saves the response as a .txt file.

    Args:
        question (str): The user input to be processed by the agent.
        output_name (str): The name (without extension) to use for the output .txt file.

    Raises:
        Exception: Logs any errors encountered during processing or file writing.
    """
    try:
        # Get today's date for dynamic prompt
        today = datetime.datetime.today().strftime("%A, %I, %M, %p, %B %d, %Y")

        # Initialize components
        google_llm = get_gemini_llm(model="gemini-2.0-flash")
        embeddings = get_sentence_embeddings()
        vector_store = get_chroma_load(
            embeddings=embeddings,
            directory=Path("storage/chroma"),
            collection_name="pdf",
        )

        retriever = vector_store.as_retriever()

        m_retriever = multi_query_retriever(retriever=retriever, llm=google_llm)

        vector_search_tool = vector_search(
            retriever=m_retriever,
            name="vector_search",
            description=f"Searches the vector database for relevant information. Today's date is {today}.",
        )

        # Create the hybrid agent
        system_message = f"Today is {today}\n" + ai_assistant_template
        hybrid_agent = create_react_agent(
            model=google_llm,
            tools=[
                search,
                crawl,
                extract,
                vector_search_tool,
            ],
            prompt=ChatPromptTemplate.from_messages(
                [
                    # ("system", "Today is {today}\n" + ai_assistance_template),
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            ),
            name="hybrid_agent",
        )

        # Prepare the input
        inputs = {
            "messages": [HumanMessage(content=question)],
            # "today": today
        }

        # Collect the final response
        final_response = ""
        for s in hybrid_agent.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if not isinstance(message, tuple):
                final_response = message.content

        # Save the response as a .tex file
        output_dir = Path("data/output/hybrid")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{output_name}.txt"

        with output_file.open("w", encoding="utf-8") as f:
            f.write(final_response)

        logger.info(f"Output successfully saved to {output_file}")
        print(f"Saved output to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to run agent or save output: {e}")
        raise
