# Define the set of tools our agent will use
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilySearch


from src.utils.setvars import _set_env

_set_env("TAVILY_API_KEY")


def vector_search(
    retriever,
    description,
    name: str = "vector_search",
):

    # Define the vector search tool
    vector_search_tool = retriever.as_tool(name=name, description=description)

    return vector_search_tool


# Define the LangChain search tool
search = TavilySearch(max_results=10, topic="general")

# Define the LangChain extract tool
extract = TavilyExtract(extract_depth="advanced")

# Define the LangChain crawl tool
crawl = TavilyCrawl()

# Define the DuckDuckGo search tool
duck_search = DuckDuckGoSearchRun()
