from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever


def flashrank_retriever(
    retriever,
    flashrank_threshold=0.5
    ):
    """
    Compresses the retriever using either RankLLMRerank or FlashrankRerank.
    
    Args:
        retriever: The base retriever to compress.
        llm: The language model to use for compression.
        rank_llm: Optional; the RankLLM to use for reranking.
        flashrank_llm: Optional; the Flashrank LLM to use for reranking.
        rank_threshold: Threshold for RankLLMRerank.
        flashrank_threshold: Threshold for FlashrankRerank.
    
    Returns:
        A compressed retriever.
    """
    compressor = FlashrankRerank(top_n=5, score_thresholds=flashrank_threshold)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    
    return compression_retriever  # No compression applied


def rankllm_retriever(
    retriever,
    rank_llm="zephyr",
    ):
    """
    Compresses the retriever using RankLLMRerank.
    
    Args:
        retriever: The base retriever to compress.
        rank_llm: The RankLLM to use for reranking.
        rank_threshold: Threshold for RankLLMRerank.
    
    Returns:
        A compressed retriever.
    """
    compressor = RankLLMRerank(model=rank_llm, top_n=5,)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    
    return compression_retriever  # No compression applied  


def multi_query_retriever(
    retriever,
    llm,
    ):
    """
    Creates a MultiQueryRetriever with the specified retriever and LLM.
    
    Args:
        retriever: The base retriever to use.
        llm: The language model to use for generating queries.
        top_k: Number of top results to return.
    
    Returns:
        A MultiQueryRetriever instance.
    """
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm,
        )
    
    return multi_query_retriever