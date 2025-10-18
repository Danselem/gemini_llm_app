from typing import Any, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.logger import logger


# Pydantic configs allow arbitrary types (retriever/llm objects)
class MultiQueryConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any
    llm: Any

    @field_validator("retriever", mode="after")
    def validate_retriever(cls, v):
        if v is None:
            raise ValueError("retriever must be provided")
        return v

    @field_validator("llm", mode="after")
    def validate_llm(cls, v):
        if v is None:
            raise ValueError("llm must be provided")
        return v


class FlashrankConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any
    flashrank_threshold: float = Field(0.5, ge=0.0, le=1.0)
    top_n: int = Field(5, ge=1)

    @field_validator("retriever", mode="after")
    def validate_retriever(cls, v):
        if v is None:
            raise ValueError("retriever must be provided")
        return v


class RankLLMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any
    rank_llm: str = "zephyr"
    top_n: int = Field(5, ge=1)

    @field_validator("retriever", mode="after")
    def validate_retriever(cls, v):
        if v is None:
            raise ValueError("retriever must be provided")
        return v


def multi_query_retriever(
    retriever,
    llm,
):
    """
    Creates a MultiQueryRetriever with the specified retriever and LLM.

    Args:
        retriever: The base retriever to use.
        llm: The language model to use for generating queries.

    Returns:
        A MultiQueryRetriever instance.
    """
    cfg = MultiQueryConfig(retriever=retriever, llm=llm)

    try:
        mqr = MultiQueryRetriever.from_llm(
            retriever=cfg.retriever,
            llm=cfg.llm,
        )
    except Exception as e:
        logger.exception("Failed to create MultiQueryRetriever: %s", e)
        raise

    return mqr


def flashrank_retriever(
    retriever, flashrank_threshold: Optional[float] = 0.5, top_n: Optional[int] = 5
):
    """
    Compresses the retriever using FlashrankRerank.

    Args:
        retriever: The base retriever to compress.
        flashrank_threshold: Threshold for FlashrankRerank (0.0 - 1.0).
        top_n: Number of top documents to consider.

    Returns:
        A compressed retriever.
    """
    try:
        cfg = FlashrankConfig(
            retriever=retriever, flashrank_threshold=flashrank_threshold, top_n=top_n
        )

        compressor = FlashrankRerank(
            top_n=cfg.top_n, score_thresholds=cfg.flashrank_threshold
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=cfg.retriever
        )
    except Exception as e:
        logger.exception("flashrank_retriever failed: %s", e)
        raise

    return compression_retriever


def rankllm_retriever(
    retriever,
    rank_llm: Optional[str] = "zephyr",
    top_n: Optional[int] = 5,
):
    """
    Compresses the retriever using RankLLMRerank.

    Args:
        retriever: The base retriever to compress.
        rank_llm: The RankLLM model name to use for reranking.
        top_n: Number of top documents to consider.

    Returns:
        A compressed retriever.
    """
    try:
        cfg = RankLLMConfig(retriever=retriever, rank_llm=rank_llm, top_n=top_n)

        compressor = RankLLMRerank(
            model=cfg.rank_llm,
            top_n=cfg.top_n,
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=cfg.retriever
        )
    except Exception as e:
        logger.exception("rankllm_retriever failed: %s", e)
        raise

    return compression_retriever
