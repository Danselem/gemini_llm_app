from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
from embeddings.sentence_embedding import get_hf_embed
from typing import Optional
from src.memory.models import MemoryType
from src.memory.storage import long_term_memory_index
from src.utils.logger import logger

hf_embed = get_hf_embed()


# If we have any memories that aren't associated with a user, we'll use this ID.
SYSTEM_USER_ID = "system"


def similar_memory_exists(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1,
) -> bool:
    """Check if a similar long-term memory already exists in Redis."""
    content_embedding = hf_embed.embed(content)

    filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type)

    if thread_id:
        filters = filters & (Tag("thread_id") == thread_id)

    # Search for similar memories
    vector_query = VectorRangeQuery(
        vector=content_embedding,
        num_results=1,
        vector_field_name="embedding",
        filter_expression=filters,
        distance_threshold=distance_threshold,
        return_fields=["id"],
    )
    results = long_term_memory_index.query(vector_query)
    logger.debug(f"Similar memory search results: {results}")

    if results:
        logger.debug(
            f"{len(results)} similar {'memory' if results.count == 1 else 'memories'} found. First: "
            f"{results[0]['id']}. Skipping storage."
        )
        return True

    return False