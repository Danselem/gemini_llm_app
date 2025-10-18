from datetime import datetime
from typing import List, Optional, Union
import ulid
from src.memory.models import MemoryType
from src.memory.models import StoredMemory
from src.memory.sim_mem import SYSTEM_USER_ID, similar_memory_exists, hf_embed
from src.memory.storage import long_term_memory_index
from redisvl.query import VectorRangeQuery
from src.utils.logger import logger


def store_memory(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    metadata: Optional[str] = None,
):
    """Store a long-term memory in Redis with deduplication.

        This function:
        1. Checks for similar existing memories to avoid duplicates
        2. Generates vector embeddings for semantic search
        3. Stores the memory with metadata for retrieval
        """
    if metadata is None:
        metadata = "{}"

    logger.info(f"Preparing to store memory: {content}")

    if similar_memory_exists(content, memory_type, user_id, thread_id):
        logger.info("Similar memory found, skipping storage")
        return

    embedding = hf_embed.embed(content)

    memory_data = {
        "user_id": user_id or SYSTEM_USER_ID,
        "content": content,
        "memory_type": memory_type.value,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "embedding": embedding,
        "memory_id": str(ulid.ULID()),
        "thread_id": thread_id,
    }

    try:
        long_term_memory_index.load([memory_data])
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return

    logger.info(f"Stored {memory_type} memory: {content}")


def retrieve_memories(
    query: str,
    memory_type: Union[Optional[MemoryType], List[MemoryType]] = None,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1,
    limit: int = 5,
) -> List[StoredMemory]:
    """Retrieve relevant memories from Redis using vector similarity search.

    """
    # Create vector query using query embedding
    logger.debug(f"Retrieving memories for query: {query}")
    vector_query = VectorRangeQuery(
        vector=hf_embed.embed(query),
        return_fields=[
            "content",
            "memory_type", 
            "metadata",
            "created_at",
            "memory_id",
            "thread_id",
            "user_id",
        ],
        num_results=limit,
        vector_field_name="embedding",
        dialect=2,
        distance_threshold=distance_threshold,
    )

    # Build filter conditions
    base_filters = [f"@user_id:{{{user_id or SYSTEM_USER_ID}}}"]

    if memory_type:
        if isinstance(memory_type, list):
            base_filters.append(f"@memory_type:{{{'|'.join(memory_type)}}}")
        else:
            base_filters.append(f"@memory_type:{{{memory_type.value}}}")

    if thread_id:
        base_filters.append(f"@thread_id:{{{thread_id}}}")

    vector_query.set_filter(" ".join(base_filters))

    # Execute vector similarity search
    results = long_term_memory_index.query(vector_query)

    # Parse results into StoredMemory objects
    memories = []
    for doc in results:
        try:
            memory = StoredMemory(
                id=doc["id"],
                memory_id=doc["memory_id"],
                user_id=doc["user_id"],
                thread_id=doc.get("thread_id", None),
                memory_type=MemoryType(doc["memory_type"]),
                content=doc["content"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
            )
            memories.append(memory)
        except Exception as e:
            logger.error(f"Error parsing memory: {e}")
            continue
    return memories