from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from src.utils import get_redis_client
from src.utils.logger import logger

redis_client = get_redis_client()

# Define the schema for our vector search index
# This creates the structure for storing and querying memories
memory_schema = IndexSchema.from_dict({
        "index": {
            "name": "agent_memories",  # Index name for identification
            "prefix": "memory",       # Redis key prefix (memory:1, memory:2, etc.)
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "memory_type", "type": "tag"},
            {"name": "metadata", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "user_id", "type": "tag"},
            {"name": "memory_id", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 768,  # HF model embedding dimension
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
)


try:
    long_term_memory_index = SearchIndex(
        schema=memory_schema,
        redis_client=redis_client,
        validate_on_load=True
    )
    long_term_memory_index.create(overwrite=True)
    logger.info("Long-term memory index ready")
except Exception as e:
    logger.debug(f"Error creating index: {e}")