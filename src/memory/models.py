import ulid

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """
    Defines the type of long-term memory for categorization and retrieval.

    EPISODIC: Personal experiences and user-specific preferences
              (e.g., "User prefers Delta airlines", "User visited Paris last year")

    SEMANTIC: General domain knowledge and facts
              (e.g., "Singapore requires passport", "Tokyo has excellent public transit")

    The type of a long-term memory.

    EPISODIC: User specific experiences and preferences

    SEMANTIC: General knowledge on top of the user's preferences and LLM's
    training data.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    """Represents a single long-term memory."""

    content: str
    memory_type: MemoryType
    metadata: str


class Memories(BaseModel):
    """
    A list of memories extracted from a conversation by an LLM.

    NOTE: OpenAI's structured output requires us to wrap the list in an object.
    """

    memories: List[Memory]


class StoredMemory(Memory):
    """A stored long-term memory"""

    id: str  # The redis key
    memory_id: ulid.ULID = Field(default_factory=lambda: ulid.ULID())
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None