import os
from redis import Redis
from langgraph.graph.message import MessagesState
from src.utils.setvars import _set_env

_set_env("REDIS_URL")


def get_redis_client():
    _set_env("REDIS_URL")
    redis_client = Redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
    return redis_client

class RuntimeState(MessagesState):
    """Runtime state for the travel agent."""
    pass