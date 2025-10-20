
from pathlib import Path
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import (AIMessage, ToolMessage,
                                     SystemMessage, HumanMessage)
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from src.llm.lang_gemini import get_gemini_llm
from src.vectors.chroma_vector import get_chroma_load
from src.embeddings.sentence_embedding import get_sentence_embeddings
from src.memory.tools import store_memory_tool, retrieve_memories_tool
from src.memory.conversation import summarize_conversation
from src.memory.utils import RuntimeState, get_redis_client
from src.utils.logger import logger


# Set up the Redis checkpointer for short term memory
redis_client = get_redis_client()
redis_saver = RedisSaver(redis_client=redis_client)
redis_saver.setup()

# Initialize LLM
llm = get_gemini_llm(model="gemini-2.0-flash")

# Initialize vector store retriever
embeddings = get_sentence_embeddings()

vector_store = get_chroma_load(
    embeddings=embeddings,
    directory=Path("storage/chroma"),
    collection_name="pdf",
)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Define the set of tools
tools = [store_memory_tool, retrieve_memories_tool, retrieve, query_or_respond]


# Define the travel agent
rag_agent = create_react_agent(
    model=llm,
    tools=tools,               # Long-term memory: provided as a set of custom tools
    checkpointer=redis_saver,  # Short-term memory: the conversation history
    prompt=SystemMessage(
        content="""
        You are a RAG assistant for retrieving info from knowledge base for 
        question-answering tasks.
        Use the following pieces of retrieved context to answer 
        the question. If you don't know the answer, say that you
        don't know. Use three sentences maximum and keep the 
        answer concise.

        You have access to the following types of memory:
        1. Short-term memory: The current conversation thread
        2. Long-term memory:
           - Episodic: User preferences and past trip experiences (e.g., "User prefers window seats")
           - Semantic: General knowledge about travel destinations and requirements

        Your procedural knowledge (how to search, knowledge base, etc.) is built into your tools and prompts.

        Always be helpful, personal, and context-aware in your responses.
        """
    ),
)

def respond_to_user(state: RuntimeState, 
                    config: RunnableConfig) -> RuntimeState:
    """Invoke the RAG agent to generate a response."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        logger.warning("No HumanMessage found in state")
        return state

    try:
        # Single agent invocation, not streamed (simplified for reliability)
        result = rag_agent.invoke({"messages": state["messages"]}, config=config)
        agent_message = result["messages"][-1]
        state["messages"].append(agent_message)
    except Exception as e:
        logger.error(f"Error invoking travel agent: {e}")
        agent_message = AIMessage(
            content="I'm sorry, I encountered an error processing your request."
        )
        state["messages"].append(agent_message)

    return state


def execute_tools(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    """Execute tools specified in the latest AIMessage and append ToolMessages."""
    messages = state["messages"]
    latest_ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None
    )

    if not latest_ai_message:
        return state  # No tool calls to process

    tool_messages = []
    for tool_call in latest_ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Find the corresponding tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            continue  # Skip if tool not found

        try:
            # Execute the tool with the provided arguments
            result = tool.invoke(tool_args, config=config)
            # Create a ToolMessage with the result
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
        except Exception as e:
            # Handle tool execution errors
            error_message = ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(error_message)

    # Append the ToolMessages to the message history
    messages.extend(tool_messages)
    state["messages"] = messages
    return state


workflow = StateGraph(RuntimeState)

# Add nodes to the graph
workflow.add_node("agent", respond_to_user)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("summarize_conversation", summarize_conversation)

def decide_next_step(state):
    latest_ai_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if latest_ai_message and latest_ai_message.tool_calls:
        return "execute_tools"
    return "summarize_conversation"


workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    decide_next_step,
    {"execute_tools": "execute_tools", "summarize_conversation": "summarize_conversation"},
)
workflow.add_edge("execute_tools", "agent")
workflow.add_edge("summarize_conversation", END)

graph = workflow.compile(checkpointer=redis_saver)