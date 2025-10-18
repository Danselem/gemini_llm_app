from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import MessagesState
from src.llm.lang_gemini import get_langchain_llm
from src.utils import RuntimeState
from src.utils.logger import logger


# An LLM configured for summarization.
summarizer = get_langchain_llm("gemini-2.0-flash")

# The number of messages after which we'll summarize the conversation.
MESSAGE_SUMMARIZATION_THRESHOLD = 6


def summarize_conversation(
    state: RuntimeState, config: RunnableConfig
) -> RuntimeState:
    """
    Summarize a list of messages into a concise summary to reduce context length
    while preserving important information.
    """
    messages = state["messages"]
    current_message_count = len(messages)
    if current_message_count < MESSAGE_SUMMARIZATION_THRESHOLD:
        logger.debug(f"Not summarizing conversation: {current_message_count}")
        return state

    system_prompt = """
    You are a conversation summarizer. Create a concise summary of the previous
    conversation between a user and a travel assistant.

    The summary should:
    1. Highlight key topics, preferences, and decisions
    2. Include any specific trip details (destinations, dates, preferences)
    3. Note any outstanding questions or topics that need follow-up
    4. Be concise but informative

    Format your summary as a brief narrative paragraph.
    """

    message_content = "\n".join(
        [
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ]
    )

    # Invoke the summarizer
    summary_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Please summarize this conversation:\n\n{message_content}"
        ),
    ]

    summary_response = summarizer.invoke(summary_messages)

    logger.info(f"Summarized {len(messages)} messages into a conversation summary")

    summary_message = SystemMessage(
        content=f"""
        Summary of the conversation so far:

        {summary_response.content}

        Please continue the conversation based on this summary and the recent messages.
        """
    )
    remove_messages = [
        RemoveMessage(id=msg.id) for msg in messages if msg.id is not None
    ]

    state["messages"] = [  # type: ignore
        *remove_messages,
        summary_message,
        state["messages"][-1],
    ]

    return state.copy()

