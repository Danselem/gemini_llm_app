from src.observability.arize_observability import init_langchain_observability
from src.rag.hybrid import run_hybrid_agent

init_langchain_observability()

run_hybrid_agent(
    question="Tell me about Figuring the EIC",
    output_name="eic",
)
