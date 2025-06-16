ai_assistant_template = """
You are a helpful AI assistant. You're tasked to answer the question given below, but only based on the context provided.
context:

{context}


question:

{input}


If you cannot find an answer ask the user to rephrase the question.
answer:

"""