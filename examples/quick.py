from dotenv import load_dotenv
from src.llm.gemini_client import get_client

load_dotenv()
model = "gemini-2.0-flash-001"

client = get_client()

prompt = "What's the largest planet in our solar system? Tell me 3 facts about it."

response = client.models.generate_content(
    model=model,
    contents=prompt
)

print(response.text)

print(response.model_dump_json(indent=2))


prompt2 = "Tell me 3 facts about Jupiter and its moons"

chat = client.chats.create(model=model)
for chunk in chat.send_message_stream(prompt2):
    print(chunk.text, end="")

for chunk in chat.send_message_stream("Tell me more..."):
    print(chunk.text, end="")





