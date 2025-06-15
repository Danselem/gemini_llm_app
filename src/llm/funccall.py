from dotenv import load_dotenv
from google.genai import types
import random
from src.llm.gemini_client import get_client


load_dotenv()
model = "gemini-2.0-flash-001"

client = get_client()


prompt = "What is the weather like in Boston?"


def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """

    weather_conditions = [
        "sunny",
        "cloudy",
        "rainy",
        "windy",
        "snowy",
        "foggy",
        "stormy",
        "partly cloudy",
        "drizzling",
        "hailing",
        "thunderstorm",
        "overcast",
        "clear sky",
        "light rain",
        "heavy rain",
        "light snow",
        "heavy snow",
        "blizzard",
        "hurricane",
        "tornado",
    ]

    conditions = random.choice(weather_conditions)

    print(f"Function called: The weather for {location} is {conditions}")
    return conditions


response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="Speak like an excitable weather announcer.",
        tools=[get_current_weather],
    ),
)

print(response.text)

system_instruction = """
Use the configured function tools to lookup the weather.
Speak like an excitable weather announcer.
Return your final response in Markdown using bullets if multiple cities are requested.
"""

prompt="""
What is the weather like in Boston, Philadelphia, Nashville, Atlanta,
Houston, Chicago, Salt Lake City, Seattle, Las Vegas, and San Francisco?
"""

response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[get_current_weather],
    ),
)
print("---")
print(response.text)