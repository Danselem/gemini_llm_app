from dotenv import load_dotenv
from google.genai import types
from src.llm.gemini_client import get_client
from typing import List
from pydantic import BaseModel, Field

load_dotenv()
model = "gemini-2.0-flash-001"

client = get_client()



class Planet(BaseModel):
  name: str = Field(description="The name of the planet")
  moons: int = Field(description="The number of moons this planet contains or '0' if none.", default=0)

class SolarSystem(BaseModel):
    planets: List[Planet] = Field(..., description="A python list of all planets in the solar system")

# Display the model's JSON schema
print(SolarSystem.model_json_schema())

solar_system=SolarSystem(planets=[Planet(name="Earth", moons=1)])

print(solar_system.model_dump_json())

prompt = "List all the planets in the solar system and the number of moons."

response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=SolarSystem,
    ),
)

solar_system = response.parsed
print(f"solar_system is an instance of: {solar_system.__class__.__name__}")
print("---")

# We can now iterate through all the planets
print("Planet List:")
[print(f"{p.name}, {p.moons} moons") for p in solar_system.planets]
