from google.genai import types
from pathlib import Path
from typing import Union
from src.handlers.error_handler import APIError, ClientError
from src.observability.arize_observability import init_observability
from google.genai.errors import ClientError as GoogleClientError
from google.genai.types import GenerateContentResponse
from src.llm.gemini_client import get_client
from src.utils.logger import logger

init_observability()

MODEL_ID = "gemini-2.5-flash-preview-04-17"

client = get_client()

def thinking_response(prompt: str, output_path: Union[str, Path]) -> None:
    """
    Generate a thinking response of a text request.

    Args:
        prompt (str): The text prompt.
        output_path (Union[str, Path]): The path where the generated summary should be saved.

    Raises:
        ClientError: If the Google API returns a client error (e.g., billing required).
        APIError: For unexpected or generic API-related issues.
    """
    try:
        response: GenerateContentResponse = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024) # thinkingBudget must be an integer in the range 0 to 24576  
                ),
            )

        if response.text:
            with open(output_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Thinking response saved to: {output_path}")
        else:
            logger.warning("No data found in the response.")
            raise APIError("Empty response: No data returned by the API.")

    except GoogleClientError as gce:
        logger.error(f"Google ClientError: {gce}")
        raise ClientError(400, {"message": str(gce)})

    except Exception as e:
        logger.exception("Unexpected error while generating summary.")
        raise APIError(str(e))


if __name__ == "__main__":
    
    prompt = "Explain the process of photosynthesis in plants and provide a brief summary."

    data_path = Path("data/outputs")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        thinking_response(prompt, data_path / "think-output.txt")
    except (APIError, ClientError) as err:
        logger.error(f"Thinking generation failed: {err}")


