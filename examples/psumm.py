import requests
from pathlib import Path
from typing import Union
from google.genai.errors import ClientError as GoogleClientError
from google.genai.types import GenerateContentResponse

from src.llm.gemini_client import get_client
from src.handlers.error_handler import APIError, ClientError
from src.utils.logger import logger

client = get_client()

MODEL_ID: str = 'gemini-2.0-flash-001'


def generate_summary(prompt: str, output_path: Union[str, Path]) -> None:
    """
    Generate a summary of an arXiv article.

    Args:
        prompt (str): The multi-part prompt.
        output_path (Union[str, Path]): The path where the generated summary should be saved.

    Raises:
        ClientError: If the Google API returns a client error (e.g., billing required).
        APIError: For unexpected or generic API-related issues.
    """
    try:
        response: GenerateContentResponse = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
            )

        if response.text:
            with open(output_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Summary saved to: {output_path}")
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
    html_url: str ="https://arxiv.org/html/2505.22139v1"
    
    # Get the contents of the HTML URL using the Python Requests library
    content = requests.get(html_url).content

    # Create a multi-part prompt
    prompt = [
    content,
    "Summarize the content in not more than 200 words."
    ]
    
    data_path: Path = Path("data/outputs")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        generate_summary(prompt, data_path / "SMGPS-stars.txt")
    except (APIError, ClientError) as err:
        logger.error(f"Summary generation failed: {err}")