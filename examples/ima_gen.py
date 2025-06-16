# src/llm/ima_gen.py

import io
from pathlib import Path
from typing import Union

from PIL import Image
from google.genai import types
from google.genai.errors import ClientError as GoogleClientError
from google.genai.types import GenerateContentResponse

from src.llm.gemini_client import get_client
from src.handlers.error_handler import APIError, ClientError
from src.utils.logger import logger

client = get_client()
model: str = 'imagen-3.0-generate-002'

def generate_and_save_image(prompt: str, output_path: Union[str, Path]) -> None:
    """
    Generate an image using the Gemini Imagen API and save it to the specified path.

    Args:
        prompt (str): The text prompt to generate the image.
        output_path (Union[str, Path]): The path where the generated image should be saved.

    Raises:
        ClientError: If the Google API returns a client error (e.g., billing required).
        APIError: For unexpected or generic API-related issues.
    """
    try:
        img_response: GenerateContentResponse = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type='image/jpeg'
            ),
        )

        if img_response.parts and img_response.parts[0].inline_data:
            image_data: bytes = img_response.parts[0].inline_data.data
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path)
            logger.info(f"Image saved to: {output_path}")
        else:
            logger.warning("No image data found in the response.")
            raise APIError("Empty response: No image data returned by the API.")

    except GoogleClientError as gce:
        logger.error(f"Google ClientError: {gce}")
        raise ClientError(400, {"message": str(gce)})

    except Exception as e:
        logger.exception("Unexpected error while generating image")
        raise APIError(str(e))

if __name__ == "__main__":
    prompt: str = 'An umbrella in the foreground, and a rainy night sky in the background'
    data_path: Path = Path("data/image/outputs")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        generate_and_save_image(prompt, data_path / "umbrella.png")
    except (APIError, ClientError) as err:
        logger.error(f"Image generation failed: {err}")
