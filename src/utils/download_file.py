import requests
from pathlib import Path


def download_file(url: str, dest_folder: Path) -> None:
    """
    Download a file from a URL into a destination folder.

    Args:
        url (str): The URL to download from.
        dest_folder (Path): The directory where the file should be saved.

    Raises:
        requests.exceptions.RequestException: If the download fails.
    """
    dest_folder.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_folder / filename

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded to: {dest_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        raise
