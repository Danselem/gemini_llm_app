import os
from pathlib import Path
from dotenv import load_dotenv

def _set_env(var: str):
    # Unset the environment variable if it's already set
    if var in os.environ:
        del os.environ[var]

    # Load environment variables from the .env file
    dotenv_path = Path('./.env')
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Retrieve the value of the environment variable
    value = os.getenv(var)

    if value is None:
        raise ValueError(f"The environment variable '{var}' is not set in the .env file or system environment.")

    # Set the environment variable explicitly
    os.environ[var] = value
