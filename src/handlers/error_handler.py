# src/handlers/error_handler.py

class APIError(Exception):
    """Generic API-related error."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ClientError(Exception):
    """Client-side error, typically from invalid API usage."""
    def __init__(self, status_code: int, details: dict):
        self.status_code = status_code
        self.details = details
        super().__init__(f"ClientError {status_code}: {details.get('message', str(details))}")
