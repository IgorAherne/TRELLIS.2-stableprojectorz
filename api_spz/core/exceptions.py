# api_spz/core/exceptions.py
# Exceptions used by the API, of trellis2-stable-projectorz

class CancelledException(Exception):
    """Raised when user requests cancellation (interrupts the generation)."""
    pass