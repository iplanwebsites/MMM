"""MMM model package."""

from .config import InferenceConfig
from .inference import generate

__all__ = [
    "generate",
    "InferenceConfig",
]
