import sys
from loguru import logger
from typing import Optional, Dict



def setup_logging(
        stdout_level: str = "INFO",
        stdout_format: Optional[str] = None,
        additional_sinks: Optional[Dict] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        stdout_level: Logging level for stdout ("INFO", "DEBUG", etc.)
        stdout_format: Custom format for stdout logging
        additional_sinks: Dictionary of additional log sinks with their configurations
    """
    # Remove default logger
    logger.remove()

    # Default stdout format if none provided
    default_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Add stdout handler
    logger.add(
        sys.stdout,
        colorize=True,
        format=stdout_format or default_format,
        level=stdout_level
    )

    # Add additional sinks if provided
    if additional_sinks:
        for sink, config in additional_sinks.items():
            logger.add(sink, **config)


# Example usage of additional sinks
file_config = {
    "logs/patent_service.log": {
        "rotation": "500 MB",
        "retention": "10 days",
        "compression": "zip",
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        "level": "DEBUG"
    }
}
