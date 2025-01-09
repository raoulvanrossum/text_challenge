"""
Tests for the text processor module.
"""

import pytest
from src.patent_search.core.processor import TextProcessor, ProcessedText
from src.patent_search.config import MODEL_NAME

@pytest.fixture
def processor():
    """Create a TextProcessor instance for testing."""
    return TextProcessor(model_name=MODEL_NAME)

def test_clean_text(processor):
    """Test text cleaning functionality and punctuation handling."""
    # Test basic cleaning
    text = "  This   is  a    test   "
    assert processor.clean_text(text) == "This is a test"

    # Test special character removal
    text = "Hello! This is a test... With punctuation???"
    cleaned = processor.clean_text(text)
    assert cleaned == "Hello This is a test With punctuation"  # Explicit expected output

    # Test more punctuation cases
    text = "Hello, world! How are you?"
    cleaned = processor.clean_text(text)
    assert cleaned == "Hello world How are you"  # No punctuation

    # Test unicode preservation
    text = "Hello in Spanish: ¡Hola!"
    cleaned = processor.clean_text(text)
    assert "Hola" in cleaned

def test_detect_language(processor):
    """Test language detection."""
    # Test English
    assert processor.detect_language("This is English text") == "en"

    # Test Spanish
    assert processor.detect_language("Este es un texto en español") == "es"

    # Test empty/invalid text
    assert processor.detect_language("") == "unknown"

def test_process_text(processor):
    """Test complete text processing."""
    text = "This is a sample English text for testing."
    result = processor.process_text(text)

    assert isinstance(result, ProcessedText)
    assert result.text == "This is a sample English text for testing"  # Note: cleaned text won't have the period
    assert result.language == "en"
    assert result.embedding is not None
    assert len(result.embedding) > 0

    # Test short text
    short_text = "Hi"
    assert processor.process_text(short_text) is None

def test_batch_process(processor):
    """Test batch processing functionality."""
    texts = [
        "This is English text",
        "Este es español",
        "Dies ist Deutsch",
    ]

    results = processor.batch_process(texts)

    assert len(results) == 3
    assert all(isinstance(r, ProcessedText) for r in results)
    assert all(r.embedding is not None for r in results)

    # Test empty batch
    assert processor.batch_process([]) == []

    # Test batch with some invalid texts
    texts = ["Valid text", "", "Another valid text", "Hi"]
    results = processor.batch_process(texts)
    assert len(results) == 2  # Only the valid texts should be processed