"""
Text processor module for handling multilingual text preprocessing and analysis.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache
from loguru import logger
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer


@dataclass
class ProcessedText:
    """Container for processed text and its metadata."""

    text: str
    language: str
    embedding: Optional[List[float]] = None
    metadata: Dict = None


class TextProcessor:
    """Handles text preprocessing, language detection, and embedding generation."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        min_text_length: int = 10,
    ):
        """
        Initialize the text processor.

        Args:
            model_name: Name of the sentence transformer model to use
            min_text_length: Minimum text length to process
        """
        self.model = SentenceTransformer(model_name)
        self.min_text_length = min_text_length
        logger.info(f"Initialized TextProcessor with model: {model_name}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        # Remove excess whitespace
        text = " ".join(text.split())

        # Remove special characters while preserving unicode
        text = re.sub(r"[^\w\s\u0080-\uffff]", " ", text)

        # Normalize whitespace again
        text = " ".join(text.split())

        return text.strip()

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Args:
            text: Input text string

        Returns:
            ISO 639-1 language code or 'unknown'
        """
        try:
            return detect(text)
        except LangDetectException:
            logger.warning(f"Could not detect language for text: {text[:100]}...")
            return "unknown"

    @lru_cache(maxsize=1000)
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the input text.

        Args:
            text: Input text string

        Returns:
            List of floating point numbers representing the text embedding
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def process_text(
        self, text: str, generate_embedding: bool = True
    ) -> Optional[ProcessedText]:
        """
        Process input text by cleaning, detecting language, and optionally generating embedding.

        Args:
            text: Input text string
            generate_embedding: Whether to generate text embedding

        Returns:
            ProcessedText object or None if text is invalid
        """
        # Check if text meets minimum length requirement
        if len(text.strip()) < self.min_text_length:
            logger.warning(f"Text too short: {len(text.strip())} chars")
            return None

        # Clean text
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            logger.warning("Text was empty after cleaning")
            return None

        # Detect language
        language = self.detect_language(cleaned_text)

        # Generate embedding if requested
        embedding = None
        if generate_embedding:
            embedding = self.generate_embedding(cleaned_text)

        # Create metadata
        metadata = {
            "original_length": len(text),
            "processed_length": len(cleaned_text),
            "language": language,
        }

        return ProcessedText(
            text=cleaned_text, language=language, embedding=embedding, metadata=metadata
        )

    def batch_process(
        self, texts: List[str], batch_size: int = 32, generate_embeddings: bool = True
    ) -> List[ProcessedText]:
        """
        Process a batch of texts efficiently.

        Args:
            texts: List of input text strings
            batch_size: Size of batches for processing
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of ProcessedText objects
        """
        results = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Clean and detect language for each text
            processed_batch = []
            for text in batch:
                processed = self.process_text(text, generate_embedding=False)
                if processed is not None:
                    processed_batch.append(processed)

            # Generate embeddings in batch if requested
            if generate_embeddings and processed_batch:
                batch_texts = [p.text for p in processed_batch]
                embeddings = self.model.encode(batch_texts, convert_to_tensor=False)

                for processed, embedding in zip(processed_batch, embeddings):
                    processed.embedding = embedding.tolist()

            results.extend(processed_batch)

            logger.info(f"Processed batch of {len(batch)} texts")

        return results
