from pathlib import Path
from typing import List, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from loguru import logger
import langdetect

# Internal imports should match your structure
from text_challenge.service.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from text_challenge.core.processor import ProcessedText
from text_challenge.core.indexer import TextIndexer
from text_challenge.data_manager.data_manager import DataManager, ProcessingConfig


class PatentSearchService:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        config: Optional[ProcessingConfig] = None,
    ):
        """
        Initialize the patent search service.

        Args:
            model_name: Name or path of the sentence transformer model
            config: Configuration for data processing and caching
        """
        logger.info(f"Initializing PatentSearchService with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.indexer = TextIndexer()
        self.data_manager = DataManager(self, config or ProcessingConfig())

    def initialize_with_data(self, data_path: Path):
        """
        Initialize the service with data from file.

        Args:
            data_path: Path to the patent data file
        """
        processed_texts = self.data_manager.load_or_process_data(data_path)
        self.indexer.add_texts(processed_texts)
        logger.info(f"Initialized with {len(processed_texts)} patents")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a given text.

        Args:
            text: Text to detect language for

        Returns:
            Language code (e.g., 'en', 'nl')
        """
        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Search for patents similar to the given keywords.

        Args:
            request: SearchRequest object containing search parameters

        Returns:
            SearchResponse: Search results with similarity scores
        """
        # Get embedding for the query
        query_text = " ".join(request.keywords)
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        # Search in the index
        similar_texts = self.indexer.search(
            query_embedding=query_embedding.tolist(),
            top_k=request.max_results,
            threshold=request.threshold,
        )

        # Convert to SearchResultItem objects
        results = [
            SearchResultItem(
                text=result.text,
                similarity=result.score,
                language=result.language,
                metadata=result.metadata,
                explanation=f"Similarity score: {result.score:.3f}",
            )
            for result in similar_texts
        ]

        # Create query info directly from request
        query_info = {
            "keywords": request.keywords,
            "threshold": request.threshold,
            "max_results": request.max_results,
            "language": request.language,
        }

        return SearchResponse(results=results, query_info=query_info)

    def add_texts(self, texts: List[str]) -> None:
        """
        Add new texts to the search index.

        Args:
            texts: List of texts to add
        """
        processed_texts = []
        for text in texts:
            language = self.detect_language(text)
            embedding = self.model.encode(text)
            processed_text = ProcessedText(
                text=text,
                embedding=embedding,
                language=language,
                metadata={"added_date": datetime.now().isoformat()},
            )
            processed_texts.append(processed_text)

        self.indexer.add_texts(processed_texts)
        logger.info(f"Added {len(processed_texts)} new texts to index")

    def get_languages_summary(self) -> dict:
        """
        Get a summary of languages in the index.

        Returns:
            Dictionary with language counts
        """
        return self.indexer.get_languages_summary()

    def get_total_patents(self) -> int:
        """Get the total number of patents in the index."""
        return self.indexer.ntotal  # or however you store/access the texts

    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the patents."""
        total = self.get_total_patents()
        languages = self.get_languages_summary()

        return {
            "total_patents": total,
            "languages": languages,
        }
