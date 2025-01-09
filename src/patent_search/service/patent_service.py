from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from loguru import logger
import langdetect

from src.patent_search.service.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from src.patent_search.core.processor import ProcessedText
from src.patent_search.core.indexer import TextIndexer
from src.patent_search.data_manager.data_manager import DataManager, ProcessingConfig
from src.patent_search.config import MODEL_NAME

KEYWORD_BONUS = 0.6


class PatentSearchService:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
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
        """Initialize the service with data from file."""
        processed_texts = self.data_manager.load_or_process_data(data_path)
        self.indexer.add_texts(processed_texts)
        logger.info(f"Initialized with {len(processed_texts)} patents")

    def detect_language(self, text: str) -> str:
        """Detect the language of a given text."""
        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def _search_single_keyword(
        self, keyword: str, max_results: int, threshold: float
    ) -> List:
        """Search for patents matching a single keyword."""
        query_embedding = self.model.encode(keyword, normalize_embeddings=True)
        return self.indexer.search(
            query_embedding=query_embedding.tolist(),
            top_k=max_results,
            threshold=threshold,
        )

    def _merge_keyword_results(
        self, keyword_results: Dict[str, List], max_results: int
    ) -> List[SearchResultItem]:
        """Merge results from multiple keywords with weighted scoring."""
        combined_scores = {}
        total_keywords = len(keyword_results)

        # Process results for each keyword
        for keyword, results in keyword_results.items():
            for result in results:
                if result.text not in combined_scores:
                    combined_scores[result.text] = {
                        "base_score": result.score,
                        "keyword_matches": {keyword},
                        "language": result.language,
                        "metadata": result.metadata,
                    }
                else:
                    entry = combined_scores[result.text]
                    # Keep the highest base score among all keyword matches
                    entry["base_score"] = max(entry["base_score"], result.score)
                    entry["keyword_matches"].add(keyword)

        # Calculate final scores with weighting
        final_results = []
        for text, data in combined_scores.items():
            # Calculate weighted score:
            # base_score + (number_of_matching_keywords * bonus_per_keyword)
            num_matches = len(data["keyword_matches"])
            keyword_bonus = KEYWORD_BONUS * (
                num_matches - 1
            )  # -1 because base score already counts as one
            final_score = min(1.0, data["base_score"] + keyword_bonus)  # Cap at 1.0

            # Create explanation
            matching_keywords = ", ".join(data["keyword_matches"])
            explanation = (
                f"Matched {num_matches}/{total_keywords} keywords: {matching_keywords}. "
                f"Base similarity: {data['base_score']:.3f}, "
                f"Keyword bonus: {keyword_bonus:.3f}, "
                f"Final score: {final_score:.3f}"
            )

            final_results.append(
                SearchResultItem(
                    text=text,
                    similarity=final_score,
                    language=data["language"],
                    metadata=data["metadata"],
                    explanation=explanation,
                )
            )

        # Sort by final score and limit results
        final_results.sort(key=lambda x: x.similarity, reverse=True)
        return final_results[:max_results]

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Search for patents matching any of the given keywords.

        Args:
            request: SearchRequest object containing search parameters

        Returns:
            SearchResponse: Search results with similarity scores
        """
        # Search for each keyword separately
        keyword_results = {}
        for keyword in request.keywords:
            results = self._search_single_keyword(
                keyword=keyword,
                max_results=request.max_results,
                threshold=request.threshold,
            )
            keyword_results[keyword] = results

        # Merge and rank results
        merged_results = self._merge_keyword_results(
            keyword_results=keyword_results,
            max_results=request.max_results,
        )

        # Create query info
        query_info = {
            "keywords": request.keywords,
            "threshold": request.threshold,
            "max_results": request.max_results,
            "language": request.language,
            "separate_keyword_search": True,
        }

        return SearchResponse(results=merged_results, query_info=query_info)

    def add_texts(self, texts: List[str]) -> None:
        """Add new texts to the search index."""
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
        """Get a summary of languages in the index."""
        return self.indexer.get_languages_summary()

    def get_total_patents(self) -> int:
        """Get the total number of patents in the index."""
        return self.indexer.ntotal

    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the patents."""
        total = self.get_total_patents()
        languages = self.get_languages_summary()

        return {
            "total_patents": total,
            "languages": languages,
        }
