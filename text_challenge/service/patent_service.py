from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langdetect import detect
from loguru import logger
from datetime import datetime

from .schemas import SearchRequest, SearchResponse, SearchResultItem
from text_challenge.core.indexer import TextIndexer


class PatentSearchService:
    def __init__(
            self,
            model_name: str = "intfloat/multilingual-e5-small",
            indexer: Optional[TextIndexer] = None
    ):
        """Initialize the patent search service.

        Args:
            model_name: Name of the sentence transformer model to use
            indexer: Optional pre-configured TextIndexer instance
        """
        logger.info(f"Initializing PatentSearchService with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.indexer = indexer or TextIndexer(dimension=384)

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            logger.warning(f"Could not detect language for text: {text[:100]}...")
            return "unknown"

    def prepare_query(self, keywords: List[str]) -> str:
        """Prepare query string from keywords."""
        return " ".join(keywords)

    def search(self, request: SearchRequest) -> SearchResponse:
        """Search for patent abstracts based on keywords.

        Args:
            request: SearchRequest object containing search parameters

        Returns:
            SearchResponse object containing search results and metadata

        Raises:
            ValueError: If the request parameters are invalid
        """
        try:
            # Prepare query
            query = self.prepare_query(request.keywords)
            query_language = self.detect_language(query)

            # Generate embedding
            logger.debug(f"Generating embedding for query: {query}")
            query_embedding = self.model.encode(query)

            # Search using indexer
            raw_results = self.indexer.search(
                query_embedding=query_embedding,
                top_k=request.max_results,
                threshold=request.threshold
            )

            # Process results
            results = []
            for r in raw_results:
                explanation = {
                    "similarity_score": r.score,
                    "matched_language": r.language,
                    "threshold_used": request.threshold,
                    "original_keywords": request.keywords,
                    "query_language": query_language
                }

                result = SearchResultItem(
                    text=r.text,
                    similarity=r.score,
                    language=r.language,
                    explanation=explanation,
                    metadata=r.metadata
                )
                results.append(result)

            # Create response
            query_info = {
                "original_query": query,
                "query_language": query_language,
                "threshold": request.threshold,
                "max_results": request.max_results,
                "results_found": len(results),
                "requested_language": request.language
            }

            return SearchResponse(
                results=results,
                query_info=query_info
            )

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and information."""
        return {
            "total_documents": self.indexer.ntotal,
            "model_name": self.model.get_model_name(),
            "embedding_dimension": self.indexer.dimension,
            "timestamp": datetime.now().isoformat()
        }
