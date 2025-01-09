from typing import List, Optional
from dataclasses import dataclass
from .processor import TextProcessor
from .indexer import TextIndexer, SearchResult


@dataclass
class SearchQuery:
    text: str
    top_k: int = 5
    threshold: float = 0.7


class TextSearcher:
    def __init__(self, processor: TextProcessor, indexer: TextIndexer):
        self.processor = processor
        self.indexer = indexer

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar texts to the query"""
        # Process query
        processed_query = self.processor.process_text(query.text)
        if not processed_query:
            return []

        # Search using the query embedding
        results = self.indexer.search(
            query_embedding=processed_query.embedding,
            top_k=query.top_k,
            threshold=query.threshold,
        )

        return results
