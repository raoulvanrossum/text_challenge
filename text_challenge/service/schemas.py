from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class SearchRequest:
    keywords: List[str]
    threshold: float = 0.7  # Precision-Recall control
    max_results: int = 10
    language: Optional[str] = None

    def __post_init__(self):
        if not self.keywords:
            raise ValueError("Keywords list cannot be empty")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if self.max_results < 1:
            raise ValueError("max_results must be positive")


@dataclass
class SearchResultItem:
    text: str
    similarity: float
    language: str
    explanation: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResponse:
    results: List[SearchResultItem]
    query_info: Dict[str, Any]
    timestamp: datetime = datetime.now()
