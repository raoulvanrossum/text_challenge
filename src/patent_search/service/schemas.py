from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict


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
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResponse:
    results: List[SearchResultItem]
    query_info: Dict[str, Any]
    timestamp: datetime = datetime.now()


class PatentSubmission(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "A new patent abstract about technology...",
                "metadata": {"source": "manual_input", "submission_date": "2024-03-14"},
            }
        }
    )


class BatchPatentSubmission(BaseModel):
    patents: List[PatentSubmission]
