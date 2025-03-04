import pytest
from src.patent_search.core.indexer import TextIndexer, ProcessedText
from src.patent_search.config import EMBEDDING_SIZE

@pytest.fixture
def indexer():
    return TextIndexer(collection_name="test_collection")


def test_add_and_search(indexer):
    # Test data
    processed_texts = [
        ProcessedText(
            text="Test document 1",
            embedding=[1.0] + [0.0] * (EMBEDDING_SIZE-1),  # Make first document clearly closest to query
            language="en",
            metadata={"source": "test"}
        ),
        ProcessedText(
            text="Test document 2",
            embedding=[0.0] * EMBEDDING_SIZE,
            language="en",
            metadata={"source": "test"}
        )
    ]

    # Add texts
    indexer.add_texts(processed_texts)

    # Search with query similar to first document
    results = indexer.search(
        query_embedding=[1.0] + [0.0] * (EMBEDDING_SIZE-1),
        top_k=1
    )

    assert len(results) == 1
    assert results[0].text == "Test document 1"
    assert results[0].metadata["source"] == "test"
