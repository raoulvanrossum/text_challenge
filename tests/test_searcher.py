import pytest
from unittest.mock import Mock, patch
from text_challenge.core.processor import TextProcessor, ProcessedText
from text_challenge.core.indexer import TextIndexer, SearchResult
from text_challenge.core.searcher import TextSearcher, SearchQuery

@pytest.fixture
def mock_processor():
    processor = Mock(spec=TextProcessor)
    processor.process_text.return_value = ProcessedText(
        text="test",
        embedding=[0.1] * 384,
        language="en"
    )
    return processor

@pytest.fixture
def mock_indexer():
    indexer = Mock(spec=TextIndexer)
    indexer.search.return_value = [
        SearchResult(
            text="test result",
            score=0.9,
            language="en"
        )
    ]
    return indexer

@pytest.fixture
def searcher(mock_processor, mock_indexer):
    return TextSearcher(mock_processor, mock_indexer)

@pytest.mark.asyncio
async def test_search(searcher, mock_processor, mock_indexer):
    query = SearchQuery(
        text="test query",
        top_k=5,
        threshold=0.7
    )
    
    results = await searcher.search(query)
    
    assert len(results) == 1
    assert results[0].text == "test result"
    assert results[0].score == 0.9
    
    mock_processor.process_text.assert_called_once_with("test query")
    mock_indexer.search.assert_called_once_with(
        query_embedding=[0.1] * 384,
        top_k=5,
        threshold=0.7
    )

@pytest.mark.asyncio
async def test_search_no_processed_query(searcher, mock_processor, mock_indexer):
    mock_processor.process_text.return_value = None
    query = SearchQuery(text="invalid query")
    
    results = await searcher.search(query)
    
    assert len(results) == 0
    mock_indexer.search.assert_not_called()
