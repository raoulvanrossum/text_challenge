import pytest
from text_challenge.service.patent_service import PatentSearchService
from text_challenge.service.schemas import SearchRequest
from text_challenge.core.indexer import ProcessedText


@pytest.fixture
def service():
    return PatentSearchService()


@pytest.fixture
def sample_data(service):
    texts = [
        ProcessedText(
            text="A solar panel with improved efficiency.",
            embedding=service.model.encode("A solar panel with improved efficiency."),
            language="en",
            metadata={"patent_id": "123"}
        )
    ]
    service.indexer.add_texts(texts)
    return texts


def test_search_basic(service, sample_data):
    request = SearchRequest(keywords=["solar", "panel"])
    response = service.search(request)

    assert response.results
    assert len(response.results) > 0
    assert response.query_info["original_query"] == "solar panel"
    assert all(r.similarity <= 1.0 for r in response.results)


def test_search_threshold(service, sample_data):
    # Test with very high threshold
    request = SearchRequest(keywords=["solar"], threshold=0.99)
    response = service.search(request)
    assert len(response.results) == 0

    # Test with lower threshold
    request = SearchRequest(keywords=["solar"], threshold=0.5)
    response = service.search(request)
    assert len(response.results) > 0


def test_invalid_request():
    service = PatentSearchService()

    with pytest.raises(ValueError):
        SearchRequest(keywords=[])

    with pytest.raises(ValueError):
        SearchRequest(keywords=["test"], threshold=1.5)

    with pytest.raises(ValueError):
        SearchRequest(keywords=["test"], max_results=0)
