import pytest
from pathlib import Path
from src.patent_search.service.patent_service import PatentSearchService
from src.patent_search.service.schemas import SearchRequest
from src.patent_search.data_manager.data_manager import ProcessingConfig

@pytest.fixture
def service_with_real_data():
    data_path = Path.cwd() / "data" / "raw" / "data.txt"

    # Initialize service with config
    config = ProcessingConfig(use_cache=True)
    service = PatentSearchService(config=config)

    # Load data
    service.initialize_with_data(data_path)
    return service

def test_specific_keyword_search(service_with_real_data):
    """Test searching for specific Dutch and English keywords."""
    # Test with lower threshold
    request = SearchRequest(
        keywords=["bloedplasmamonsters", "robot"],
        threshold=0.3,
        max_results=10
    )

    response = service_with_real_data.search(request)

    print("\nSearch Results for 'bloedplasmamonsters' and 'robot':")
    print(f"Total results found: {len(response.results)}")
    print(f"Query Info: {response.query_info}")

    # Also try individual searches
    for keyword in ["bloedplasmamonsters", "robot"]:
        single_request = SearchRequest(
            keywords=[keyword],
            threshold=0.3
        )
        single_response = service_with_real_data.search(single_request)
        print(f"\nResults for single keyword '{keyword}':")
        print(f"Found {len(single_response.results)} results")
        for result in single_response.results:
            print(f"\nText preview: {result.text[:200]}")
            print(f"Score: {result.similarity}")
            print(f"Language: {result.language}")

