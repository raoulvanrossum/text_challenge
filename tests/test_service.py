# tests/test_service.py
import pytest
from pathlib import Path
from text_challenge.service.patent_service import PatentSearchService
from text_challenge.service.schemas import SearchRequest
from text_challenge.core.indexer import ProcessedText
from datetime import datetime
import pickle
from tqdm import tqdm


@pytest.fixture
def service_with_real_data():
    """Fixture that provides a service loaded with real patent data."""
    # ensure_directories()

    # Define paths relative to test directory
    data_path = Path.cwd().parent / "data" / "raw" / "data.txt"
    index_path = Path.cwd().parent / "data" / "indexes" / "patent_index"

    # Initialize service
    service = PatentSearchService()

    print("Processing new data...")
    if not data_path.exists():
        pytest.skip(f"Patent data file not found at {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(data_path, 'r', encoding='latin-1') as f:
            content = f.read()

    # Split patents by single empty line, handling both \n and \r\n
    # patent_texts = [p.strip() for p in content.split('\n\n') if p.strip()]
    patent_texts = [abstract for abstract in content.split("\n")]

    print(f"\nFound {len(patent_texts)} patents")

    # Debug: Print first few patents
    print("\nFirst 3 patents (preview):")
    for i, patent in enumerate(patent_texts[:3], 1):
        print(f"\nPatent {i} ({len(patent)} characters):")
        print(f"{patent[:150]}...")

    # Debug: Look for specific keywords
    debug_keywords = ['bloedplasmamonsters', 'robot']
    for keyword in debug_keywords:
        matching_patents = [p for p in patent_texts if keyword.lower() in p.lower()]
        if matching_patents:
            print(f"\nFound {len(matching_patents)} patents containing '{keyword}'")
            print("First matching patent preview:")
            print(matching_patents[0][:200])

    processed_texts = []
    # Process each patent with progress bar
    for text in tqdm(patent_texts,
                     desc="Processing patents",
                     unit="patent"):
        try:
            # Detect language first to validate text
            language = service.detect_language(text)

            # Create embedding
            embedding = service.model.encode(text)

            # Create ProcessedText object
            processed_text = ProcessedText(
                text=text,
                embedding=embedding,
                language=language,
                metadata={
                    "processed_date": datetime.now().isoformat(),
                    "char_length": len(text),
                    "word_count": len(text.split())
                }
            )
            processed_texts.append(processed_text)

        except Exception as e:
            print(f"Error processing patent: {e}")
            continue

    print(f"\nSuccessfully processed {len(processed_texts)} patents")

    # Clear any existing index
    if (Path(index_path) / "state.json").exists():
        print("Removing existing index...")
        import shutil
        shutil.rmtree(index_path)

    print(f"Adding {len(processed_texts)} patents to index...")
    # Add to index with progress tracking
    service.indexer.add_texts(processed_texts)

    return service

def test_specific_keyword_search(service_with_real_data):
    """Test searching for specific Dutch and English keywords."""
    # Test with lower threshold
    request = SearchRequest(
        keywords=["bloedplasmamonsters", "robot"],
        threshold=0.3,  # Lower threshold for testing
        max_results=5
    )

    response = service_with_real_data.search(request)

    print("\nSearch Results for 'bloedplasmamonsters' and 'robot':")
    print(f"Total results found: {len(response.results)}")
    print(f"Query Info: {response.query_info}")

    # Also try individual searches
    for keyword in ["bloedplasmamonsters", "robot"]:
        single_request = SearchRequest(
            keywords=[keyword],
            threshold=0.3,
            max_results=5
        )
        single_response = service_with_real_data.search(single_request)
        print(f"\nResults for single keyword '{keyword}':")
        print(f"Found {len(single_response.results)} results")
        for result in single_response.results:
            print(f"\nText preview: {result.text[:200]}")
            print(f"Score: {result.similarity}")
            print(f"Language: {result.language}")

