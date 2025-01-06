from text_challenge.service.patent_service import PatentSearchService
from text_challenge.service.schemas import SearchRequest
from text_challenge.core.indexer import TextIndexer, ProcessedText


def main():
    # Initialize service
    service = PatentSearchService()

    # Add some sample data
    sample_texts = [
        ProcessedText(
            text="A solar panel with improved efficiency using nanotechnology.",
            embedding=service.model.encode("A solar panel with improved efficiency using nanotechnology."),
            language="en",
            metadata={"patent_id": "123"}
        ),
        ProcessedText(
            text="Un panneau solaire avec une efficacité améliorée utilisant la nanotechnologie.",
            embedding=service.model.encode(
                "Un panneau solaire avec une efficacité améliorée utilisant la nanotechnologie."),
            language="fr",
            metadata={"patent_id": "124"}
        )
    ]
    service.indexer.add_texts(sample_texts)

    # Create search request
    request = SearchRequest(
        keywords=["solar", "nanotechnology"],
        threshold=0.7,
        max_results=5
    )

    # Perform search
    response = service.search(request)

    # Print results
    print("\nSearch Results:")
    print(f"Query Info: {response.query_info}")
    print("\nMatches:")
    for result in response.results:
        print(f"\nText: {result.text}")
        print(f"Similarity: {result.similarity:.3f}")
        print(f"Language: {result.language}")
        print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
