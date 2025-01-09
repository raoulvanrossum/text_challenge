from pathlib import Path
from loguru import logger

from src.patent_search.service.patent_service import PatentSearchService
from src.patent_search.data_manager.data_manager import ProcessingConfig
from src.patent_search.service.schemas import (
    SearchRequest,
    SearchResponse,
)
from src.patent_search.utils.logger import setup_logging
from src.patent_search.config import MODEL_NAME, BASE_FOLDER


def main():
    # Setup basic logging
    setup_logging(stdout_level="INFO")

    try:
        logger.info("Starting Patent Search Service")

        # Configure paths
        data_path = BASE_FOLDER / "data" / "raw" / "data.txt"
        cache_path = BASE_FOLDER / "data" / "processed" / "processed_patents.pkl"

        # Create directories if they don't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize config
        config = ProcessingConfig(
            use_cache=True, cache_path=cache_path, force_reprocess=False
        )

        # Initialize service
        logger.info("Initializing service...")
        service = PatentSearchService(model_name=MODEL_NAME, config=config)

        # Load the data
        logger.info(f"Loading data from {data_path}")
        service.initialize_with_data(data_path)


        stats = service.get_statistics()
        logger.info(f"Total patents loaded: {stats['total_patents']}")

        request: SearchRequest = SearchRequest(
            ["robot", "nano"], threshold=0.7, max_results=10
        )
        response: SearchResponse = service.search(request)

        for res in response.results:
            print(res.similarity, res.text[:100])

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
