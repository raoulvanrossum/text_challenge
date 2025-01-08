from pathlib import Path
from loguru import logger

from text_challenge.service.patent_service import PatentSearchService
from text_challenge.data_manager.data_manager import ProcessingConfig
from text_challenge.service.schemas import SearchRequest, SearchResponse, SearchResultItem
from text_challenge.utils.logger import setup_logging


def main():
    # Setup basic logging
    setup_logging(stdout_level="INFO")


    try:
        logger.info("Starting Patent Search Service")

        # Configure paths
        base_path = Path(__file__).parent
        data_path = base_path / "data" / "raw" / "data.txt"
        cache_path = base_path / "data" / "processed" / "processed_patents.pkl"

        # Create directories if they don't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize config
        config = ProcessingConfig(
            use_cache=True,
            cache_path=cache_path,
            force_reprocess=False
        )

        # Initialize service
        logger.info("Initializing service...")
        service = PatentSearchService(
            model_name="intfloat/multilingual-e5-small",
            config=config
        )

        # Load the data
        logger.info(f"Loading data from {data_path}")
        service.initialize_with_data(data_path)


        # Rest of your main function...
        stats = service.get_statistics()
        logger.info(f"Total patents loaded: {stats['total_patents']}")

        request: SearchRequest = SearchRequest(["robot", "nano"], threshold=0.7, max_results=10)
        response : SearchResponse = service.search(request)

        for res in response.results:
            print(res.similarity, res.text[:100])




    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()