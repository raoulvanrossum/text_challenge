from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from text_challenge.api.endpoints import router, initialize_service
from text_challenge.service.patent_service import PatentSearchService
from text_challenge.data_manager.data_manager import ProcessingConfig
from text_challenge.utils.logger import setup_logging
from text_challenge.config import MODEL_NAME, EMBEDDING_SIZE

app = FastAPI(title="Patent Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main index.html page"""
    return FileResponse(str(static_path / "index.html"))


# Include the router
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize the patent service when the API starts"""
    setup_logging(stdout_level="INFO")

    # Configure paths
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "raw" / "data.txt"
    cache_path = base_path / "data" / "processed" / "processed_patents.pkl"

    # Initialize config
    config = ProcessingConfig(
        use_cache=True, cache_path=cache_path, force_reprocess=False
    )

    # Initialize service
    service = PatentSearchService(model_name=MODEL_NAME, config=config)

    # Load the data
    service.initialize_with_data(data_path)

    # Initialize the service in endpoints
    initialize_service(service)
