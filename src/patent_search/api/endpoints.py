from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from src.patent_search.service.patent_service import PatentSearchService
from src.patent_search.service.schemas import SearchRequest

from src.patent_search.service.schemas import BatchPatentSubmission
from typing import Dict
router = APIRouter()

# Reference to the service instance
patent_service: PatentSearchService = None


def initialize_service(service: PatentSearchService):
    """Initialize the global service instance"""
    global patent_service
    patent_service = service


@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Patent Search API"}


@router.get("/stats")
async def get_stats():
    """Get statistics about the patent database"""
    if not patent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return patent_service.get_statistics()


@router.post("/search")
async def search(request: SearchRequest):
    """
    Search patents based on keywords
    """
    if not patent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        results = patent_service.search(request)
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patents/add")
async def add_patents(
        submission: BatchPatentSubmission,
        background_tasks: BackgroundTasks
) -> Dict:
    """Add new patents to the system."""
    if not patent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Process and add patents
        texts = [patent.text for patent in submission.patents]
        metadata = [patent.metadata for patent in submission.patents]

        # Add to index
        processed_texts = patent_service.add_texts(texts, metadata)

        # Update cache in background
        background_tasks.add_task(
            patent_service.data_manager.append_to_cache,
            processed_texts
        )

        return {
            "status": "success",
            "processed": len(texts),
            "message": "Patents added successfully and cache update scheduled"
        }
    except Exception as e:
        logger.error(f"Error adding patents: {e}")
        raise HTTPException(status_code=500, detail=str(e))