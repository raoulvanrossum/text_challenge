from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from text_challenge.service.patent_service import PatentSearchService
from text_challenge.service.schemas import SearchRequest

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