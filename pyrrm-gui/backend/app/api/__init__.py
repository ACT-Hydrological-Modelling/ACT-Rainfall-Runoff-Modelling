"""
API routes for pyrrm-gui.
"""

from fastapi import APIRouter

from app.api import catchments, datasets, experiments, results, reference

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(
    catchments.router,
    prefix="/catchments",
    tags=["Catchments"]
)

api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"]
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["Experiments"]
)

api_router.include_router(
    results.router,
    prefix="/results",
    tags=["Results"]
)

api_router.include_router(
    reference.router,
    prefix="/reference",
    tags=["Reference Data"]
)
