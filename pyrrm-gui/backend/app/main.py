"""
pyrrm-gui FastAPI Application

Main entry point for the pyrrm-gui backend API.
"""

import sys
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add pyrrm to path - handles both Docker and local environments
def _setup_pyrrm_path():
    current_file = Path(__file__)
    
    # Try Docker path first (/app)
    docker_app_dir = current_file.parents[1]  # /app/app -> /app
    if (docker_app_dir / "pyrrm").exists():
        if str(docker_app_dir) not in sys.path:
            sys.path.insert(0, str(docker_app_dir))
        return
    
    # Try local development path
    try:
        local_path = current_file.parents[3]
        if str(local_path) not in sys.path:
            sys.path.insert(0, str(local_path))
    except IndexError:
        pass

_setup_pyrrm_path()

from app.config import get_settings
from app.database import init_db
from app.api import api_router
from app.websocket.progress import progress_manager

settings = get_settings()

# Redis for subscribing to progress updates
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Runs on startup and shutdown.
    """
    # Startup
    print("=" * 60)
    print("pyrrm-gui Backend Starting...")
    print("=" * 60)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize database
    init_db()
    print(f"Database initialized: {settings.database_url}")
    
    # Check pyrrm availability
    try:
        from pyrrm.models.sacramento import Sacramento
        print("pyrrm library loaded successfully")
    except ImportError as e:
        print(f"WARNING: pyrrm not available: {e}")
    
    print("=" * 60)
    print("Backend ready!")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("pyrrm-gui Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="pyrrm-gui API",
    description="REST API for the pyrrm rainfall-runoff model calibration GUI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


# WebSocket endpoint for real-time progress updates
@app.websocket("/ws/experiments/{experiment_id}/progress")
async def websocket_progress(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time calibration progress.
    
    Clients can connect to receive live updates during calibration.
    Updates are pushed when the calibration task publishes to Redis.
    """
    await progress_manager.connect(websocket, experiment_id)
    
    try:
        if REDIS_AVAILABLE:
            # Subscribe to Redis channel for this experiment
            redis_client = aioredis.from_url(settings.redis_url)
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(f"calibration:{experiment_id}")
            
            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                    
                    # Check for completion
                    if data.get("type") == "completed":
                        break
        else:
            # Fallback: just keep connection open
            while True:
                # Wait for any message from client (ping/pong)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    # Send ping
                    await websocket.send_json({"type": "ping"})
                    
    except WebSocketDisconnect:
        pass
    finally:
        progress_manager.disconnect(websocket, experiment_id)
        if REDIS_AVAILABLE:
            await pubsub.unsubscribe(f"calibration:{experiment_id}")
            await redis_client.close()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "redis_available": REDIS_AVAILABLE
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/api/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
