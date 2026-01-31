#!/usr/bin/env python
"""
Celery worker entry point.

Run with:
    celery -A celery_worker worker --loglevel=info
"""

import sys
import os
from pathlib import Path

# Add pyrrm to path - in Docker it's mounted at /app/pyrrm
# For local development, it's at the parent of pyrrm-gui
app_dir = Path(__file__).parent  # /app in Docker
pyrrm_in_docker = app_dir / "pyrrm"

if pyrrm_in_docker.exists():
    # Running in Docker with mounted pyrrm
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
else:
    # Running locally - pyrrm is at parent of pyrrm-gui
    try:
        pyrrm_path = Path(__file__).parents[2]
        if str(pyrrm_path) not in sys.path:
            sys.path.insert(0, str(pyrrm_path))
    except IndexError:
        pass  # Not enough parent directories

# Add app directory to path
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from app.celery_app import celery_app

# Import tasks to register them
from app.tasks import celery_tasks  # noqa: F401

if __name__ == "__main__":
    celery_app.start()
