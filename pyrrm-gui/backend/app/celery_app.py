"""
Celery application configuration.

This module configures Celery for running calibration tasks asynchronously.
"""

from celery import Celery

from app.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "pyrrm_gui",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.celery_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Concurrency
    worker_concurrency=settings.max_concurrent_calibrations,
    
    # Task time limits
    task_soft_time_limit=7200,  # 2 hours soft limit
    task_time_limit=7500,       # 2.5 hours hard limit
    
    # Result expiration
    result_expires=86400,  # 24 hours
    
    # Worker prefetch
    worker_prefetch_multiplier=1,
)
