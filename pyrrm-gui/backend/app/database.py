"""
Database configuration and session management.

Uses SQLAlchemy 2.0 async patterns with SQLite.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from app.config import get_settings

settings = get_settings()

# Create engine with appropriate settings for SQLite
# Use StaticPool for SQLite to handle concurrent access
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug
    )
else:
    engine = create_engine(
        settings.database_url,
        echo=settings.debug
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency that provides a database session.
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database, creating all tables."""
    # Import models to register them with Base
    from app.models import catchment, dataset, experiment, result  # noqa: F401
    
    Base.metadata.create_all(bind=engine)


def reset_db():
    """Reset the database (drop and recreate all tables). Use with caution!"""
    from app.models import catchment, dataset, experiment, result  # noqa: F401
    
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
