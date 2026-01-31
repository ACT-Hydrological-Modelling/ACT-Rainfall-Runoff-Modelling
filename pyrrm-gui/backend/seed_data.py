#!/usr/bin/env python
"""
Seed script to preload catchment 410734 data into the pyrrm-gui database.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.database import engine, init_db
from app.models import Catchment, Dataset, DatasetType


def seed_catchment_410734():
    """Seed the database with catchment 410734 and its datasets."""
    
    # Initialize database
    init_db()
    
    # Data source directory (mounted from host)
    source_dir = Path("/app/source_data/410734")
    
    # Uploads directory
    uploads_dir = Path("/app/data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    with Session(engine) as db:
        # Check if catchment already exists
        existing = db.query(Catchment).filter(Catchment.gauge_id == "410734").first()
        if existing:
            print(f"Catchment 410734 already exists (id: {existing.id})")
            return existing.id
        
        # Create catchment
        catchment = Catchment(
            name="Queanbeyan River at Queanbeyan",
            gauge_id="410734",
            area_km2=490.0,
            description="Queanbeyan River catchment upstream of Queanbeyan gauge. "
                       "Data includes SILO rainfall and Morton's wet-environment PET from 1890, "
                       "and observed streamflow from 1985."
        )
        db.add(catchment)
        db.flush()  # Get the ID
        
        catchment_id = catchment.id
        print(f"Created catchment: {catchment.name} (id: {catchment_id})")
        
        # Dataset configurations
        datasets_config = [
            {
                "name": "SILO Rainfall (1890-2025)",
                "type": DatasetType.RAINFALL,
                "source_file": "Default Input Set - Rain_QBN01.csv",
                "start_date": "1890-01-01",
                "end_date": "2025-01-14",
            },
            {
                "name": "Morton's PET (1890-2025)",
                "type": DatasetType.PET,
                "source_file": "Default Input Set - Mwet_QBN01.csv",
                "start_date": "1890-01-01",
                "end_date": "2025-01-14",
            },
            {
                "name": "Observed Streamflow (1985-2025)",
                "type": DatasetType.OBSERVED_FLOW,
                "source_file": "410734_recorded_Flow.csv",
                "start_date": "1985-03-03",
                "end_date": "2025-01-14",
            },
        ]
        
        for config in datasets_config:
            source_path = source_dir / config["source_file"]
            
            if not source_path.exists():
                print(f"  WARNING: Source file not found: {source_path}")
                continue
            
            # Copy file to uploads directory
            dest_filename = f"{catchment_id}_{config['type'].value}_{config['source_file']}"
            dest_path = uploads_dir / dest_filename
            shutil.copy2(source_path, dest_path)
            
            # Count records
            with open(dest_path, 'r') as f:
                record_count = sum(1 for _ in f) - 1  # Subtract header
            
            # Create dataset record
            dataset = Dataset(
                catchment_id=catchment_id,
                name=config["name"],
                type=config["type"],
                file_path=str(dest_path),
                start_date=datetime.strptime(config["start_date"], "%Y-%m-%d").date(),
                end_date=datetime.strptime(config["end_date"], "%Y-%m-%d").date(),
                record_count=record_count,
                extra_metadata={
                    "source": "ACT Government / SILO",
                    "units": "mm/day" if config["type"] != DatasetType.OBSERVED_FLOW else "ML/day"
                }
            )
            db.add(dataset)
            print(f"  Created dataset: {config['name']} ({record_count} records)")
        
        db.commit()
        print(f"\nCatchment 410734 seeded successfully!")
        return catchment_id


if __name__ == "__main__":
    seed_catchment_410734()
