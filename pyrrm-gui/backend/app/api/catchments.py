"""
Catchment API endpoints.

Provides CRUD operations for catchments.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd

from app.database import get_db
from app.models.catchment import Catchment
from app.models.dataset import Dataset, DatasetType
from app.schemas.catchment import (
    CatchmentCreate,
    CatchmentUpdate,
    CatchmentResponse,
    CatchmentListResponse,
)
from app.services.data_handler import DataHandlerService

router = APIRouter()


@router.get("/", response_model=List[CatchmentListResponse])
def list_catchments(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all catchments.
    
    Returns summary information for each catchment including
    dataset and experiment counts.
    """
    catchments = db.query(Catchment).offset(skip).limit(limit).all()
    
    result = []
    for c in catchments:
        result.append(CatchmentListResponse(
            id=c.id,
            name=c.name,
            gauge_id=c.gauge_id,
            area_km2=c.area_km2,
            created_at=c.created_at,
            dataset_count=len(c.datasets),
            experiment_count=len(c.experiments)
        ))
    
    return result


@router.post("/", response_model=CatchmentResponse, status_code=status.HTTP_201_CREATED)
def create_catchment(
    catchment: CatchmentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new catchment.
    """
    db_catchment = Catchment(
        name=catchment.name,
        gauge_id=catchment.gauge_id,
        area_km2=catchment.area_km2,
        description=catchment.description
    )
    
    db.add(db_catchment)
    db.commit()
    db.refresh(db_catchment)
    
    return db_catchment


@router.get("/{catchment_id}", response_model=CatchmentResponse)
def get_catchment(
    catchment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a catchment by ID.
    
    Returns full details including datasets and experiments.
    """
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    return catchment


@router.put("/{catchment_id}", response_model=CatchmentResponse)
def update_catchment(
    catchment_id: str,
    catchment_update: CatchmentUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a catchment.
    
    Only provided fields are updated.
    """
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    update_data = catchment_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(catchment, field, value)
    
    db.commit()
    db.refresh(catchment)
    
    return catchment


@router.delete("/{catchment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_catchment(
    catchment_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a catchment and all associated data.
    
    This will also delete all datasets and experiments.
    """
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    db.delete(catchment)
    db.commit()
    
    return None


@router.get("/{catchment_id}/timeseries")
def get_catchment_timeseries(
    catchment_id: str,
    max_points: int = Query(1000, ge=100, le=10000, description="Maximum data points to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get merged time series data for a catchment.
    
    Returns rainfall, PET, and observed flow data merged by date,
    sampled to max_points for efficient visualization.
    
    This is used for interactive calibration period selection.
    """
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    # Load datasets
    datasets = db.query(Dataset).filter(Dataset.catchment_id == catchment_id).all()
    
    rainfall_df = None
    pet_df = None
    observed_df = None
    
    # Track observed flow range separately (for calibration constraints)
    observed_flow_start = None
    observed_flow_end = None
    
    for ds in datasets:
        try:
            df, _ = DataHandlerService.load_csv(ds.file_path, ds.type.value)
            if ds.type == DatasetType.RAINFALL:
                rainfall_df = df
            elif ds.type == DatasetType.PET:
                pet_df = df
            elif ds.type == DatasetType.OBSERVED_FLOW:
                observed_df = df
                # Get the actual date range of observed flow data (excluding NaN)
                valid_flow = df.dropna()
                if len(valid_flow) > 0:
                    observed_flow_start = valid_flow.index.min()
                    observed_flow_end = valid_flow.index.max()
        except Exception:
            continue
    
    if rainfall_df is None and pet_df is None and observed_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No datasets found for this catchment"
        )
    
    # Merge datasets
    try:
        merged = DataHandlerService.merge_datasets(rainfall_df, pet_df, observed_df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to merge datasets: {str(e)}"
        )
    
    # Store original data range info BEFORE sampling
    original_start = merged.index.min()
    original_end = merged.index.max()
    original_total_days = len(merged)
    
    # Sample data if too large (for visualization only)
    n_points = len(merged)
    if n_points > max_points:
        step = n_points // max_points
        indices = np.arange(0, n_points, step)[:max_points]
        merged = merged.iloc[indices]
    
    # Prepare response with sampled data
    dates = [d.strftime('%Y-%m-%d') for d in merged.index]
    
    # Get column data, handling potential NaN values
    def safe_list(arr):
        return [float(x) if not np.isnan(x) else None for x in arr]
    
    # Find column names (may vary)
    rainfall_col = 'rainfall' if 'rainfall' in merged.columns else 'precipitation'
    pet_col = 'pet' if 'pet' in merged.columns else 'evaporation'
    flow_col = 'observed_flow' if 'observed_flow' in merged.columns else 'flow'
    
    # Use observed flow range for calibration constraints (most restrictive)
    # Fall back to merged range if no observed flow
    calibration_start = observed_flow_start if observed_flow_start else original_start
    calibration_end = observed_flow_end if observed_flow_end else original_end
    calibration_days = (calibration_end - calibration_start).days + 1
    
    response = {
        "dates": dates,
        "rainfall": safe_list(merged[rainfall_col].values) if rainfall_col in merged.columns else None,
        "pet": safe_list(merged[pet_col].values) if pet_col in merged.columns else None,
        "observed_flow": safe_list(merged[flow_col].values) if flow_col in merged.columns else None,
        "data_range": {
            "start": original_start.strftime('%Y-%m-%d'),
            "end": original_end.strftime('%Y-%m-%d'),
            "total_days": original_total_days,  # Actual days, not sampled
        },
        # Calibration range = observed flow range (can't calibrate without observed data)
        "calibration_range": {
            "start": calibration_start.strftime('%Y-%m-%d'),
            "end": calibration_end.strftime('%Y-%m-%d'),
            "total_days": calibration_days,
        },
        "statistics": {
            "rainfall": {
                "mean": float(merged[rainfall_col].mean()) if rainfall_col in merged.columns else None,
                "max": float(merged[rainfall_col].max()) if rainfall_col in merged.columns else None,
                "total": float(merged[rainfall_col].sum()) if rainfall_col in merged.columns else None,
            } if rainfall_col in merged.columns else None,
            "pet": {
                "mean": float(merged[pet_col].mean()) if pet_col in merged.columns else None,
                "max": float(merged[pet_col].max()) if pet_col in merged.columns else None,
            } if pet_col in merged.columns else None,
            "observed_flow": {
                "mean": float(np.nanmean(merged[flow_col].values)) if flow_col in merged.columns else None,
                "max": float(np.nanmax(merged[flow_col].values)) if flow_col in merged.columns else None,
                "min": float(np.nanmin(merged[flow_col].values[merged[flow_col].values > 0])) if flow_col in merged.columns else None,
                "p10": float(np.nanpercentile(merged[flow_col].values, 10)) if flow_col in merged.columns else None,
                "p50": float(np.nanpercentile(merged[flow_col].values, 50)) if flow_col in merged.columns else None,
                "p90": float(np.nanpercentile(merged[flow_col].values, 90)) if flow_col in merged.columns else None,
            } if flow_col in merged.columns else None,
        }
    }
    
    return response
