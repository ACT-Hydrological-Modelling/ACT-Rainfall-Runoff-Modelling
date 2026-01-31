"""
Dataset API endpoints.

Provides operations for uploading, validating, and managing datasets.
"""

import os
import shutil
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.catchment import Catchment
from app.models.dataset import Dataset, DatasetType
from app.schemas.dataset import DatasetResponse, DatasetPreview, DatasetValidation
from app.services.data_handler import DataHandlerService, DataQualityReport, CleaningConfig
from app.config import get_settings

settings = get_settings()
router = APIRouter()


# Pydantic models for data quality endpoints
from pydantic import BaseModel
from typing import Optional, List


class CleaningOptions(BaseModel):
    """Options for data cleaning operation."""
    replace_sentinel: bool = True
    replace_negative: bool = True
    sentinel_values: List[float] = [-9999.0, -999.0, -99.0, -1.0]
    drop_na: bool = False
    interpolate: bool = False
    max_interpolate_gap: int = 3


@router.post("/catchments/{catchment_id}/upload", response_model=DatasetResponse)
async def upload_dataset(
    catchment_id: str,
    name: str = Form(...),
    type: DatasetType = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a dataset file for a catchment.
    
    The file should be a CSV with:
    - A date column (Date, datetime, etc.)
    - A data column (rainfall, pet, flow, etc.)
    """
    # Verify catchment exists
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    # Create upload directory
    upload_dir = settings.uploads_dir / catchment_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in name if c.isalnum() or c in "._-")
    filename = f"{timestamp}_{safe_name}_{type.value}.csv"
    file_path = upload_dir / filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    finally:
        file.file.close()
    
    # Load and validate file
    try:
        df, metadata = DataHandlerService.load_csv(str(file_path), type.value)
        
        if metadata.get('errors'):
            # Clean up file on error
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file: {'; '.join(metadata['errors'])}"
            )
        
        # Extract date range
        start_date = df.index.min().date() if hasattr(df.index, 'min') else None
        end_date = df.index.max().date() if hasattr(df.index, 'max') else None
        record_count = len(df)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {str(e)}"
        )
    
    # Create database record
    db_dataset = Dataset(
        catchment_id=catchment_id,
        name=name,
        type=type,
        file_path=str(file_path),
        start_date=start_date,
        end_date=end_date,
        record_count=record_count,
        extra_metadata=metadata
    )
    
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset


@router.get("/catchments/{catchment_id}", response_model=List[DatasetResponse])
def list_datasets(
    catchment_id: str,
    db: Session = Depends(get_db)
):
    """
    List all datasets for a catchment.
    """
    # Verify catchment exists
    catchment = db.query(Catchment).filter(Catchment.id == catchment_id).first()
    if not catchment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Catchment with id '{catchment_id}' not found"
        )
    
    return catchment.datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a dataset by ID.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    n_rows: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get a preview of a dataset with statistics.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    try:
        df, metadata = DataHandlerService.load_csv(dataset.file_path, dataset.type.value)
        statistics = DataHandlerService.compute_statistics(df)
        sample_data = DataHandlerService.get_preview(df, n_rows)
        
        col_name = df.columns[0]
        missing_count = int(df[col_name].isna().sum())
        missing_pct = missing_count / len(df) * 100 if len(df) > 0 else 0
        
        return DatasetPreview(
            id=dataset.id,
            name=dataset.name,
            type=dataset.type,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
            record_count=len(df),
            statistics=statistics,
            sample_data=sample_data,
            missing_count=missing_count,
            missing_percentage=round(missing_pct, 2),
            warnings=metadata.get('warnings', []),
            errors=metadata.get('errors', [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset: {str(e)}"
        )


@router.get("/{dataset_id}/validate", response_model=DatasetValidation)
def validate_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Validate a dataset and return detailed validation results.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    result = DataHandlerService.validate_dataset(dataset.file_path, dataset.type.value)
    return DatasetValidation(**result)


@router.get("/{dataset_id}/plot")
def plot_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a time series plot of the dataset as Plotly JSON.
    """
    import plotly.graph_objects as go
    import math
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    try:
        df, _ = DataHandlerService.load_csv(dataset.file_path, dataset.type.value)
        col_name = df.columns[0]
        
        # Get unit label based on type
        type_labels = {
            'rainfall': 'Rainfall (mm/day)',
            'pet': 'PET (mm/day)',
            'observed_flow': 'Flow (ML/day)'
        }
        y_label = type_labels.get(dataset.type.value, 'Value')
        
        # Create time series plot
        fig = go.Figure()
        
        # Sanitize values (replace inf/nan with None)
        values = df[col_name].values.tolist()
        dates = df.index.strftime('%Y-%m-%d').tolist()
        sanitized_values = [None if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else v for v in values]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=sanitized_values,
            mode='lines',
            name=dataset.name,
            line=dict(color='#3b82f6', width=1)
        ))
        
        fig.update_layout(
            title=f'{dataset.name}',
            xaxis_title='Date',
            yaxis_title=y_label,
            template='plotly_white',
            height=400,
            margin=dict(l=60, r=20, t=50, b=50)
        )
        
        return fig.to_dict()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create plot: {str(e)}"
        )


@router.get("/{dataset_id}/statistics")
def get_dataset_statistics(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for a dataset including hydrologic signatures for flow data
    and data quality assessment.
    """
    import math
    import numpy as np
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    try:
        df, _ = DataHandlerService.load_csv(dataset.file_path, dataset.type.value)
        col_name = df.columns[0]
        values = df[col_name].dropna().values
        
        def safe_float(val):
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return None
            return float(val)
        
        # ====================================
        # Data Quality Assessment
        # ====================================
        quality_report = DataHandlerService.assess_data_quality(df, dataset.type.value)
        
        # Get cleaning history from dataset metadata
        cleaning_history = []
        if dataset.extra_metadata and 'cleaning_applied' in dataset.extra_metadata:
            cleaning_history = dataset.extra_metadata['cleaning_applied']
        
        quality_info = {
            'total_records': quality_report.total_records,
            'clean_records': quality_report.clean_records,
            'sentinel_values': quality_report.sentinel_values,
            'negative_values': quality_report.negative_values,
            'nan_values': quality_report.nan_values,
            'zero_values': quality_report.zero_values,
            'potential_outliers': quality_report.potential_outliers,
            'has_issues': quality_report.has_issues,
            'issue_percentage': safe_float(quality_report.issue_percentage),
            'issues': quality_report.issues,
            'cleaning_applied': cleaning_history if cleaning_history else quality_report.cleaning_applied
        }
        
        # ====================================
        # Basic statistics
        # ====================================
        stats = {
            'count': len(values),
            'missing': int(df[col_name].isna().sum()),
            'missing_pct': safe_float(df[col_name].isna().sum() / len(df) * 100 if len(df) > 0 else 0),
            'min': safe_float(np.min(values)) if len(values) > 0 else None,
            'max': safe_float(np.max(values)) if len(values) > 0 else None,
            'mean': safe_float(np.mean(values)) if len(values) > 0 else None,
            'median': safe_float(np.median(values)) if len(values) > 0 else None,
            'std': safe_float(np.std(values)) if len(values) > 0 else None,
            'sum': safe_float(np.sum(values)) if len(values) > 0 else None,
        }
        
        # Add percentiles
        if len(values) > 0:
            for p in [5, 10, 25, 75, 90, 95]:
                stats[f'p{p}'] = safe_float(np.percentile(values, p))
        
        # Hydrologic signatures for flow data
        signatures = {}
        if dataset.type.value == 'observed_flow' and len(values) > 0:
            # Coefficient of variation
            signatures['cv'] = safe_float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else None
            
            # Baseflow index (simplified - ratio of 10th percentile to mean)
            signatures['baseflow_index'] = safe_float(np.percentile(values, 10) / np.mean(values)) if np.mean(values) != 0 else None
            
            # High flow index (Q10/Q50)
            q10 = np.percentile(values, 10)  # flow exceeded 90% of time
            q50 = np.percentile(values, 50)
            signatures['high_flow_index'] = safe_float(q10 / q50) if q50 != 0 else None
            
            # Low flow index (Q90/Q50)
            q90 = np.percentile(values, 90)  # flow exceeded 10% of time
            signatures['low_flow_index'] = safe_float(q90 / q50) if q50 != 0 else None
            
            # Runoff ratio (if we had precip, this would be more meaningful)
            signatures['mean_daily_flow'] = safe_float(np.mean(values))
            signatures['total_volume'] = safe_float(np.sum(values))
            
            # Zero flow days
            zero_days = int(np.sum(values <= 0.001))
            signatures['zero_flow_days'] = zero_days
            signatures['zero_flow_pct'] = safe_float(zero_days / len(values) * 100)
        
        # Seasonal statistics for rainfall/PET
        seasonal_stats = {}
        if dataset.type.value in ['rainfall', 'pet'] and len(df) > 0:
            df['month'] = df.index.month
            monthly_means = df.groupby('month')[col_name].mean()
            for month, val in monthly_means.items():
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                seasonal_stats[month_names[month-1]] = safe_float(val)
        
        return {
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'dataset_type': dataset.type.value,
            'date_range': {
                'start': str(dataset.start_date) if dataset.start_date else None,
                'end': str(dataset.end_date) if dataset.end_date else None
            },
            'data_quality': quality_info,
            'statistics': stats,
            'signatures': signatures if signatures else None,
            'seasonal': seasonal_stats if seasonal_stats else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute statistics: {str(e)}"
        )


@router.get("/{dataset_id}/quality")
def get_data_quality(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get data quality assessment for a dataset.
    
    This endpoint identifies common data quality issues:
    - Sentinel values (e.g., -9999 used for missing data)
    - Negative values (physically impossible for rainfall/PET/flow)
    - Missing values (NaN)
    - Potential outliers
    
    Returns a detailed report with recommendations for cleaning.
    """
    import math
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    try:
        # Load the data
        df, metadata = DataHandlerService.load_csv(dataset.file_path, dataset.type.value)
        
        # Assess quality
        report = DataHandlerService.assess_data_quality(df, dataset.type.value)
        
        # Build response
        def safe_float(val):
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return None
            return float(val)
        
        return {
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'dataset_type': dataset.type.value,
            'quality_report': {
                'total_records': report.total_records,
                'clean_records': report.clean_records,
                'sentinel_values': report.sentinel_values,
                'negative_values': report.negative_values,
                'nan_values': report.nan_values,
                'zero_values': report.zero_values,
                'potential_outliers': report.potential_outliers,
                'issue_percentage': safe_float(report.issue_percentage),
                'has_issues': report.has_issues,
                'issues': report.issues
            },
            'recommendations': _get_cleaning_recommendations(report, dataset.type.value)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess data quality: {str(e)}"
        )


def _get_cleaning_recommendations(report: DataQualityReport, data_type: str) -> List[str]:
    """Generate cleaning recommendations based on quality report."""
    recommendations = []
    
    if report.sentinel_values > 0:
        recommendations.append(
            f"Replace {report.sentinel_values} sentinel values (-9999, etc.) with NaN"
        )
    
    if report.negative_values > 0 and data_type in ['rainfall', 'pet', 'observed_flow']:
        type_name = {
            'rainfall': 'Rainfall',
            'pet': 'Evapotranspiration',
            'observed_flow': 'Streamflow'
        }.get(data_type, data_type)
        recommendations.append(
            f"Replace {report.negative_values} negative values with NaN ({type_name} cannot be negative)"
        )
    
    total_missing = report.sentinel_values + report.negative_values + report.nan_values
    if total_missing > 0:
        pct = (total_missing / report.total_records * 100) if report.total_records > 0 else 0
        recommendations.append(
            f"After cleaning, {pct:.1f}% of data will be marked as missing. Consider interpolation for short gaps."
        )
    
    if not recommendations:
        recommendations.append("Data quality is good - no cleaning required.")
    
    return recommendations


@router.post("/{dataset_id}/clean")
def clean_dataset(
    dataset_id: str,
    options: CleaningOptions = CleaningOptions(),
    db: Session = Depends(get_db)
):
    """
    Clean a dataset by removing/replacing problematic values.
    
    This operation:
    1. Replaces sentinel values (-9999, etc.) with NaN
    2. Replaces negative values with NaN (for rainfall/PET/flow)
    3. Optionally drops rows with NaN values
    4. Optionally interpolates short gaps
    
    The cleaned data is saved to a new file and the dataset is updated.
    """
    import math
    import numpy as np
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    try:
        # Create cleaning config from options
        config = CleaningConfig(
            replace_sentinel=options.replace_sentinel,
            replace_negative=options.replace_negative,
            sentinel_values=options.sentinel_values,
            drop_na=options.drop_na,
            interpolate=options.interpolate,
            max_interpolate_gap=options.max_interpolate_gap
        )
        
        # Load and clean the data
        original_path = dataset.file_path
        cleaned_path = original_path.replace('.csv', '_cleaned.csv')
        
        cleaned_df, report = DataHandlerService.clean_file(
            file_path=original_path,
            data_type=dataset.type.value,
            config=config,
            save=True,
            output_path=cleaned_path
        )
        
        # Helper to convert numpy types to Python native types
        def to_python_int(val):
            if val is None:
                return 0
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            return int(val)
        
        def safe_float(val):
            if val is None:
                return None
            if isinstance(val, (np.floating, np.float64, np.float32)):
                val = float(val)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return float(val)
        
        # Update dataset record
        if cleaned_df is not None and len(cleaned_df) > 0:
            from sqlalchemy.orm.attributes import flag_modified
            
            dataset.file_path = cleaned_path
            dataset.record_count = to_python_int(len(cleaned_df))
            dataset.start_date = cleaned_df.index.min().date() if hasattr(cleaned_df.index, 'min') else dataset.start_date
            dataset.end_date = cleaned_df.index.max().date() if hasattr(cleaned_df.index, 'max') else dataset.end_date
            
            # Store cleaning info in metadata (use dict() to ensure new object)
            existing_metadata = dict(dataset.extra_metadata or {})
            existing_metadata['cleaning_applied'] = list(report.cleaning_applied)
            existing_metadata['original_file'] = original_path
            dataset.extra_metadata = existing_metadata
            flag_modified(dataset, 'extra_metadata')  # Required for JSON column updates
            
            db.commit()
            db.refresh(dataset)
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'cleaning_report': {
                'original_records': to_python_int(report.total_records),
                'clean_records': to_python_int(report.clean_records),
                'records_after_cleaning': to_python_int(len(cleaned_df)) if cleaned_df is not None else 0,
                'operations_applied': list(report.cleaning_applied),
                'issue_percentage': safe_float(report.issue_percentage)
            },
            'message': f"Dataset cleaned successfully. {len(report.cleaning_applied)} operations applied."
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean dataset: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a dataset and its file.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id '{dataset_id}' not found"
        )
    
    # Delete file
    file_path = Path(dataset.file_path)
    if file_path.exists():
        os.remove(file_path)
    
    # Delete database record
    db.delete(dataset)
    db.commit()
    
    return None
