"""
Data handling service for loading, validating, and processing hydrological data.

This service integrates with pyrrm's InputDataHandler and provides additional
functionality for the web application.

Data Cleaning:
    Hydrological data often contains quality issues that need to be addressed
    before calibration:
    - Sentinel values (e.g., -9999) indicating missing data
    - Negative values for physically non-negative quantities (flow, rainfall)
    - Outliers and erroneous readings
    
    This service provides functions to detect and clean these issues.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import date
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# Path setup handled by calibration.py when imported together
# This module doesn't directly import pyrrm

from app.config import get_settings

settings = get_settings()


@dataclass
class DataQualityReport:
    """
    Report on data quality issues found in a dataset.
    
    Attributes:
        total_records: Total number of records in the dataset
        sentinel_values: Count of sentinel values (e.g., -9999)
        negative_values: Count of negative values
        nan_values: Count of NaN/missing values
        zero_values: Count of zero values
        potential_outliers: Count of potential outliers (>3 std from mean)
        clean_records: Count of valid records after identifying issues
        issues: List of specific issue descriptions
        cleaning_applied: List of cleaning operations that have been applied
    """
    total_records: int = 0
    sentinel_values: int = 0
    negative_values: int = 0
    nan_values: int = 0
    zero_values: int = 0
    potential_outliers: int = 0
    clean_records: int = 0
    issues: List[str] = field(default_factory=list)
    cleaning_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Python native types."""
        return {
            'total_records': int(self.total_records),
            'sentinel_values': int(self.sentinel_values),
            'negative_values': int(self.negative_values),
            'nan_values': int(self.nan_values),
            'zero_values': int(self.zero_values),
            'potential_outliers': int(self.potential_outliers),
            'clean_records': int(self.clean_records),
            'issues': list(self.issues),
            'cleaning_applied': list(self.cleaning_applied),
            'has_issues': self.has_issues,
            'issue_percentage': float(self.issue_percentage)
        }
    
    @property
    def has_issues(self) -> bool:
        """Check if any data quality issues were found."""
        return (
            self.sentinel_values > 0 or 
            self.negative_values > 0 or 
            self.nan_values > 0
        )
    
    @property
    def issue_percentage(self) -> float:
        """Calculate percentage of records with issues."""
        if self.total_records == 0:
            return 0.0
        total_issues = self.sentinel_values + self.negative_values + self.nan_values
        return float((total_issues / self.total_records) * 100)


@dataclass
class CleaningConfig:
    """
    Configuration for data cleaning operations.
    
    Attributes:
        replace_sentinel: Replace sentinel values (-9999, -999, etc.) with NaN
        replace_negative: Replace negative values with NaN
        sentinel_values: List of sentinel values to detect/replace
        drop_na: Drop rows with NaN values after cleaning
        interpolate: Interpolate missing values (only for short gaps)
        max_interpolate_gap: Maximum gap size for interpolation (days)
    """
    replace_sentinel: bool = True
    replace_negative: bool = True
    sentinel_values: List[float] = field(default_factory=lambda: [-9999.0, -999.0, -99.0, -1.0])
    drop_na: bool = False
    interpolate: bool = False
    max_interpolate_gap: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataHandlerService:
    """
    Service for handling hydrological data operations.
    
    Provides functionality for:
    - Loading and validating CSV data files
    - Detecting column types (rainfall, PET, flow)
    - Computing statistics and previews
    - Merging datasets for calibration
    """
    
    # Column name mappings for auto-detection
    RAINFALL_COLUMNS = ['precipitation', 'rainfall', 'precip', 'rain', 'p']
    PET_COLUMNS = ['evapotranspiration', 'pet', 'evap', 'et', 'mwet']
    FLOW_COLUMNS = ['flow', 'runoff', 'discharge', 'q', 'streamflow', 'recorded']
    DATE_COLUMNS = ['date', 'datetime', 'time', 'timestamp', 'index']
    
    @staticmethod
    def detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """Detect the date column in a DataFrame."""
        columns_lower = {col.lower(): col for col in df.columns}
        
        for date_col in DataHandlerService.DATE_COLUMNS:
            if date_col in columns_lower:
                return columns_lower[date_col]
        
        # Check for datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            return None  # Already has datetime index
        
        return None
    
    @staticmethod
    def detect_data_column(df: pd.DataFrame, data_type: str) -> Optional[str]:
        """
        Detect the data column based on type.
        
        Args:
            df: DataFrame to search
            data_type: Type of data ('rainfall', 'pet', 'observed_flow')
            
        Returns:
            Column name if found, None otherwise
        """
        columns_lower = {col.lower(): col for col in df.columns}
        
        if data_type == 'rainfall':
            candidates = DataHandlerService.RAINFALL_COLUMNS
        elif data_type == 'pet':
            candidates = DataHandlerService.PET_COLUMNS
        elif data_type == 'observed_flow':
            candidates = DataHandlerService.FLOW_COLUMNS
        else:
            return None
        
        # Check for exact match first
        for candidate in candidates:
            if candidate in columns_lower:
                return columns_lower[candidate]
        
        # Check for partial match
        for col_lower, col_original in columns_lower.items():
            for candidate in candidates:
                if candidate in col_lower:
                    return col_original
        
        return None
    
    @staticmethod
    def load_csv(
        file_path: str,
        data_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a CSV file and return DataFrame with metadata.
        
        Args:
            file_path: Path to CSV file
            data_type: Expected data type ('rainfall', 'pet', 'observed_flow')
            
        Returns:
            Tuple of (DataFrame with DatetimeIndex, metadata dict)
        """
        # Read CSV
        df = pd.read_csv(file_path)
        
        metadata = {
            'original_columns': list(df.columns),
            'original_shape': df.shape,
            'warnings': [],
            'errors': []
        }
        
        # Detect and parse date column
        date_col = DataHandlerService.detect_date_column(df)
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            except Exception as e:
                metadata['errors'].append(f"Could not parse date column: {e}")
        
        # Detect data column
        data_col = DataHandlerService.detect_data_column(df, data_type)
        if data_col:
            metadata['detected_column'] = data_col
            # Standardize column name
            standard_name = {
                'rainfall': 'rainfall',
                'pet': 'pet',
                'observed_flow': 'observed_flow'
            }.get(data_type, data_type)
            df = df[[data_col]].rename(columns={data_col: standard_name})
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                standard_name = {
                    'rainfall': 'rainfall',
                    'pet': 'pet',
                    'observed_flow': 'observed_flow'
                }.get(data_type, data_type)
                df = df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: standard_name})
                metadata['warnings'].append(
                    f"Could not auto-detect {data_type} column, using '{numeric_cols[0]}'"
                )
            else:
                metadata['errors'].append("No numeric columns found in file")
        
        # Handle missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            metadata['warnings'].append(f"Found {missing_count} missing values")
        
        # Handle negative values for rainfall/PET
        if data_type in ['rainfall', 'pet']:
            col_name = list(df.columns)[0]
            neg_count = (df[col_name] < 0).sum()
            if neg_count > 0:
                metadata['warnings'].append(f"Found {neg_count} negative values")
        
        return df, metadata
    
    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> Dict[str, float]:
        """Compute summary statistics for a DataFrame."""
        if df.empty:
            return {}
        
        col = df.columns[0]
        data = df[col].dropna()
        
        return {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'count': int(len(data)),
            'missing': int(df[col].isna().sum())
        }
    
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        n_rows: int = 10
    ) -> List[Dict[str, Any]]:
        """Get a preview of the first n rows."""
        preview_df = df.head(n_rows).reset_index()
        return preview_df.to_dict('records')
    
    @staticmethod
    def merge_datasets(
        rainfall_df: pd.DataFrame,
        pet_df: pd.DataFrame,
        observed_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge rainfall, PET, and optionally observed flow datasets.
        
        Args:
            rainfall_df: DataFrame with 'rainfall' column
            pet_df: DataFrame with 'pet' column
            observed_df: Optional DataFrame with 'observed_flow' column
            
        Returns:
            Merged DataFrame with aligned dates
        """
        # Merge rainfall and PET
        merged = rainfall_df.join(pet_df, how='inner')
        
        # Add observed flow if provided
        if observed_df is not None:
            merged = merged.join(observed_df, how='inner')
        
        return merged
    
    @staticmethod
    def prepare_for_calibration(
        merged_df: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare merged data for calibration.
        
        Args:
            merged_df: Merged DataFrame with rainfall, pet, observed_flow
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (inputs DataFrame, observed array)
        """
        df = merged_df.copy()
        
        # Apply date filter
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        # Extract inputs and observed
        input_cols = ['rainfall', 'pet']
        if 'precipitation' in df.columns:
            input_cols[0] = 'precipitation'
        
        inputs = df[input_cols].copy()
        inputs.columns = ['precipitation', 'pet']
        
        observed = df['observed_flow'].values if 'observed_flow' in df.columns else None
        
        return inputs, observed
    
    @staticmethod
    def validate_dataset(
        file_path: str,
        data_type: str
    ) -> Dict[str, Any]:
        """
        Validate a dataset file.
        
        Returns validation result with warnings and errors.
        """
        result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'detected_columns': {}
        }
        
        try:
            df, metadata = DataHandlerService.load_csv(file_path, data_type)
            
            result['warnings'] = metadata.get('warnings', [])
            result['errors'] = metadata.get('errors', [])
            
            if metadata.get('detected_column'):
                result['detected_columns'][data_type] = metadata['detected_column']
            
            # Check for minimum data
            if len(df) < 365:
                result['warnings'].append(
                    f"Dataset has only {len(df)} records. Recommend at least 365 days."
                )
            
            # Check date index
            if not isinstance(df.index, pd.DatetimeIndex):
                result['errors'].append("Could not create datetime index")
            
            if result['errors']:
                result['is_valid'] = False
                
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Error reading file: {str(e)}")
        
        return result
    
    # ==========================================
    # DATA CLEANING METHODS
    # ==========================================
    
    @staticmethod
    def assess_data_quality(
        df: pd.DataFrame,
        data_type: str,
        sentinel_values: Optional[List[float]] = None
    ) -> DataQualityReport:
        """
        Assess data quality and identify issues.
        
        Args:
            df: DataFrame with data to assess
            data_type: Type of data ('rainfall', 'pet', 'observed_flow')
            sentinel_values: List of sentinel values to check for
            
        Returns:
            DataQualityReport with identified issues
        """
        if sentinel_values is None:
            sentinel_values = [-9999.0, -999.0, -99.0, -1.0]
        
        report = DataQualityReport()
        
        if df.empty:
            report.issues.append("Dataset is empty")
            return report
        
        # Get the data column
        col = df.columns[0]
        data = df[col]
        
        report.total_records = len(data)
        
        # Check for NaN values (already missing)
        report.nan_values = int(data.isna().sum())
        
        # Check for sentinel values
        for sentinel in sentinel_values:
            sentinel_count = (data == sentinel).sum()
            if sentinel_count > 0:
                report.sentinel_values += int(sentinel_count)
                report.issues.append(
                    f"Found {sentinel_count} sentinel values ({sentinel})"
                )
        
        # Check for negative values (inappropriate for rainfall, PET, flow)
        if data_type in ['rainfall', 'pet', 'observed_flow']:
            # Exclude sentinel values when counting negatives
            non_sentinel_data = data[~data.isin(sentinel_values)]
            negative_count = (non_sentinel_data < 0).sum()
            if negative_count > 0:
                report.negative_values = int(negative_count)
                report.issues.append(
                    f"Found {negative_count} negative values (physically impossible for {data_type})"
                )
        
        # Count zeros (informational)
        report.zero_values = int((data == 0).sum())
        
        # Check for outliers (>3 std from mean, for non-zero values)
        valid_data = data.replace(sentinel_values, np.nan).dropna()
        valid_data = valid_data[valid_data > 0]
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            if std_val > 0:
                outlier_threshold = mean_val + 3 * std_val
                outlier_count = (valid_data > outlier_threshold).sum()
                report.potential_outliers = int(outlier_count)
                if outlier_count > 0:
                    report.issues.append(
                        f"Found {outlier_count} potential outliers (>3 std from mean)"
                    )
        
        # Calculate clean record count
        # (excluding sentinel, negative, and NaN values)
        clean_mask = ~(
            data.isna() | 
            data.isin(sentinel_values) | 
            ((data < 0) & (~data.isin(sentinel_values)))
        )
        report.clean_records = int(clean_mask.sum())
        
        # Add summary issue if there are problems
        if report.has_issues:
            total_issues = report.sentinel_values + report.negative_values + report.nan_values
            pct = (total_issues / report.total_records) * 100
            report.issues.insert(0, 
                f"Data quality: {report.clean_records}/{report.total_records} clean records ({pct:.1f}% have issues)"
            )
        
        return report
    
    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        data_type: str,
        config: Optional[CleaningConfig] = None
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Clean dataset based on configuration.
        
        Args:
            df: DataFrame to clean
            data_type: Type of data ('rainfall', 'pet', 'observed_flow')
            config: Cleaning configuration (uses defaults if None)
            
        Returns:
            Tuple of (cleaned DataFrame, quality report)
        """
        if config is None:
            config = CleaningConfig()
        
        # First assess quality
        report = DataHandlerService.assess_data_quality(
            df, data_type, config.sentinel_values
        )
        
        if df.empty:
            return df, report
        
        col = df.columns[0]
        cleaned_df = df.copy()
        
        # Replace sentinel values with NaN
        if config.replace_sentinel and report.sentinel_values > 0:
            for sentinel in config.sentinel_values:
                mask = cleaned_df[col] == sentinel
                if mask.any():
                    cleaned_df.loc[mask, col] = np.nan
                    report.cleaning_applied.append(
                        f"Replaced {mask.sum()} sentinel values ({sentinel}) with NaN"
                    )
        
        # Replace negative values with NaN
        if config.replace_negative and data_type in ['rainfall', 'pet', 'observed_flow']:
            mask = cleaned_df[col] < 0
            if mask.any():
                neg_count = mask.sum()
                cleaned_df.loc[mask, col] = np.nan
                report.cleaning_applied.append(
                    f"Replaced {neg_count} negative values with NaN"
                )
        
        # Interpolate small gaps if requested
        if config.interpolate:
            # Only interpolate gaps up to max_interpolate_gap days
            nan_mask = cleaned_df[col].isna()
            if nan_mask.any():
                # Use linear interpolation with limit
                cleaned_df[col] = cleaned_df[col].interpolate(
                    method='linear',
                    limit=config.max_interpolate_gap,
                    limit_direction='both'
                )
                interpolated = nan_mask & ~cleaned_df[col].isna()
                if interpolated.any():
                    report.cleaning_applied.append(
                        f"Interpolated {interpolated.sum()} missing values (gaps ≤ {config.max_interpolate_gap} days)"
                    )
        
        # Drop NaN rows if requested
        if config.drop_na:
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            dropped = original_len - len(cleaned_df)
            if dropped > 0:
                report.cleaning_applied.append(
                    f"Dropped {dropped} rows with missing values"
                )
        
        # Update report with final counts
        final_nan = cleaned_df[col].isna().sum() if not cleaned_df.empty else 0
        report.clean_records = len(cleaned_df) - final_nan
        
        return cleaned_df, report
    
    @staticmethod
    def clean_file(
        file_path: str,
        data_type: str,
        config: Optional[CleaningConfig] = None,
        save: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load, clean, and optionally save a data file.
        
        Args:
            file_path: Path to input CSV file
            data_type: Type of data ('rainfall', 'pet', 'observed_flow')
            config: Cleaning configuration
            save: Whether to save the cleaned data
            output_path: Path to save cleaned data (if save=True)
            
        Returns:
            Tuple of (cleaned DataFrame, quality report)
        """
        # Load the file
        df, metadata = DataHandlerService.load_csv(file_path, data_type)
        
        # Clean the data
        cleaned_df, report = DataHandlerService.clean_data(df, data_type, config)
        
        # Save if requested
        if save:
            save_path = output_path or file_path.replace('.csv', '_cleaned.csv')
            cleaned_df.to_csv(save_path)
            report.cleaning_applied.append(f"Saved cleaned data to {save_path}")
        
        return cleaned_df, report
    
    @staticmethod
    def get_quality_report_for_file(
        file_path: str,
        data_type: str
    ) -> Dict[str, Any]:
        """
        Get a comprehensive quality report for a data file.
        
        Args:
            file_path: Path to the CSV file
            data_type: Type of data
            
        Returns:
            Dictionary with quality report and statistics
        """
        # Load the file
        df, metadata = DataHandlerService.load_csv(file_path, data_type)
        
        # Assess quality
        report = DataHandlerService.assess_data_quality(df, data_type)
        
        # Compute statistics
        stats = DataHandlerService.compute_statistics(df)
        
        return {
            'quality_report': report.to_dict(),
            'statistics': stats,
            'metadata': metadata,
            'has_issues': report.has_issues,
            'issue_percentage': report.issue_percentage
        }
