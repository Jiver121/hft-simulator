"""
Data processing module for HFT Simulator

This module handles data ingestion, preprocessing, and validation for
high-frequency trading datasets, particularly from Kaggle sources.
"""

from .ingestion import DataIngestion
from .preprocessor import DataPreprocessor
from .validators import DataValidator

__all__ = [
    'DataIngestion',
    'DataPreprocessor', 
    'DataValidator',
]