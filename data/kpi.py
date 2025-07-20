import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import os
import pickle
try:
    from .default_ts import DefaultTimeSeriesDataset
except ImportError:
    from default_ts import DefaultTimeSeriesDataset


class KPIDataset(DefaultTimeSeriesDataset):
    """
    KPI Dataset for anomaly detection
    Contains 29 KPI time series with anomaly labels
    """
    
    def __init__(
        self,
        data_dir: str,
        kpi_id: Optional[str] = None,
        window_size: int = 100,
        stride: int = 1,
        transform=None,
        split: str = 'train',
        mean: Optional[float] = None,
        std: Optional[float] = None,
        val_ratio: float = 0.1
    ):
        """
        Initialize KPI dataset
        
        Args:
            data_dir: Directory containing KPI data files
            kpi_id: KPI ID to use (None for all KPIs)
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean
            std: Normalization std
            val_ratio: Ratio of training data to use for validation
        """
        self.data_dir = data_dir
        self.kpi_id = kpi_id
        self.split = split
        self.val_ratio = val_ratio
        
        # Load data
        data, labels = self._load_kpi_data()
        
        # Split data based on split type
        if split == 'train':
            # Use first 70% for training
            split_idx = int(len(data) * 0.7)
            data = data[:split_idx]
            labels = labels[:split_idx]
        elif split == 'val':
            # Use next 10% for validation
            split_idx_start = int(len(data) * 0.7)
            split_idx_end = int(len(data) * 0.8)
            data = data[split_idx_start:split_idx_end]
            labels = labels[split_idx_start:split_idx_end]
        elif split == 'test':
            # Use last 20% for testing
            split_idx = int(len(data) * 0.8)
            data = data[split_idx:]
            labels = labels[split_idx:]
        
        # Initialize parent class
        super().__init__(
            data=data,
            labels=labels,
            window_size=window_size,
            stride=stride,
            transform=transform,
            split=split,
            mean=mean,
            std=std
        )
    
    def _load_kpi_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load KPI data from CSV file"""
        csv_path = os.path.join(self.data_dir, "KPI", "KPI.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"KPI data file not found at {csv_path}")
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        if self.kpi_id is None:
            # Use all KPIs
            return self._load_all_kpis(df)
        else:
            # Use specific KPI
            return self._load_single_kpi(df, self.kpi_id)
    
    def _load_single_kpi(self, df: pd.DataFrame, kpi_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data for a single KPI"""
        # Filter data for the specific KPI
        kpi_data = df[df['KPI ID'] == kpi_id].copy()
        
        if len(kpi_data) == 0:
            raise ValueError(f"No data found for KPI ID: {kpi_id}")
        
        # Sort by timestamp
        kpi_data = kpi_data.sort_values('timestamp')
        
        # Extract values and labels
        values = kpi_data['value'].values
        labels = kpi_data['label'].values
        
        # Reshape for univariate time series (add feature dimension)
        data = values.reshape(-1, 1)
        
        return data, labels
    
    def _load_all_kpis(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate all KPIs"""
        all_data = []
        all_labels = []
        
        # Get all unique KPI IDs
        kpi_ids = df['KPI ID'].unique()
        
        for kpi_id in kpi_ids:
            kpi_data = df[df['KPI ID'] == kpi_id].copy()
            kpi_data = kpi_data.sort_values('timestamp')
            
            values = kpi_data['value'].values
            labels = kpi_data['label'].values
            
            # Reshape for univariate time series
            data = values.reshape(-1, 1)
            
            all_data.append(data)
            all_labels.append(labels)
        
        # Concatenate all KPIs
        concatenated_data = np.concatenate(all_data, axis=0)
        concatenated_labels = np.concatenate(all_labels, axis=0)
        
        return concatenated_data, concatenated_labels
    
    def get_available_kpis(self) -> List[str]:
        """Get list of available KPI IDs"""
        csv_path = os.path.join(self.data_dir, "KPI", "KPI.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"KPI data file not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df['KPI ID'].unique().tolist()
    
    def get_kpi_stats(self, kpi_id: Optional[str] = None) -> dict:
        """Get statistics for a specific KPI or all KPIs"""
        csv_path = os.path.join(self.data_dir, "KPI", "KPI.csv")
        df = pd.read_csv(csv_path)
        
        if kpi_id is not None:
            df = df[df['KPI ID'] == kpi_id]
        
        stats = {
            'total_points': len(df),
            'anomaly_points': (df['label'] == 1).sum(),
            'normal_points': (df['label'] == 0).sum(),
            'anomaly_rate': df['label'].mean(),
            'value_mean': df['value'].mean(),
            'value_std': df['value'].std(),
            'value_min': df['value'].min(),
            'value_max': df['value'].max(),
            'unique_kpis': df['KPI ID'].nunique() if kpi_id is None else 1
        }
        
        return stats


def load_kpi_data(
    data_dir: str, 
    kpi_id: Optional[str] = None, 
    val_ratio: float = 0.1
) -> Tuple[KPIDataset, KPIDataset, KPIDataset]:
    """
    Load KPI dataset and split into train/val/test
    
    Args:
        data_dir: Directory containing KPI data
        kpi_id: KPI ID to use (None for all KPIs)
        val_ratio: Ratio of training data to use for validation
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load datasets
    train_dataset = KPIDataset(
        data_dir=data_dir,
        kpi_id=kpi_id,
        split='train',
        val_ratio=val_ratio
    )
    
    val_dataset = KPIDataset(
        data_dir=data_dir,
        kpi_id=kpi_id,
        split='val',
        mean=train_dataset.mean,
        std=train_dataset.std,
        val_ratio=val_ratio
    )
    
    test_dataset = KPIDataset(
        data_dir=data_dir,
        kpi_id=kpi_id,
        split='test',
        mean=train_dataset.mean,
        std=train_dataset.std,
        val_ratio=val_ratio
    )
    
    return train_dataset, val_dataset, test_dataset


def get_available_kpi_ids(data_dir: str) -> List[str]:
    """Get list of available KPI IDs"""
    csv_path = os.path.join(data_dir, "KPI", "KPI.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KPI data file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df['KPI ID'].unique().tolist()


def get_kpi_summary(data_dir: str) -> pd.DataFrame:
    """Get summary statistics for all KPIs"""
    csv_path = os.path.join(data_dir, "KPI", "KPI.csv")
    df = pd.read_csv(csv_path)
    
    # Group by KPI ID and calculate statistics
    summary = df.groupby('KPI ID').agg({
        'value': ['count', 'mean', 'std', 'min', 'max'],
        'label': ['sum', 'mean']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Rename columns for clarity
    summary = summary.rename(columns={
        'value_count': 'total_points',
        'value_mean': 'value_mean',
        'value_std': 'value_std',
        'value_min': 'value_min',
        'value_max': 'value_max',
        'label_sum': 'anomaly_points',
        'label_mean': 'anomaly_rate'
    })
    
    return summary.reset_index() 