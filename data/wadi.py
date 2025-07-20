import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import os
from .default_ts import DefaultTimeSeriesDataset


class MemoryEfficientWADIDataset(Dataset):
    """
    Memory-efficient WADI Dataset that loads windows on-demand
    Avoids storing all windows in memory at once
    """
    
    def __init__(
        self,
        data_dir: str,
        filename: str = None,
        window_size: int = 50,
        stride: int = 10,
        transform=None,
        split: str = 'train',
        mean: Optional[float] = None,
        std: Optional[float] = None,
        val_ratio: float = 0.2,
        feature_dim: Optional[int] = None,
        train_ratio: float = None,
        max_samples: Optional[int] = None  # New parameter to limit dataset size
    ):
        """
        Initialize memory-efficient WADI dataset
        
        Args:
            data_dir: Directory containing WADI data files
            filename: Not used (kept for compatibility)
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean
            std: Normalization std
            val_ratio: Ratio of training data to use for validation
            feature_dim: Expected feature dimension (should be 127 for WADI)
            train_ratio: Not used (kept for compatibility)
            max_samples: Maximum number of samples to use (for memory limiting)
        """
        self.data_dir = data_dir
        self.split = split
        self.val_ratio = val_ratio
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.max_samples = max_samples
        
        # Load raw data but don't create windows yet
        self.data, self.labels = self._load_wadi_data()
        
        # Apply data size limiting if specified
        if self.max_samples is not None and len(self.data) > self.max_samples:
            print(f"Limiting dataset to {self.max_samples} samples (was {len(self.data)})")
            self.data = self.data[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        # Compute normalization statistics
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True) + 1e-8
        
        # Normalize data
        self.data = (self.data - self.mean) / self.std
        
        # Calculate number of windows
        self.n_windows = (len(self.data) - self.window_size) // self.stride + 1
        
        print(f"Memory-efficient WADI {self.split}: {len(self.data)} samples -> {self.n_windows} windows")
        
    def _load_wadi_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load WADI data from pre-split train.csv and test.csv files"""
        
        if self.split in ['train', 'val']:
            # Load training data (contains only normal samples)
            data_path = os.path.join(self.data_dir, 'train.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"WADI training data not found: {data_path}")
            
            # Read CSV file with normal header
            df = pd.read_csv(data_path, low_memory=False)
            
            # Drop Row, Date, Time columns
            columns_to_drop = ['Row', 'Date', 'Time']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # All training data are normal samples (label = 0)
            labels = np.zeros(len(df), dtype=np.int64)
            
            # Extract features (all remaining columns)
            feature_data = df
            
        else:  # test
            # Load test data (contains both normal and anomalous samples)
            data_path = os.path.join(self.data_dir, 'test.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"WADI test data not found: {data_path}")
            
            # Read CSV file with proper header handling
            df = pd.read_csv(data_path, header=1, low_memory=False)
            
            # Drop Row, Date, Time columns
            columns_to_drop = ['Row ', 'Date ', 'Time']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Extract labels (last column)
            label_column = df.columns[-1]
            raw_labels = df[label_column].values
            
            # Convert labels: 1 (No Attack) -> 0 (normal), -1 (Attack) -> 1 (anomaly)
            labels = np.where(raw_labels == 1, 0, 1).astype(np.int64)
            
            # Extract features (all columns except the last one)
            feature_columns = df.columns[:-1]
            feature_data = df[feature_columns]
        
        # Handle missing values and convert to float
        feature_data = feature_data.fillna(0.0)
        
        # Convert to numeric, forcing errors to NaN then filling with 0
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
        
        data = feature_data.values.astype(np.float32)
        
        # Verify feature dimension
        if self.feature_dim is not None and data.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features for WADI, got {data.shape[1]}")
        
        # Handle train/val split for training data
        if self.split == 'train':
            # Use most of the training data, excluding validation portion
            val_samples = int(len(data) * self.val_ratio)
            if val_samples > 0:
                data = data[:-val_samples]
                labels = labels[:-val_samples]
        elif self.split == 'val':
            # Use validation portion from training data
            val_samples = int(len(data) * self.val_ratio)
            if val_samples > 0:
                data = data[-val_samples:]
                labels = labels[-val_samples:]
            else:
                # If no validation samples, use a small portion from the end
                data = data[-self.window_size:]
                labels = labels[-self.window_size:]
        
        return data, labels
    
    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single window on-demand"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Create window on-demand
        window = self.data[start_idx:end_idx]
        window_label = np.max(self.labels[start_idx:end_idx])
        
        # Convert to tensor
        window = torch.from_numpy(window).float()
        label = torch.tensor(window_label, dtype=torch.long)
        
        # Apply transforms
        if self.transform is not None:
            window = self.transform(window)
        
        return window, label


class WADIDataset(DefaultTimeSeriesDataset):
    """
    WADI (Water Distribution) Dataset
    Contains industrial control system data with anomaly detection tasks
    
    WADI files: Pre-split train.csv and test.csv with 127 features after preprocessing
    """
    
    def __init__(
        self,
        data_dir: str,
        filename: str = None,  # Not used for pre-split files
        window_size: int = 50,
        stride: int = 1,
        transform=None,
        split: str = 'train',
        mean: Optional[float] = None,
        std: Optional[float] = None,
        val_ratio: float = 0.2,
        feature_dim: Optional[int] = None,
        train_ratio: float = None,  # Not used for pre-split files
        use_memory_efficient: bool = True,  # New parameter
        max_samples: Optional[int] = None   # New parameter
    ):
        """
        Initialize WADI dataset
        
        Args:
            data_dir: Directory containing WADI data files (train.csv, test.csv)
            filename: Not used (kept for compatibility)
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean
            std: Normalization std
            val_ratio: Ratio of training data to use for validation
            feature_dim: Expected feature dimension (should be 127 for WADI)
            train_ratio: Not used (kept for compatibility)
            use_memory_efficient: Whether to use memory-efficient loading
            max_samples: Maximum number of samples to use (for memory limiting)
        """
        # Use memory-efficient version by default for large datasets
        if use_memory_efficient:
            # Initialize as memory-efficient dataset
            self.memory_efficient_dataset = MemoryEfficientWADIDataset(
                data_dir=data_dir,
                filename=filename,
                window_size=window_size,
                stride=stride,
                transform=transform,
                split=split,
                mean=mean,
                std=std,
                val_ratio=val_ratio,
                feature_dim=feature_dim,
                train_ratio=train_ratio,
                max_samples=max_samples
            )
            
            # Copy attributes for compatibility
            self.mean = self.memory_efficient_dataset.mean
            self.std = self.memory_efficient_dataset.std
            self.data = self.memory_efficient_dataset.data
            self.labels = self.memory_efficient_dataset.labels
            self.window_size = window_size
            self.stride = stride
            self.transform = transform
            self.split = split
            
        else:
            # Use original implementation
            self.data_dir = data_dir
            self.split = split
            self.val_ratio = val_ratio
            self.feature_dim = feature_dim
            self.window_size = window_size
            
            # Load data
            data, labels = self._load_wadi_data()
            
            # Apply data size limiting if specified
            if max_samples is not None and len(data) > max_samples:
                print(f"Limiting dataset to {max_samples} samples (was {len(data)})")
                data = data[:max_samples]
                labels = labels[:max_samples]
            
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
    
    def __len__(self) -> int:
        if hasattr(self, 'memory_efficient_dataset'):
            return len(self.memory_efficient_dataset)
        else:
            return super().__len__()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if hasattr(self, 'memory_efficient_dataset'):
            return self.memory_efficient_dataset[idx]
        else:
            return super().__getitem__(idx)
    
    def _load_wadi_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load WADI data from pre-split train.csv and test.csv files"""
        
        if self.split in ['train', 'val']:
            # Load training data (contains only normal samples)
            data_path = os.path.join(self.data_dir, 'train.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"WADI training data not found: {data_path}")
            
            # Read CSV file with normal header
            df = pd.read_csv(data_path, low_memory=False)
            
            # Drop Row, Date, Time columns
            columns_to_drop = ['Row', 'Date', 'Time']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # All training data are normal samples (label = 0)
            labels = np.zeros(len(df), dtype=np.int64)
            
            # Extract features (all remaining columns)
            feature_data = df
            
        else:  # test
            # Load test data (contains both normal and anomalous samples)
            data_path = os.path.join(self.data_dir, 'test.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"WADI test data not found: {data_path}")
            
            # Read CSV file with proper header handling
            # The first row is index numbers, second row is actual column names
            df = pd.read_csv(data_path, header=1, low_memory=False)
            
            # Drop Row, Date, Time columns
            columns_to_drop = ['Row ', 'Date ', 'Time']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Extract labels (last column)
            label_column = df.columns[-1]  # "Attack LABLE (1:No Attack, -1:Attack)"
            raw_labels = df[label_column].values
            
            # Convert labels: 1 (No Attack) -> 0 (normal), -1 (Attack) -> 1 (anomaly)
            labels = np.where(raw_labels == 1, 0, 1).astype(np.int64)
            
            # Extract features (all columns except the last one)
            feature_columns = df.columns[:-1]
            feature_data = df[feature_columns]
        
        # Handle missing values and convert to float
        feature_data = feature_data.fillna(0.0)  # Fill NaN with 0
        
        # Convert to numeric, forcing errors to NaN then filling with 0
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
        
        data = feature_data.values.astype(np.float32)
        
        # Verify feature dimension
        if self.feature_dim is not None and data.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features for WADI, got {data.shape[1]}")
        
        # Handle train/val split for training data
        if self.split == 'train':
            # Use most of the training data, excluding validation portion
            val_samples = int(len(data) * self.val_ratio)
            if val_samples > 0:
                data = data[:-val_samples]
                labels = labels[:-val_samples]
        elif self.split == 'val':
            # Use validation portion from training data
            val_samples = int(len(data) * self.val_ratio)
            if val_samples > 0:
                data = data[-val_samples:]
                labels = labels[-val_samples:]
            else:
                # If no validation samples, use a small portion from the end
                data = data[-self.window_size:]
                labels = labels[-self.window_size:]
        
        print(f"WADI {self.split} split: {len(data)} samples, {data.shape[1]} features")
        if self.split in ['train', 'val']:
            anomaly_count = np.sum(labels)
            print(f"  {self.split.capitalize()} anomalies: {anomaly_count} (should be 0 for proper anomaly detection)")
        else:  # test
            anomaly_count = np.sum(labels)
            anomaly_ratio = anomaly_count / len(labels) if len(labels) > 0 else 0
            print(f"  Test anomalies: {anomaly_count} ({anomaly_ratio:.4f} ratio)")
        
        return data, labels
    
    @staticmethod
    def get_wadi_info(data_dir: str) -> dict:
        """Get WADI dataset information from pre-split train and test files"""
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"WADI training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"WADI test data not found: {test_path}")
        
        # Read training data
        train_df = pd.read_csv(train_path, low_memory=False)
        train_df = train_df.drop(columns=['Row', 'Date', 'Time'], errors='ignore')
        train_samples = len(train_df)
        
        # Read test data
        test_df = pd.read_csv(test_path, header=1, low_memory=False)
        test_df = test_df.drop(columns=['Row ', 'Date ', 'Time'], errors='ignore')
        
        # Extract test labels
        label_column = test_df.columns[-1]
        raw_labels = test_df[label_column].values
        test_labels = np.where(raw_labels == 1, 0, 1)
        
        test_samples = len(test_df)
        test_normal_count = np.sum(test_labels == 0)
        test_anomaly_count = np.sum(test_labels == 1)
        
        # Calculate total statistics
        total_samples = train_samples + test_samples
        total_normal_count = train_samples + test_normal_count  # All training data is normal
        total_anomaly_count = test_anomaly_count
        
        feature_count = len(train_df.columns)  # All columns are features in training data
        
        return {
            'total_samples': total_samples,
            'feature_count': feature_count,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'normal_count': total_normal_count,
            'anomaly_count': total_anomaly_count,
            'anomaly_ratio': total_anomaly_count / total_samples,
            'test_anomaly_ratio': test_anomaly_count / test_samples,
            'label_column': label_column,
            'feature_columns': train_df.columns.tolist()
        }


def load_wadi_data(data_dir: str, val_ratio: float = 0.2, max_samples: Optional[int] = None) -> Tuple[WADIDataset, WADIDataset, WADIDataset]:
    """
    Load WADI train, validation, and test datasets from pre-split files
    
    Args:
        data_dir: Directory containing WADI data (train.csv, test.csv)
        val_ratio: Ratio of training data to use for validation
        max_samples: Maximum number of samples to use (for memory limiting)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = WADIDataset(
        data_dir, 
        split='train', 
        val_ratio=val_ratio,
        feature_dim=127,
        use_memory_efficient=True,
        max_samples=max_samples
    )
    val_dataset = WADIDataset(
        data_dir,
        split='val',
        val_ratio=val_ratio,
        feature_dim=127,
        mean=train_dataset.mean,
        std=train_dataset.std,
        use_memory_efficient=True,
        max_samples=max_samples
    )
    test_dataset = WADIDataset(
        data_dir,
        split='test',
        val_ratio=val_ratio,
        feature_dim=127,
        mean=train_dataset.mean,
        std=train_dataset.std,
        use_memory_efficient=True,
        max_samples=max_samples
    )
    
    return train_dataset, val_dataset, test_dataset