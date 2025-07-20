import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import os
from .default_ts import DefaultTimeSeriesDataset


class SMDDataset(DefaultTimeSeriesDataset):
    """
    SMD (Server Machine Dataset) Dataset
    Contains machine monitoring data with anomaly detection tasks
    """
    
    def __init__(
        self,
        data_dir: str,
        filename: str,
        window_size: int = 100,
        stride: int = 1,
        transform=None,
        split: str = 'train',
        mean: Optional[float] = None,
        std: Optional[float] = None,
        val_ratio: float = 0.2
    ):
        """
        Initialize SMD dataset
        
        Args:
            data_dir: Directory containing SMD data files
            filename: Data filename (e.g., 'machine-1-1', 'machine-2-5')
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean
            std: Normalization std
            val_ratio: Ratio of training data to use for validation
        """
        self.data_dir = data_dir
        self.filename = filename
        self.split = split
        self.val_ratio = val_ratio
        
        # Load data
        data, labels = self._load_smd_data()
        
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
    
    def _load_smd_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SMD data from text files"""
        if self.filename is None:
            # Load all machine files (machine-1-1 to machine-3-11) and aggregate
            return self._load_all_machine_files_aggregated()
        else:
            # Load specific file
            return self._load_single_machine_file()
    
    def _load_single_machine_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single machine file"""
        if self.split == 'train':
            # Load training data
            data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.txt')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            # Load CSV data
            data = pd.read_csv(data_path, header=None)
            data = data.values.astype(np.float32)
            
            # Training data assumed to be normal
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Split for training only (keep validation separate)
            val_split_idx = int(len(data) * (1 - self.val_ratio))
            data = data[:val_split_idx]
            labels = labels[:val_split_idx]
            
        elif self.split == 'val':
            # Load training data for validation split
            data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.txt')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            # Load CSV data
            data = pd.read_csv(data_path, header=None)
            data = data.values.astype(np.float32)
            
            # Training data assumed to be normal
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Split for validation only
            val_split_idx = int(len(data) * (1 - self.val_ratio))
            data = data[val_split_idx:]
            labels = labels[val_split_idx:]
            
        else:  # test
            # Load test data and labels
            data_path = os.path.join(self.data_dir, 'test', f'{self.filename}.txt')
            label_path = os.path.join(self.data_dir, 'test_label', f'{self.filename}.txt')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Test data not found: {data_path}")
            
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Test labels not found: {label_path}")
            
            # Load CSV data
            data = pd.read_csv(data_path, header=None)
            data = data.values.astype(np.float32)
            
            # Load labels (one per line)
            with open(label_path, 'r') as f:
                labels = np.array([int(line.strip()) for line in f.readlines()], dtype=np.int64)
            
            # Verify data and labels have same length
            if len(data) != len(labels):
                min_len = min(len(data), len(labels))
                print(f"Warning: Data length ({len(data)}) != Label length ({len(labels)}). Truncating to {min_len}")
                data = data[:min_len]
                labels = labels[:min_len]
        
        return data, labels
    
    def _load_all_machine_files_aggregated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all machine files and aggregate them"""
        # Get all machine files
        if self.split == 'train':
            machine_dir = os.path.join(self.data_dir, 'train')
        elif self.split == 'val':
            machine_dir = os.path.join(self.data_dir, 'train')  # Use train files for validation split
        else:  # test
            machine_dir = os.path.join(self.data_dir, 'test')
        
        if not os.path.exists(machine_dir):
            raise FileNotFoundError(f"SMD directory not found: {machine_dir}")
        
        # Get all machine files
        machine_files = [f for f in os.listdir(machine_dir) if f.startswith('machine-') and f.endswith('.txt')]
        
        if not machine_files:
            raise ValueError(f"No machine files found in {machine_dir}")
        
        # Sort files by machine group and number (machine-1-1, machine-1-2, ..., machine-3-11)
        def sort_key(filename):
            # Extract machine group and number from filename
            parts = filename.replace('.txt', '').split('-')
            return (int(parts[1]), int(parts[2]))
        
        machine_files.sort(key=sort_key)
        
        print(f"Loading {len(machine_files)} SMD machine files for {self.split} split...")
        
        all_data = []
        all_labels = []
        
        for machine_file in machine_files:
            machine_name = machine_file.replace('.txt', '')
            
            try:
                if self.split == 'train':
                    # Load training data
                    data_path = os.path.join(machine_dir, machine_file)
                    data = pd.read_csv(data_path, header=None)
                    data = data.values.astype(np.float32)
                    
                    # Training data assumed to be normal
                    labels = np.zeros(len(data), dtype=np.int64)
                    
                    # Split for training only (keep validation separate)
                    val_split_idx = int(len(data) * (1 - self.val_ratio))
                    data = data[:val_split_idx]
                    labels = labels[:val_split_idx]
                    
                elif self.split == 'val':
                    # Load training data for validation split
                    data_path = os.path.join(machine_dir, machine_file)
                    data = pd.read_csv(data_path, header=None)
                    data = data.values.astype(np.float32)
                    
                    # Training data assumed to be normal
                    labels = np.zeros(len(data), dtype=np.int64)
                    
                    # Split for validation only
                    val_split_idx = int(len(data) * (1 - self.val_ratio))
                    data = data[val_split_idx:]
                    labels = labels[val_split_idx:]
                    
                else:  # test
                    # Load test data and labels
                    data_path = os.path.join(machine_dir, machine_file)
                    label_path = os.path.join(self.data_dir, 'test_label', machine_file)
                    
                    if not os.path.exists(label_path):
                        print(f"Warning: Label file not found for {machine_file}, skipping...")
                        continue
                    
                    # Load CSV data
                    data = pd.read_csv(data_path, header=None)
                    data = data.values.astype(np.float32)
                    
                    # Load labels (one per line)
                    with open(label_path, 'r') as f:
                        labels = np.array([int(line.strip()) for line in f.readlines()], dtype=np.int64)
                    
                    # Verify data and labels have same length
                    if len(data) != len(labels):
                        min_len = min(len(data), len(labels))
                        print(f"Warning: {machine_file} - Data length ({len(data)}) != Label length ({len(labels)}). Truncating to {min_len}")
                        data = data[:min_len]
                        labels = labels[:min_len]
                
                all_data.append(data)
                all_labels.append(labels)
                
                print(f"  Loaded {machine_name}: {len(data)} samples")
                
            except Exception as e:
                print(f"Warning: Could not load {machine_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid machine files could be loaded")
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total aggregated data points: {len(combined_data)}")
        print(f"Total features: {combined_data.shape[1]}")
        print(f"Total anomalies: {np.sum(combined_labels)}")
        print(f"Anomaly ratio: {np.sum(combined_labels)/len(combined_labels):.4f}")
        
        return combined_data, combined_labels
    
    def _create_synthetic_labels(self, data: np.ndarray) -> np.ndarray:
        """Create synthetic labels for data without labels"""
        # Simple synthetic labeling: mark outliers based on statistical thresholds
        # This is a fallback when no labels are available
        labels = np.zeros(len(data), dtype=np.int64)
        
        # Mark top 5% of points with highest variance as anomalies
        if len(data.shape) > 1:
            # Multivariate case
            data_std = np.std(data, axis=1)
            threshold = np.percentile(data_std, 95)
            labels[data_std > threshold] = 1
        else:
            # Univariate case
            data_abs = np.abs(data - np.mean(data))
            threshold = np.percentile(data_abs, 95)
            labels[data_abs > threshold] = 1
        
        return labels


def load_smd_data(data_dir: str, filename: str, val_ratio: float = 0.2) -> Tuple[SMDDataset, SMDDataset, SMDDataset]:
    """
    Load SMD train, validation, and test datasets
    
    Args:
        data_dir: Directory containing SMD data
        filename: Data filename (e.g., 'machine-1-1', 'machine-2-5')
        val_ratio: Ratio of training data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = SMDDataset(data_dir, filename, split='train', val_ratio=val_ratio)
    val_dataset = SMDDataset(
        data_dir, 
        filename,
        split='val',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    test_dataset = SMDDataset(
        data_dir, 
        filename,
        split='test',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    return train_dataset, val_dataset, test_dataset 