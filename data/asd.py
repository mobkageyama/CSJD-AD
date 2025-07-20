import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import os
import pickle
from .default_ts import DefaultTimeSeriesDataset


class ASDDataset(DefaultTimeSeriesDataset):
    """
    ASD (Anomaly Sequence Detection) Dataset from DualTF
    Contains machine and omi data files
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
        Initialize ASD dataset
        
        Args:
            data_dir: Directory containing ASD data files
            filename: Data filename (e.g., 'machine-1-1', 'omi-1')
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
        data, labels = self._load_asd_data()
        
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
    
    def _load_asd_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load ASD data from pickle files"""
        if self.filename is None:
            # Load all omi files (omi-1 to omi-12) and aggregate
            return self._load_all_omi_files_aggregated()
        else:
            # Load specific file
            return self._load_single_omi_file()
    
    def _load_single_omi_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single omi file"""
        if self.split == 'train':
            # Load training data and create train split
            if os.path.exists(os.path.join(self.data_dir, 'processed')):
                # Use processed files
                data_path = os.path.join(self.data_dir, 'processed', f'{self.filename}_train.pkl')
            else:
                # Use train directory
                data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.pkl')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Training data assumed to be normal
            data = np.array(data, dtype=np.float32)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Split for training only (keep validation separate)
            val_split_idx = int(len(data) * (1 - self.val_ratio))
            data = data[:val_split_idx]
            labels = labels[:val_split_idx]
            
        elif self.split == 'val':
            # Load training data and create validation split
            if os.path.exists(os.path.join(self.data_dir, 'processed')):
                # Use processed files
                data_path = os.path.join(self.data_dir, 'processed', f'{self.filename}_train.pkl')
            else:
                # Use train directory
                data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.pkl')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Training data assumed to be normal
            data = np.array(data, dtype=np.float32)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Split for validation only
            val_split_idx = int(len(data) * (1 - self.val_ratio))
            data = data[val_split_idx:]
            labels = labels[val_split_idx:]
            
        else:  # test
            # Load test data and labels from separate files
            if os.path.exists(os.path.join(self.data_dir, 'processed')):
                # Use processed files
                data_path = os.path.join(self.data_dir, 'processed', f'{self.filename}_test.pkl')
                label_path = os.path.join(self.data_dir, 'processed', f'{self.filename}_test_label.pkl')
            else:
                # Use test and test_label directories
                data_path = os.path.join(self.data_dir, 'test', f'{self.filename}.pkl')
                label_path = os.path.join(self.data_dir, 'test_label', f'{self.filename}.pkl')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Test data not found: {data_path}")
            
            # Load test data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            data = np.array(data, dtype=np.float32)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Load test labels
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    labels = pickle.load(f)
                labels = np.array(labels, dtype=np.int64)
            else:
                # Create synthetic labels if no labels available
                labels = self._create_synthetic_labels(data)
        
        return data, labels
    
    def _load_all_omi_files_aggregated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all omi files (omi-1 to omi-12) and aggregate them"""
        processed_dir = os.path.join(self.data_dir, 'processed')
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"ASD processed directory not found: {processed_dir}")
        
        # Get all omi files
        if self.split == 'train':
            omi_files = [f for f in os.listdir(processed_dir) if f.startswith('omi-') and f.endswith('_train.pkl')]
        elif self.split == 'val':
            omi_files = [f for f in os.listdir(processed_dir) if f.startswith('omi-') and f.endswith('_train.pkl')]
        else:  # test
            omi_files = [f for f in os.listdir(processed_dir) if f.startswith('omi-') and f.endswith('_test.pkl')]
        
        if not omi_files:
            raise ValueError(f"No omi files found for {self.split} split")
        
        # Sort files by number (omi-1, omi-2, ..., omi-12)
        omi_files.sort(key=lambda x: int(x.split('-')[1].split('_')[0]))
        
        print(f"Loading {len(omi_files)} ASD omi files for {self.split} split...")
        
        all_data = []
        all_labels = []
        
        for omi_file in omi_files:
            file_path = os.path.join(processed_dir, omi_file)
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                data = np.array(data, dtype=np.float32)
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                if self.split == 'test':
                    # Load corresponding test labels
                    label_file = omi_file.replace('_test.pkl', '_test_label.pkl')
                    label_path = os.path.join(processed_dir, label_file)
                    
                    if os.path.exists(label_path):
                        with open(label_path, 'rb') as f:
                            labels = pickle.load(f)
                        labels = np.array(labels, dtype=np.int64)
                    else:
                        # Create synthetic labels if no labels available
                        labels = self._create_synthetic_labels(data)
                else:
                    # Training/validation data assumed to be normal
                    labels = np.zeros(len(data), dtype=np.int64)
                
                all_data.append(data)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Warning: Could not load {omi_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid omi files could be loaded")
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total aggregated data points: {len(combined_data)}")
        print(f"Total anomalies: {np.sum(combined_labels)}")
        print(f"Anomaly ratio: {np.sum(combined_labels)/len(combined_labels):.4f}")
        
        # Apply train/val split for training and validation
        if self.split == 'train':
            # Split for training only (keep validation separate)
            val_split_idx = int(len(combined_data) * (1 - self.val_ratio))
            data = combined_data[:val_split_idx]
            labels = combined_labels[:val_split_idx]
        elif self.split == 'val':
            # Split for validation only
            val_split_idx = int(len(combined_data) * (1 - self.val_ratio))
            data = combined_data[val_split_idx:]
            labels = combined_labels[val_split_idx:]
        else:  # test
            data = combined_data
            labels = combined_labels
        
        return data, labels
    
    def _create_synthetic_labels(self, data: np.ndarray) -> np.ndarray:
        """Create synthetic anomaly labels based on statistical outliers"""
        from scipy import stats
        
        # Use z-score to identify outliers
        if len(data.shape) > 1 and data.shape[1] > 1:
            # For multivariate data, use mean across features
            data_1d = np.mean(data, axis=1)
        else:
            data_1d = data.flatten()
        
        z_scores = np.abs(stats.zscore(data_1d))
        # Mark points with z-score > 3 as anomalies
        labels = (z_scores > 3).astype(np.int64)
        
        return labels
    


def load_asd_data(data_dir: str, filename: str, val_ratio: float = 0.2) -> Tuple[ASDDataset, ASDDataset, ASDDataset]:
    """
    Load ASD train, validation, and test datasets
    
    Args:
        data_dir: Directory containing ASD data
        filename: Data filename (e.g., 'machine-1-1', 'omi-1')
        val_ratio: Ratio of training data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ASDDataset(data_dir, filename, split='train', val_ratio=val_ratio)
    val_dataset = ASDDataset(
        data_dir, 
        filename,
        split='val',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    test_dataset = ASDDataset(
        data_dir, 
        filename,
        split='test',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    return train_dataset, val_dataset, test_dataset


def get_available_asd_files(data_dir: str) -> list:
    """Get list of available ASD files"""
    files = []
    
    # Check processed directory
    processed_dir = os.path.join(data_dir, 'processed')
    if os.path.exists(processed_dir):
        train_files = [f.replace('_train.pkl', '') for f in os.listdir(processed_dir) 
                      if f.endswith('_train.pkl')]
        files.extend(train_files)
    
    # Check train directory
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        train_files = [f.replace('.pkl', '') for f in os.listdir(train_dir) 
                      if f.endswith('.pkl')]
        for f in train_files:
            if f not in files:
                files.append(f)
    
    return sorted(files)