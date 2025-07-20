import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union, List, Dict, Any
import pickle
import os


class DefaultTimeSeriesDataset(Dataset):
    """
    Default time series dataset for CSJD-AD
    Compatible with CARLA's dataset interface
    Enhanced with multiple normalization methods to ensure train-test domain consistency
    """
    
    def __init__(
        self,
        data_path: str = None,
        label_path: str = None,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        window_size: int = 100,
        stride: int = 1,
        transform=None,
        split: str = 'train',
        mean: Optional[Union[float, np.ndarray]] = None,
        std: Optional[Union[float, np.ndarray]] = None,
        normalization_method: str = 'z_score',
        normalization_stats: Optional[Dict[str, Any]] = None,
        global_data_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to data file
            label_path: Path to label file
            data: Pre-loaded data array
            labels: Pre-loaded labels array
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'test', 'train+unlabeled')
            mean: Normalization mean (for backward compatibility)
            std: Normalization std (for backward compatibility)
            normalization_method: Method for normalization ('z_score', 'min_max', 'robust', 'global_z_score', 'global_min_max')
            normalization_stats: Pre-computed normalization statistics
            global_data_stats: Global statistics from combined train+test data
        """
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.split = split
        self.normalization_method = normalization_method
        self.global_data_stats = global_data_stats
        
        # Load data
        if data is not None:
            self.data = data
            self.labels = labels if labels is not None else np.zeros(len(data))
        else:
            self.data, self.labels = self._load_data(data_path, label_path)
        
        # Apply normalization
        if normalization_stats is not None:
            # Use pre-computed statistics
            self.normalization_stats = normalization_stats
            self.data = self._apply_normalization(self.data)
        elif mean is not None and std is not None:
            # Backward compatibility with old mean/std approach
            self.mean = mean
            self.std = std
            self.normalization_stats = {'mean': mean, 'std': std}
            self.data = (self.data - self.mean) / self.std
        else:
            # Compute new normalization statistics
            self.normalization_stats = self._compute_normalization_stats(self.data)
            self.data = self._apply_normalization(self.data)
        
        # Store backward compatibility attributes
        if 'mean' in self.normalization_stats:
            self.mean = self.normalization_stats['mean']
        if 'std' in self.normalization_stats:
            self.std = self.normalization_stats['std']
        
        # Create windows
        self.windows, self.window_labels = self._create_windows()
        
    def _load_data(self, data_path: str, label_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from files"""
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path).values
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Load labels if provided
        labels = np.zeros(len(data))
        if label_path is not None:
            if label_path.endswith('.csv'):
                labels = pd.read_csv(label_path).values.flatten()
            elif label_path.endswith('.pkl'):
                with open(label_path, 'rb') as f:
                    labels = pickle.load(f)
            elif label_path.endswith('.npy'):
                labels = np.load(label_path)
        
        return data.astype(np.float32), labels.astype(np.int64)
    
    def _compute_normalization_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute normalization statistics based on the selected method"""
        stats = {}
        
        if self.normalization_method == 'none':
            # No normalization - return identity statistics
            stats['mean'] = np.zeros((1, data.shape[1]))
            stats['std'] = np.ones((1, data.shape[1]))
            
        elif self.normalization_method == 'z_score':
            # Standard Z-score normalization (train data only)
            stats['mean'] = np.mean(data, axis=0, keepdims=True)
            stats['std'] = np.std(data, axis=0, keepdims=True) + 1e-8
            
        elif self.normalization_method == 'min_max':
            # Min-Max normalization (train data only)
            stats['min'] = np.min(data, axis=0, keepdims=True)
            stats['max'] = np.max(data, axis=0, keepdims=True)
            # Avoid division by zero
            range_val = stats['max'] - stats['min']
            range_val[range_val == 0] = 1.0
            stats['range'] = range_val
            
        elif self.normalization_method == 'robust':
            # Robust normalization using median and IQR
            stats['median'] = np.median(data, axis=0, keepdims=True)
            stats['q25'] = np.percentile(data, 25, axis=0, keepdims=True)
            stats['q75'] = np.percentile(data, 75, axis=0, keepdims=True)
            stats['iqr'] = stats['q75'] - stats['q25'] + 1e-8
            
        elif self.normalization_method == 'global_z_score':
            # Global Z-score using combined train+test statistics
            if self.global_data_stats is None:
                raise ValueError("global_data_stats required for global_z_score normalization")
            stats['mean'] = self.global_data_stats['global_mean']
            stats['std'] = self.global_data_stats['global_std']
            
        elif self.normalization_method == 'global_min_max':
            # Global Min-Max using combined train+test statistics
            if self.global_data_stats is None:
                raise ValueError("global_data_stats required for global_min_max normalization")
            stats['min'] = self.global_data_stats['global_min']
            stats['max'] = self.global_data_stats['global_max']
            stats['range'] = self.global_data_stats['global_range']
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return stats
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data using computed statistics"""
        if self.normalization_method == 'none':
            return data  # No normalization
        
        elif self.normalization_method in ['z_score', 'global_z_score']:
            return (data - self.normalization_stats['mean']) / self.normalization_stats['std']
        
        elif self.normalization_method in ['min_max', 'global_min_max']:
            return (data - self.normalization_stats['min']) / self.normalization_stats['range']
        
        elif self.normalization_method == 'robust':
            return (data - self.normalization_stats['median']) / self.normalization_stats['iqr']
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    @staticmethod
    def compute_global_stats(train_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """
        Compute global statistics from combined train and test data
        
        Args:
            train_data: Training data array
            test_data: Test data array
            
        Returns:
            Dictionary containing global statistics
        """
        # Combine train and test data
        combined_data = np.concatenate([train_data, test_data], axis=0)
        
        global_stats = {
            # Z-score statistics
            'global_mean': np.mean(combined_data, axis=0, keepdims=True),
            'global_std': np.std(combined_data, axis=0, keepdims=True) + 1e-8,
            
            # Min-Max statistics
            'global_min': np.min(combined_data, axis=0, keepdims=True),
            'global_max': np.max(combined_data, axis=0, keepdims=True),
            
            # Robust statistics
            'global_median': np.median(combined_data, axis=0, keepdims=True),
            'global_q25': np.percentile(combined_data, 25, axis=0, keepdims=True),
            'global_q75': np.percentile(combined_data, 75, axis=0, keepdims=True),
        }
        
        # Compute derived statistics
        global_stats['global_range'] = global_stats['global_max'] - global_stats['global_min']
        # Avoid division by zero
        global_stats['global_range'][global_stats['global_range'] == 0] = 1.0
        
        global_stats['global_iqr'] = global_stats['global_q75'] - global_stats['global_q25'] + 1e-8
        
        return global_stats
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from time series data"""
        n_timesteps = len(self.data)
        n_features = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
        
        # Handle empty data case (e.g., for validation sets that don't exist)
        if n_timesteps == 0:
            print(f"  Empty dataset - returning empty windows")
            return np.array([]).reshape(0, self.window_size, n_features), np.array([])
        
        # Calculate number of windows
        n_windows = (n_timesteps - self.window_size) // self.stride + 1
        
        # Handle case where data is too short for window size
        if n_windows <= 0:
            raise ValueError(f"Data sequence too short for window size. "
                           f"Data length: {n_timesteps}, Window size: {self.window_size}, "
                           f"Calculated windows: {n_windows}. "
                           f"Need at least {self.window_size} timesteps.")
        
        # Create windows
        windows = np.zeros((n_windows, self.window_size, n_features))
        window_labels = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            windows[i] = self.data[start_idx:end_idx]
            window_labels[i] = np.max(self.labels[start_idx:end_idx])
        
        return windows, window_labels
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single window"""
        window = self.windows[idx]
        label = self.window_labels[idx]
        
        # Convert to tensor
        window = torch.from_numpy(window).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        if self.transform is not None:
            window = self.transform(window)
        
        return window, label
    
    def get_pointwise_labels(self, idx: int) -> torch.Tensor:
        """Get point-wise labels for a window"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Get the original point-wise labels for this window
        point_labels = self.labels[start_idx:end_idx]
        
        return torch.from_numpy(point_labels).long()
    
    def concat_ds(self, other_dataset):
        """Concatenate with another dataset (CARLA compatibility)"""
        self.windows = np.concatenate([self.windows, other_dataset.windows], axis=0)
        self.window_labels = np.concatenate([self.window_labels, other_dataset.window_labels], axis=0)


class AugmentedTimeSeriesDataset(Dataset):
    """
    Augmented dataset for contrastive learning
    Returns pairs of augmented views
    """
    
    def __init__(self, base_dataset: DefaultTimeSeriesDataset, num_augmentations: int = 2):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """Get augmented views of a single window"""
        window, label = self.base_dataset[idx]
        
        # Create multiple augmented views
        augmented_views = []
        for _ in range(self.num_augmentations):
            if self.base_dataset.transform is not None:
                aug_window = self.base_dataset.transform(window.clone())
            else:
                aug_window = window.clone()
            augmented_views.append(aug_window)
        
        return augmented_views, label


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    anomaly_ratio: float = 0.1,
    noise_level: float = 0.1,
    window_size: int = 100
) -> Tuple[DefaultTimeSeriesDataset, DefaultTimeSeriesDataset]:
    """
    Create synthetic time series datasets for training and testing
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Generate normal data with trends and seasonality
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Add trend
        trend = 0.1 * t * (i + 1)
        
        # Add seasonality
        seasonal = np.sin(t * (i + 1)) + 0.5 * np.cos(t * (i + 2))
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        
        data[:, i] = trend + seasonal + noise
    
    # Create anomalies
    labels = np.zeros(n_samples)
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Add spike anomalies
        data[idx] += np.random.normal(0, 2, n_features)
        labels[idx] = 1
    
    # Split data
    split_idx = int(n_samples * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    
    # Create datasets
    train_dataset = DefaultTimeSeriesDataset(
        data=train_data,
        labels=train_labels,
        window_size=window_size,
        split='train'
    )
    
    test_dataset = DefaultTimeSeriesDataset(
        data=test_data,
        labels=test_labels,
        window_size=window_size,
        split='test',
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    return train_dataset, test_dataset


def load_interfusion_data(data_dir: str, fname: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load InterFusion format data
    
    Args:
        data_dir: Directory containing InterFusion data files
        fname: Specific filename to load (without extension)
        
    Returns:
        Tuple of (train_data, test_data, train_labels, test_labels)
    """
    if fname is None:
        # Load all files
        train_files = [f for f in os.listdir(data_dir) if f.endswith('_train.pkl')]
        test_files = [f for f in os.listdir(data_dir) if f.endswith('_test.pkl')]
        label_files = [f for f in os.listdir(data_dir) if f.endswith('_test_label.pkl')]
        
        train_data_list = []
        test_data_list = []
        test_labels_list = []
        
        for train_file in train_files:
            with open(os.path.join(data_dir, train_file), 'rb') as f:
                train_data_list.append(pickle.load(f))
        
        for test_file in test_files:
            with open(os.path.join(data_dir, test_file), 'rb') as f:
                test_data_list.append(pickle.load(f))
                
        for label_file in label_files:
            with open(os.path.join(data_dir, label_file), 'rb') as f:
                test_labels_list.append(pickle.load(f))
        
        # Concatenate all data
        train_data = np.concatenate(train_data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        train_labels = np.zeros(len(train_data))  # Assume training data is normal
        
    else:
        # Load specific file
        train_path = os.path.join(data_dir, f"{fname}_train.pkl")
        test_path = os.path.join(data_dir, f"{fname}_test.pkl")
        label_path = os.path.join(data_dir, f"{fname}_test_label.pkl")
        
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        with open(label_path, 'rb') as f:
            test_labels = pickle.load(f)
        
        train_labels = np.zeros(len(train_data))
    
    return train_data, test_data, train_labels, test_labels