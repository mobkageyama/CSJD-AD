import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from .default_ts import DefaultTimeSeriesDataset


class YahooDataset(DefaultTimeSeriesDataset):
    """
    Yahoo Anomaly Detection Dataset
    
    Contains various time series from Yahoo with anomaly detection tasks
    """
    
    def __init__(
        self,
        data_dir: str,
        filename: str = None,
        window_size: int = 250,
        stride: int = 1,
        transform=None,
        split: str = 'train',
        mean: Optional[float] = None,
        std: Optional[float] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.0  # No validation set needed
    ):
        """
        Initialize Yahoo dataset
        
        Args:
            data_dir: Directory containing Yahoo data files
            filename: Specific Yahoo filename (e.g., 'real_1' for real_1.csv)
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', 'test')
            mean: Normalization mean
            std: Normalization std
            train_ratio: Ratio of data to use for training vs test
            val_ratio: Ratio of training data to use for validation
        """
        self.data_dir = data_dir
        self.filename = filename
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.window_size = window_size  # Store window_size for validation ratio adjustment
        
        # Load data
        data, labels = self._load_yahoo_data()
        
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
    
    def _load_yahoo_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Yahoo data from CSV files"""
        if self.filename is None:
            # Load all available files and aggregate before splitting
            return self._load_all_yahoo_files_aggregated()
        else:
            # Load specific file
            return self._load_single_yahoo_file()
    
    def _has_both_normal_and_anomaly_in_test(self, filename: str) -> bool:
        """Check if a file has both normal and anomaly samples in the test split"""
        file_path = os.path.join(self.data_dir, f"{filename}.csv")
        
        if not os.path.exists(file_path):
            return False
            
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Extract labels
            if 'timestamp' in df.columns:
                df = df.drop('timestamp', axis=1)
            
            if 'is_anomaly' in df.columns:
                labels = df['is_anomaly'].values.astype(np.int64)
            elif df.shape[1] > 1:
                labels = df.iloc[:, 1].values.astype(np.int64)
            else:
                # No labels available, assume no anomalies
                return False
            
            # Get test split
            test_split_idx = int(len(labels) * self.train_ratio)
            test_labels = labels[test_split_idx:]
            
            # Check if both normal (0) and anomaly (1) samples exist in test set
            has_normal = np.any(test_labels == 0)
            has_anomaly = np.any(test_labels == 1)
            
            return has_normal and has_anomaly
            
        except Exception as e:
            print(f"  Error checking file {filename}: {e}")
            return False

    def _has_sufficient_normal_data_in_train(self, filename: str, min_normal_ratio: float = 0.1, min_normal_samples: int = None) -> bool:
        """Check if a file has sufficient normal data in the training split"""
        file_path = os.path.join(self.data_dir, f"{filename}.csv")
        
        if not os.path.exists(file_path):
            return False
            
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Extract labels
            if 'timestamp' in df.columns:
                df = df.drop('timestamp', axis=1)
            
            if 'is_anomaly' in df.columns:
                labels = df['is_anomaly'].values.astype(np.int64)
            elif df.shape[1] > 1:
                labels = df.iloc[:, 1].values.astype(np.int64)
            else:
                # No labels available, assume all normal
                return True
            
            # Get training split
            train_split_idx = int(len(labels) * self.train_ratio)
            train_labels = labels[:train_split_idx]
            
            # Count normal samples in training data
            normal_count = np.sum(train_labels == 0)
            total_train_samples = len(train_labels)
            
            # Check minimum number of normal samples
            if min_normal_samples is None:
                min_normal_samples = self.window_size * 2  # At least 2x window size
            
            # Check both ratio and absolute count
            normal_ratio = normal_count / total_train_samples if total_train_samples > 0 else 0
            has_sufficient_ratio = normal_ratio >= min_normal_ratio
            has_sufficient_count = normal_count >= min_normal_samples
            
            return has_sufficient_ratio and has_sufficient_count
            
        except Exception as e:
            print(f"  Error checking normal data for file {filename}: {e}")
            return False

    def _get_adjusted_val_ratio(self, data_length: int, window_size: int) -> float:
        """Calculate adjusted validation ratio to ensure validation split is large enough for window_size"""
        # Minimum validation samples needed for window creation
        min_val_samples = window_size + 10  # Add small buffer
        
        # If default val_ratio gives enough samples, use it
        default_val_samples = int(data_length * self.val_ratio)
        if default_val_samples >= min_val_samples:
            return self.val_ratio
        
        # Calculate minimum val_ratio needed
        min_val_ratio = min_val_samples / data_length
        
        # Cap at 0.5 to ensure we still have enough training data (increased from 0.4)
        adjusted_ratio = min(min_val_ratio, 0.5)
        
        # If even max val_ratio doesn't give enough samples, try using more aggressive approach
        max_possible_val_samples = int(data_length * adjusted_ratio)
        if max_possible_val_samples < min_val_samples:
            # For very short files, use the entire remaining data for validation
            if data_length >= window_size:
                # Use all available data for validation if possible
                adjusted_ratio = min(window_size / data_length + 0.05, 0.8)  # Use window_size + 5% or 80%, whichever is smaller
                print(f"  File {self.filename} very short. Using aggressive val_ratio={adjusted_ratio:.3f} "
                      f"(val samples: {int(data_length * adjusted_ratio)})")
                return adjusted_ratio
            else:
                print(f"  File {self.filename} too short for validation with window_size={window_size}. "
                      f"Total length: {data_length}, needed: {min_val_samples}. Skipping validation (val_ratio=0)")
                return 0.0
        
        if adjusted_ratio != self.val_ratio:
            print(f"  Adjusting val_ratio for {self.filename}: {self.val_ratio:.3f} → {adjusted_ratio:.3f} "
                  f"(val samples: {int(data_length * adjusted_ratio)})")
        
        return adjusted_ratio
    
    def _load_single_yahoo_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single Yahoo CSV file"""
        file_path = os.path.join(self.data_dir, f"{self.filename}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Yahoo data file not found: {file_path}")
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Expected columns: timestamp, value, is_anomaly
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        # Extract values and labels
        if 'value' in df.columns and 'is_anomaly' in df.columns:
            data = df['value'].values.astype(np.float32)
            labels = df['is_anomaly'].values.astype(np.int64)
        else:
            # Fallback: assume first column is value, second is label
            data = df.iloc[:, 0].values.astype(np.float32)
            if df.shape[1] > 1:
                labels = df.iloc[:, 1].values.astype(np.int64)
            else:
                labels = np.zeros(len(data), dtype=np.int64)
        
        # Reshape data to have feature dimension
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Split data based on split type - following CARLA's approach
        test_split_idx = int(len(data) * self.train_ratio)
        
        if self.split == 'train':
            # Get training portion (everything before test split)
            data = data[:test_split_idx]
            labels = labels[:test_split_idx]
            
            # For training, keep only normal data (following CARLA's unsupervised approach)
            normal_indices = labels == 0
            data = data[normal_indices]
            labels = labels[normal_indices]
            
            # Check if we have enough normal data for window creation
            if len(data) < self.window_size:
                print(f"  Warning: After filtering for normal data, only {len(data)} samples remain, "
                      f"but need {self.window_size} for window creation.")
                if len(data) > 0:
                    # Repeat the data to reach minimum window size
                    repeat_times = (self.window_size // len(data)) + 1
                    data = np.tile(data, (repeat_times, 1))[:self.window_size]
                    labels = np.tile(labels, repeat_times)[:self.window_size]
                    print(f"  Repeated normal data to create {len(data)} samples for training")
                else:
                    # No normal data available, create synthetic normal data
                    data = np.zeros((self.window_size, data.shape[1] if len(data.shape) > 1 else 1), dtype=np.float32)
                    labels = np.zeros(self.window_size, dtype=np.int64)
                    print(f"  No normal data available, created {len(data)} synthetic normal samples for training")
        elif self.split == 'val':
            # No validation set - return empty data
            data = np.array([]).reshape(0, 1)
            labels = np.array([])
            print(f"  No validation set used - returning empty data")
        else:  # test - keep all data including anomalies for evaluation
            test_data = data[test_split_idx:]
            test_labels = labels[test_split_idx:]
            
            # Check if test data is sufficient for window size
            if len(test_data) < self.window_size:
                print(f"  Warning: Test split too short ({len(test_data)} < {self.window_size}). "
                      f"Attempting to extend test data.")
                
                # Strategy 1: Try to take more data from before the test split
                # Calculate how much more data we need
                needed_samples = self.window_size - len(test_data)
                available_before_test = test_split_idx
                
                if available_before_test >= needed_samples:
                    # Take the needed samples from the end of training data
                    extended_start_idx = test_split_idx - needed_samples
                    extended_data = data[extended_start_idx:test_split_idx]  # Get data before test split
                    extended_labels = labels[extended_start_idx:test_split_idx]  # Get labels before test split
                    
                    # Combine the extended portion with test data
                    data = np.concatenate([extended_data, test_data])
                    labels = np.concatenate([extended_labels, test_labels])
                    print(f"  Extended test data by taking {needed_samples} samples from before test split. "
                          f"New test length: {len(data)}")
                else:
                    # Strategy 2: Use all available data (entire file) for test
                    print(f"  Not enough data before test split. Using entire file for test.")
                    # data and labels already contain the full file data
                    print(f"  Using entire file for test: {len(data)} samples")
                    
                    # Strategy 3: If even the entire file is too short, repeat the data
                    if len(data) < self.window_size:
                        print(f"  Entire file too short ({len(data)} < {self.window_size}). "
                              f"Repeating data to reach minimum window size.")
                        repeat_times = (self.window_size // len(data)) + 1
                        data = np.tile(data, (repeat_times, 1))[:self.window_size]
                        labels = np.tile(labels, repeat_times)[:self.window_size]
                        print(f"  Repeated data to create {len(data)} samples for test")
            else:
                # Test data is sufficient, use as is
                data = test_data
                labels = test_labels
        
        return data, labels
    
    def _load_all_yahoo_files_aggregated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all Yahoo CSV files, aggregate them, then split into train/test"""
        # Use the data directory directly (it already points to the Yahoo directory)
        yahoo_dir = self.data_dir
        if not os.path.exists(yahoo_dir):
            raise FileNotFoundError(f"Yahoo dataset directory not found: {yahoo_dir}")
        
        # Get all real_*.csv files
        csv_files = [f for f in os.listdir(yahoo_dir) 
                    if f.startswith('real_') and f.endswith('.csv') and not f.startswith('._')]
        
        if not csv_files:
            raise FileNotFoundError(f"No Yahoo CSV files found in {yahoo_dir}")
        
        print(f"Loading {len(csv_files)} Yahoo dataset files...")
        
        all_data = []
        all_labels = []
        
        # Load all files and aggregate
        for csv_file in sorted(csv_files, key=lambda x: int(x.split('_')[1].split('.')[0])):
            file_path = os.path.join(yahoo_dir, csv_file)
            
            try:
                df = pd.read_csv(file_path)
                
                # Remove timestamp column if present
                if 'timestamp' in df.columns:
                    df = df.drop('timestamp', axis=1)
                
                # Extract values and labels
                if 'value' in df.columns and 'is_anomaly' in df.columns:
                    data = df['value'].values.astype(np.float32)
                    labels = df['is_anomaly'].values.astype(np.int64)
                else:
                    # Fallback: assume first column is value, second is label
                    data = df.iloc[:, 0].values.astype(np.float32)
                    if df.shape[1] > 1:
                        labels = df.iloc[:, 1].values.astype(np.int64)
                    else:
                        labels = np.zeros(len(data), dtype=np.int64)
                
                all_data.append(data)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        # Concatenate all data from all files
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total aggregated data points: {len(combined_data)}")
        print(f"Total anomalies: {np.sum(combined_labels)}")
        print(f"Anomaly ratio: {np.sum(combined_labels)/len(combined_labels):.4f}")
        
        # Reshape data to have feature dimension
        if len(combined_data.shape) == 1:
            combined_data = combined_data.reshape(-1, 1)
        
        # Now split the aggregated data based on split type - following CARLA's approach
        test_split_idx = int(len(combined_data) * self.train_ratio)
        
        if self.split == 'train':
            # Get training portion first
            train_data = combined_data[:test_split_idx]
            train_labels = combined_labels[:test_split_idx]
            
            # Adjust val_ratio if file is too short for window_size
            adjusted_val_ratio = self._get_adjusted_val_ratio(len(train_data), self.window_size)
            
            # Then split training data for train/validation
            val_split_idx = int(len(train_data) * (1 - adjusted_val_ratio))
            data = train_data[:val_split_idx]
            labels = train_labels[:val_split_idx]
            
            # For training, keep only normal data (following CARLA's unsupervised approach)
            normal_indices = labels == 0
            data = data[normal_indices]
            labels = labels[normal_indices]
            print(f"Training data points (normal only): {len(data)}")
        elif self.split == 'val':
            # Get training portion first
            train_data = combined_data[:test_split_idx]
            train_labels = combined_labels[:test_split_idx]
            
            # Adjust val_ratio if file is too short for window_size
            adjusted_val_ratio = self._get_adjusted_val_ratio(len(train_data), self.window_size)
            
            # If val_ratio is 0 (file too short), create minimal validation data from end of training data
            if adjusted_val_ratio == 0.0:
                # Use last window_size samples for validation
                if len(train_data) >= self.window_size:
                    data = train_data[-self.window_size:]
                    labels = train_labels[-self.window_size:]
                    print(f"  Using last {self.window_size} samples for validation")
                else:
                    # File is extremely short, use all available training data for validation
                    data = train_data
                    labels = train_labels
                    print(f"  File extremely short, using all {len(train_data)} training samples for validation")
            else:
                # Then split training data for train/validation
                val_split_idx = int(len(train_data) * (1 - adjusted_val_ratio))
                data = train_data[val_split_idx:]
                labels = train_labels[val_split_idx:]
                
                # Double-check that we have enough validation data
                if len(data) < self.window_size:
                    print(f"  Warning: Validation split still too short ({len(data)} < {self.window_size}). "
                          f"Using more training data for validation.")
                    # Take more data from training to ensure we have enough for validation
                    required_val_samples = self.window_size + 10
                    if len(train_data) >= required_val_samples:
                        data = train_data[-required_val_samples:]
                        labels = train_labels[-required_val_samples:]
                        print(f"  Adjusted validation to use last {required_val_samples} samples")
                    else:
                        # Use all training data for validation as last resort
                        data = train_data
                        labels = train_labels
                        print(f"  Using all {len(train_data)} training samples for validation")
            
            # For validation, keep only normal data (following CARLA's unsupervised approach)
            normal_indices = labels == 0
            data = data[normal_indices]
            labels = labels[normal_indices]
            
            # Final check: if normal data is still too short for window size, pad or repeat
            if len(data) < self.window_size:
                print(f"  Warning: After filtering for normal data, only {len(data)} samples remain, "
                      f"but need {self.window_size} for window creation.")
                if len(data) > 0:
                    # Repeat the data to reach minimum window size
                    repeat_times = (self.window_size // len(data)) + 1
                    data = np.tile(data, (repeat_times, 1))[:self.window_size]
                    labels = np.tile(labels, repeat_times)[:self.window_size]
                    print(f"  Repeated normal data to create {len(data)} samples for validation")
                else:
                    # No normal data available, create synthetic normal data
                    data = np.zeros((self.window_size, data.shape[1] if len(data.shape) > 1 else 1), dtype=np.float32)
                    labels = np.zeros(self.window_size, dtype=np.int64)
                    print(f"  No normal data available, created {len(data)} synthetic normal samples for validation")
            
            print(f"Validation data points (normal only): {len(data)}")
        else:  # test - keep all data including anomalies for evaluation
            data = combined_data[test_split_idx:]
            labels = combined_labels[test_split_idx:]
            print(f"Test data points: {len(data)}")
            print(f"Test anomalies: {np.sum(labels)}")
        
        return data, labels


def load_yahoo_data(data_dir: str, filename: str = None, train_ratio: float = 0.8, val_ratio: float = 0.0) -> Tuple[YahooDataset, YahooDataset]:
    """
    Load Yahoo train and test datasets (no validation set)
    
    Args:
        data_dir: Directory containing Yahoo data
        filename: Specific Yahoo filename (optional, None for all files aggregated)
        train_ratio: Ratio of data to use for training vs test
        val_ratio: Not used (kept for compatibility)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Create train dataset first to compute normalization stats
    train_dataset = YahooDataset(data_dir, filename, split='train', train_ratio=train_ratio, val_ratio=val_ratio)
    
    # Create test dataset using same normalization stats
    test_dataset = YahooDataset(
        data_dir, 
        filename,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    return train_dataset, test_dataset


def get_available_yahoo_files(data_dir: str, filter_test_sets: bool = True, filter_train_normal: bool = True, 
                            min_normal_ratio: float = 0.1, min_normal_samples: int = None) -> list:
    """
    Get list of available Yahoo files with optional filtering
    
    Args:
        data_dir: Directory containing Yahoo data files
        filter_test_sets: Filter out files without both normal and anomaly samples in test set
        filter_train_normal: Filter out files without sufficient normal data in training set
        min_normal_ratio: Minimum ratio of normal samples in training data (default: 0.1 = 10%)
        min_normal_samples: Minimum absolute number of normal samples (default: 2x window_size)
    """
    files = []
    
    for f in os.listdir(data_dir):
        if f.endswith('.csv') and not f.startswith('._'):
            filename = f.replace('.csv', '')
            
            # Create a temporary dataset instance for checking
            temp_dataset = YahooDataset(data_dir, filename, split='train', train_ratio=0.8, val_ratio=0.0)
            
            skip_file = False
            skip_reasons = []
            
            if filter_test_sets:
                if not temp_dataset._has_both_normal_and_anomaly_in_test(filename):
                    skip_reasons.append("test set doesn't have both normal and anomaly samples")
                    skip_file = True
            
            if filter_train_normal and not skip_file:
                if not temp_dataset._has_sufficient_normal_data_in_train(filename, min_normal_ratio, min_normal_samples):
                    skip_reasons.append("insufficient normal data in training set")
                    skip_file = True
            
            if skip_file:
                print(f"  Skipping file {filename}: {', '.join(skip_reasons)}")
            else:
                files.append(filename)
    
    return sorted(files)


# Legacy Yahoo class for backward compatibility
class Yahoo(Dataset):
    """
    Legacy Yahoo Dataset class for backward compatibility
    """
    
    def __init__(self, fname, root, train=True, transform=None, sanomaly=None, 
                 mean_data=None, std_data=None, data=None, label=None):
        super(Yahoo, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train
        self.classes = ['Normal', 'Anomaly']
        
        self.mean, self.std = mean_data, std_data
        
        if data is not None and label is not None:
            # Use provided data
            if self.train:
                self.mean = data.mean()
                self.std = data.std()
            else:
                if self.std == 0.0:
                    self.std = 1.0
                data = (data - self.mean) / self.std
            
            self.data = np.asarray(data)
            self.targets = np.asarray(label)
        else:
            # Load data from file
            file_path = os.path.join(root, f"{fname}.csv")
            df = pd.read_csv(file_path)
            
            # Extract data and labels
            if 'value' in df.columns and 'is_anomaly' in df.columns:
                data = df['value'].values.astype(np.float32)
                labels = df['is_anomaly'].values.astype(np.int64)
            else:
                data = df.iloc[:, 0].values.astype(np.float32)
                if df.shape[1] > 1:
                    labels = df.iloc[:, 1].values.astype(np.int64)
                else:
                    labels = np.zeros(len(data), dtype=np.int64)
            
            # Normalization
            if self.train:
                self.mean = data.mean()
                self.std = data.std()
            else:
                if self.std == 0.0:
                    self.std = 1.0
                data = (data - self.mean) / self.std
            
            self.data = np.asarray(data)
            self.targets = np.asarray(labels)
        
        # Convert to windows
        wsz, stride = 250, 1
        self.data, self.targets = self.convert_to_windows(wsz, stride)
    
    def convert_to_windows(self, w_size, stride):
        """Convert time series to sliding windows"""
        windows = []
        wlabels = []
        
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
        
        sz = int((self.data.shape[0] - w_size) / stride) + 1
        
        for i in range(sz):
            st = i * stride
            w = self.data[st:st + w_size]
            
            # Label window as anomaly if any point is anomaly
            if sum(self.targets[st:st + w_size]) > 0:
                lbl = 1
            else:
                lbl = 0
            
            windows.append(w)
            wlabels.append(lbl)
        
        return np.stack(windows), np.stack(wlabels)
    
    def __getitem__(self, index):
        """Get item by index"""
        ts_org = torch.from_numpy(self.data[index]).float()
        
        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''
        
        ts_size = len(ts_org)
        
        out = {
            'ts_org': ts_org,
            'target': target,
            'meta': {
                'ts_size': ts_size,
                'index': index,
                'class_name': class_name
            }
        }
        
        return out
    
    def get_ts(self, index):
        """Get time series by index"""
        return self.data[index]
    
    def get_info(self):
        """Get normalization info"""
        return self.mean, self.std
    
    def concat_ds(self, new_ds):
        """Concatenate with another dataset"""
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")