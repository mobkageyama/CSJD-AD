import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import os
import pickle
from .default_ts import DefaultTimeSeriesDataset


class ECGDataset(DefaultTimeSeriesDataset):
    """
    ECG Dataset from DualTF
    Contains various ECG recordings with anomaly detection tasks
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
        use_processed: bool = True,
        val_ratio: float = 0.2
    ):
        """
        Initialize ECG dataset
        
        Args:
            data_dir: Directory containing ECG data files
            filename: ECG filename (without extension)
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean
            std: Normalization std
            use_processed: Whether to use processed pickle files or raw txt files
            val_ratio: Ratio of training data to use for validation
        """
        self.data_dir = data_dir
        self.filename = filename
        self.split = split
        self.use_processed = use_processed
        self.val_ratio = val_ratio
        self.window_size = window_size
        
        # Load data
        data, labels = self._load_ecg_data()
        
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
    
    def _load_ecg_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load ECG data from files"""
        if self.filename is None:
            # Load all ECG files and aggregate
            return self._load_all_ecg_files_aggregated()
        else:
            # Load specific file
            if self.use_processed and os.path.exists(os.path.join(self.data_dir, 'labeled')):
                return self._load_processed_ecg()
            else:
                return self._load_raw_ecg()
    
    def _load_all_ecg_files_aggregated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load ECG files and aggregate them to specific target sizes"""
        labeled_dir = os.path.join(self.data_dir, 'labeled')
        if not os.path.exists(labeled_dir):
            raise FileNotFoundError(f"ECG labeled directory not found: {labeled_dir}")
        
        # Target sizes for ECG dataset
        target_train_size = 6999
        target_test_size = 2851
        
        # Get all ECG files
        if self.split == 'train':
            ecg_dir = os.path.join(labeled_dir, 'train')
            target_size = target_train_size
        elif self.split == 'val':
            ecg_dir = os.path.join(labeled_dir, 'train')  # Use train files for validation split
            # Load full training data for proper splitting
            target_size = target_train_size
        else:  # test
            ecg_dir = os.path.join(labeled_dir, 'test')
            target_size = target_test_size
        
        if not os.path.exists(ecg_dir):
            raise FileNotFoundError(f"ECG directory not found: {ecg_dir}")
        
        ecg_files = [f for f in os.listdir(ecg_dir) if f.endswith('.pkl')]
        
        if not ecg_files:
            raise ValueError(f"No ECG files found in {ecg_dir}")
        
        # Sort files for consistent ordering
        ecg_files.sort()
        
        print(f"Loading ECG files for {self.split} split (target size: {target_size})...")
        
        all_data = []
        all_labels = []
        current_size = 0
        
        # Load files until we reach the target size
        for ecg_file in ecg_files:
            if current_size >= target_size:
                break
                
            file_path = os.path.join(ecg_dir, ecg_file)
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Convert to numpy array
                raw_data = np.array(data, dtype=np.float32)
                
                # Reshape if 1D
                if len(raw_data.shape) == 1:
                    raw_data = raw_data.reshape(-1, 1)
                
                # ECG data structure: [signal_ch0, signal_ch1, anomaly_labels]
                if raw_data.shape[1] == 3:
                    # Extract signal data (channels 0 and 1) and labels (channel 2)
                    time_series = raw_data[:, :2]  # Use only channels 0 and 1 as features
                    ground_truth_labels = raw_data[:, 2].astype(np.int64)  # Channel 2 as labels
                    
                    print(f"ECG structure detected - using channels 0-1 as features, channel 2 as labels")
                    
                    # Verify that channel 2 contains binary labels (0 and 1)
                    unique_labels = np.unique(ground_truth_labels)
                    if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
                        anomaly_ratio = np.sum(ground_truth_labels == 1) / len(ground_truth_labels)
                        print(f"  Binary anomaly labels found: {anomaly_ratio:.4f} anomaly ratio")
                    else:
                        print(f"  Warning: Channel 2 labels are not binary 0/1: {unique_labels}")
                else:
                    # Fallback for unexpected structure
                    time_series = raw_data
                    ground_truth_labels = np.zeros(len(raw_data), dtype=np.int64)
                    print(f"  Warning: Unexpected ECG structure, using all channels as features")
                
                # Truncate if this file would exceed target size
                remaining_needed = target_size - current_size
                if len(time_series) > remaining_needed:
                    time_series = time_series[:remaining_needed]
                    ground_truth_labels = ground_truth_labels[:remaining_needed]
                
                # Use appropriate labels based on split
                if self.split == 'test':
                    # For test data, use actual ground truth labels from channel 2
                    labels = ground_truth_labels
                else:
                    # For training/validation, use zeros (unsupervised learning)
                    # Channel 2 should be all zeros for training data anyway
                    labels = np.zeros(len(time_series), dtype=np.int64)
                
                all_data.append(time_series)
                all_labels.append(labels)
                current_size += len(time_series)
                
                print(f"Loaded {ecg_file}: {time_series.shape[0]} samples, {time_series.shape[1]} features")
                if self.split == 'test':
                    print(f"  Test anomalies: {np.sum(labels)} ({np.sum(labels)/len(labels):.4f} ratio)")
                
            except Exception as e:
                print(f"Warning: Could not load {ecg_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid ECG files could be loaded")
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total ECG data points loaded: {len(combined_data)} (target: {target_size})")
        print(f"ECG feature dimensions: {combined_data.shape[1]}")
        if self.split == 'test':
            print(f"Total test anomalies: {np.sum(combined_labels)} ({np.sum(combined_labels)/len(combined_labels):.4f} ratio)")
        else:
            print(f"Training/validation labels: all zeros (unsupervised)")
        
        # Apply train/val split for training and validation
        if self.split == 'train':
            # For training, use most of the data
            val_split_idx = int(len(combined_data) * (1 - self.val_ratio))
            data = combined_data[:val_split_idx]
            labels = combined_labels[:val_split_idx]
            print(f"Training split: {len(data)} timesteps")
        elif self.split == 'val':
            # For validation, use remaining portion
            val_split_idx = int(len(combined_data) * (1 - self.val_ratio))
            data = combined_data[val_split_idx:]
            labels = combined_labels[val_split_idx:]
            print(f"Validation split: {len(data)} timesteps")
            
            # Ensure validation data has enough samples for window creation
            if len(data) < self.window_size:
                # If not enough data, take a larger portion to ensure minimum samples
                min_val_size = max(self.window_size * 2, int(len(combined_data) * 0.05))  # At least 2x window size or 5% of data
                if len(combined_data) < min_val_size:
                    # If total data is too small, use all of it for validation
                    data = combined_data
                    labels = combined_labels
                else:
                    # Take the last min_val_size samples for validation
                    data = combined_data[-min_val_size:]
                    labels = combined_labels[-min_val_size:]
                print(f"Adjusted validation split to ensure sufficient data: {len(data)} timesteps")
        else:  # test
            data = combined_data
            labels = combined_labels
            print(f"Test split: {len(data)} timesteps")
        
        return data, labels

    def _load_processed_ecg(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed ECG data from pickle files"""
        # Always load from the labeled directory with proper train/test split
        if self.split == 'train':
            data_path = os.path.join(self.data_dir, 'labeled', 'train', f'{self.filename}.pkl')
        elif self.split == 'val':
            # Load training data for validation split
            data_path = os.path.join(self.data_dir, 'labeled', 'train', f'{self.filename}.pkl')
        elif self.split == 'test':
            data_path = os.path.join(self.data_dir, 'labeled', 'test', f'{self.filename}.pkl')
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ECG data file not found: {data_path}")
        
        # Load from train/test split files
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to numpy array
        raw_data = np.array(data, dtype=np.float32)
        
        # Reshape if 1D
        if len(raw_data.shape) == 1:
            raw_data = raw_data.reshape(-1, 1)
        
        # ECG data structure: [signal_ch0, signal_ch1, anomaly_labels]
        if raw_data.shape[1] == 3:
            # Extract signal data (channels 0 and 1) and labels (channel 2)
            time_series = raw_data[:, :2]  # Use only channels 0 and 1 as features
            ground_truth_labels = raw_data[:, 2].astype(np.int64)  # Channel 2 as labels
            
            print(f"ECG structure detected - using channels 0-1 as features, channel 2 as labels")
            
            # Verify that channel 2 contains binary labels (0 and 1)
            unique_labels = np.unique(ground_truth_labels)
            if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
                anomaly_ratio = np.sum(ground_truth_labels == 1) / len(ground_truth_labels)
                print(f"  Binary anomaly labels found: {anomaly_ratio:.4f} anomaly ratio")
            else:
                print(f"  Warning: Channel 2 labels are not binary 0/1: {unique_labels}")
        else:
            # Fallback for unexpected structure
            time_series = raw_data
            ground_truth_labels = np.zeros(len(raw_data), dtype=np.int64)
            print(f"  Warning: Unexpected ECG structure, using all channels as features")
        
        # Create labels based on split
        if self.split == 'train':
            # Training: all normal data (standard unsupervised anomaly detection)
            labels = np.zeros(len(time_series), dtype=np.int64)
            # Split for training only (keep validation separate)
            val_split_idx = int(len(time_series) * (1 - self.val_ratio))
            time_series = time_series[:val_split_idx]
            labels = labels[:val_split_idx]
            
            # Ensure training data has enough samples for window creation
            if len(time_series) < self.window_size:
                print(f"Warning: Training split too small ({len(time_series)} < {self.window_size}), using more data")
                # If training split is too small, use at least window_size * 2 samples
                min_train_size = max(self.window_size * 2, int(len(time_series) * 0.8))
                time_series = time_series[:min_train_size]
                labels = labels[:min_train_size]
        elif self.split == 'val':
            # Validation: create validation split from training data
            labels = np.zeros(len(time_series), dtype=np.int64)
            # Split for validation only
            val_split_idx = int(len(time_series) * (1 - self.val_ratio))
            time_series = time_series[val_split_idx:]
            labels = labels[val_split_idx:]
            
            # Ensure validation data has enough samples for window creation
            if len(time_series) < self.window_size:
                # If not enough data, take a larger portion to ensure minimum samples
                min_val_size = max(self.window_size * 2, int(len(time_series) * 0.05))  # At least 2x window size or 5% of data
                
                # Reload original data for proper handling
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                raw_data = np.array(data, dtype=np.float32)
                if len(raw_data.shape) == 1:
                    raw_data = raw_data.reshape(-1, 1)
                
                if raw_data.shape[1] == 3:
                    original_time_series = raw_data[:, :2]
                    original_labels = np.zeros(len(original_time_series), dtype=np.int64)
                else:
                    original_time_series = raw_data
                    original_labels = np.zeros(len(raw_data), dtype=np.int64)
                
                if len(original_time_series) < min_val_size:
                    # If total data is too small, use all of it for validation
                    time_series = original_time_series
                    labels = original_labels
                else:
                    # Take the last min_val_size samples for validation
                    time_series = original_time_series[-min_val_size:]
                    labels = original_labels[-min_val_size:]
                print(f"Adjusted validation split to ensure sufficient data: {len(time_series)} timesteps")
        else:  # test
            # Test: use actual ground truth labels from channel 2
            labels = ground_truth_labels
        
        return time_series, labels
    
    def _load_raw_ecg(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw ECG data from text files"""
        # Try different possible file locations
        possible_paths = [
            os.path.join(self.data_dir, f'{self.filename}.txt'),
            os.path.join(self.data_dir, 'raw', f'{self.filename}.txt')
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"ECG data file not found: {self.filename}.txt")
        
        # Load the text file
        try:
            # Try space-separated format first
            data = pd.read_csv(data_path, sep='\s+', header=None)
        except:
            try:
                # Try comma-separated format
                data = pd.read_csv(data_path, header=None)
            except:
                # Try reading as plain text
                with open(data_path, 'r') as f:
                    lines = f.readlines()
                values = []
                for line in lines:
                    try:
                        # Extract numeric values from line
                        nums = [float(x) for x in line.strip().split() if x.replace('.', '').replace('-', '').isdigit()]
                        if nums:
                            values.extend(nums)
                    except:
                        continue
                data = pd.DataFrame(values)
        
        # Convert to numpy
        time_series = data.values.astype(np.float32)
        
        # If multiple columns, use all as features; if single column, treat as univariate
        if time_series.shape[1] == 1:
            time_series = time_series.flatten()
        
        # Reshape if 1D
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
        
        # Split data based on split type
        if self.split == 'train':
            # Create train split from first part of data
            val_split_idx = int(len(time_series) * (1 - self.val_ratio))
            test_split_idx = int(len(time_series) * 0.8)
            
            time_series = time_series[:min(val_split_idx, test_split_idx)]
            labels = np.zeros(len(time_series), dtype=np.int64)
        elif self.split == 'val':
            # Create validation split from training portion
            val_split_idx = int(len(time_series) * (1 - self.val_ratio))
            test_split_idx = int(len(time_series) * 0.8)
            
            time_series = time_series[val_split_idx:test_split_idx]
            labels = np.zeros(len(time_series), dtype=np.int64)
        else:  # test
            # Create test split from last part of data
            test_split_idx = int(len(time_series) * 0.8)
            time_series = time_series[test_split_idx:]
            labels = self._create_synthetic_labels(time_series)
        
        return time_series, labels
    
    def _create_ecg_anomaly_labels(self, data: np.ndarray) -> np.ndarray:
        """Create ECG-specific anomaly labels based on medical criteria"""
        from scipy import stats
        from scipy.signal import find_peaks
        
        # ECG data typically has multiple leads (channels)
        if len(data.shape) > 1 and data.shape[1] > 1:
            # Use the lead with highest variance (usually lead II)
            lead_variances = np.var(data, axis=0)
            primary_lead = data[:, np.argmax(lead_variances)]
        else:
            primary_lead = data.flatten()
        
        labels = np.zeros(len(primary_lead), dtype=np.int64)
        
        # 1. Detect extreme amplitude anomalies (balanced medical threshold)
        z_scores = np.abs(stats.zscore(primary_lead))
        amplitude_anomalies = z_scores > 3.0  # Balanced threshold for better sensitivity
        
        # 2. Detect rhythm irregularities using peak detection
        try:
            # Find R-peaks (QRS complexes) - assuming normalized ECG data
            peaks, properties = find_peaks(primary_lead, 
                                         height=np.percentile(primary_lead, 75),
                                         distance=50)  # Minimum distance between beats
            
            if len(peaks) > 3:
                # Calculate RR intervals (time between consecutive R-peaks)
                rr_intervals = np.diff(peaks)
                
                if len(rr_intervals) > 1:
                    # Detect irregular rhythms (high variability in RR intervals)
                    rr_std = np.std(rr_intervals)
                    rr_mean = np.mean(rr_intervals)
                    
                    # Mark periods with irregular rhythm
                    for i in range(len(rr_intervals)-1):
                        current_rr = rr_intervals[i]
                        next_rr = rr_intervals[i+1]
                        
                        # If RR interval changes dramatically (> 50% change)
                        if (abs(current_rr - next_rr) > 0.5 * rr_mean or 
                            abs(current_rr - rr_mean) > 2 * rr_std):
                            
                            # Mark region around this irregular beat
                            start_idx = max(0, peaks[i] - 25)
                            end_idx = min(len(labels), peaks[i+1] + 25)
                            labels[start_idx:end_idx] = 1
                            
        except Exception:
            # If peak detection fails, fall back to amplitude-only detection
            pass
        
        # 3. Combine amplitude and rhythm anomalies
        labels = np.logical_or(labels, amplitude_anomalies).astype(np.int64)
        
        # 4. Apply smoothing to reduce isolated false positives
        # If anomaly is isolated (< 5 consecutive points), remove it
        anomaly_regions = self._find_anomaly_regions(labels)
        for start, end in anomaly_regions:
            if end - start < 5:  # Remove very short anomalies
                labels[start:end] = 0
        
        # Ensure reasonable anomaly rate (3-20% for balanced ECG detection)
        anomaly_rate = np.mean(labels)
        if anomaly_rate > 0.20:  # Too many anomalies
            # Keep only the most extreme anomalies
            scores = np.abs(stats.zscore(primary_lead))
            threshold = np.percentile(scores, 85)  # Top 15% as anomalies
            labels = (scores > threshold).astype(np.int64)
        elif anomaly_rate < 0.03:  # Too few anomalies
            # Lower the threshold slightly for better sensitivity
            scores = np.abs(stats.zscore(primary_lead))
            threshold = np.percentile(scores, 90)  # Top 10% as anomalies
            labels = (scores > threshold).astype(np.int64)
        
        return labels
    
    def _find_anomaly_regions(self, labels: np.ndarray) -> list:
        """Find contiguous regions of anomalies"""
        regions = []
        in_anomaly = False
        start = 0
        
        for i, label in enumerate(labels):
            if label == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif label == 0 and in_anomaly:
                regions.append((start, i))
                in_anomaly = False
        
        if in_anomaly:
            regions.append((start, len(labels)))
        
        return regions
    
    def _create_synthetic_labels(self, data: np.ndarray) -> np.ndarray:
        """Create synthetic anomaly labels based on statistical outliers (fallback method)"""
        from scipy import stats
        
        # Flatten if multivariate
        if len(data.shape) > 1 and data.shape[1] > 1:
            # Use first principal component or mean across features
            data_1d = np.mean(data, axis=1)
        else:
            data_1d = data.flatten()
        
        # Use z-score to identify outliers
        z_scores = np.abs(stats.zscore(data_1d))
        # Mark points with z-score > 3.2 as anomalies (balanced approach)
        labels = (z_scores > 3.2).astype(np.int64)
        
        return labels


def load_ecg_data(data_dir: str, filename: str, val_ratio: float = 0.2) -> Tuple[ECGDataset, ECGDataset, ECGDataset]:
    """
    Load ECG train, validation, and test datasets
    
    Args:
        data_dir: Directory containing ECG data
        filename: ECG filename (without extension)
        val_ratio: Ratio of training data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ECGDataset(data_dir, filename, split='train', val_ratio=val_ratio)
    val_dataset = ECGDataset(
        data_dir, 
        filename,
        split='val',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    test_dataset = ECGDataset(
        data_dir, 
        filename,
        split='test',
        val_ratio=val_ratio,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    return train_dataset, val_dataset, test_dataset


def get_available_ecg_files(data_dir: str) -> list:
    """Get list of available ECG files"""
    files = []
    
    # Check raw directory
    raw_dir = os.path.join(data_dir, 'raw')
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith('.txt'):
                files.append(f.replace('.txt', ''))
    
    # Check processed directory
    labeled_dir = os.path.join(data_dir, 'labeled', 'whole')
    if os.path.exists(labeled_dir):
        for f in os.listdir(labeled_dir):
            if f.endswith('.pkl'):
                filename = f.replace('.pkl', '')
                if filename not in files:
                    files.append(filename)
    
    # Check root directory
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            filename = f.replace('.txt', '')
            if filename not in files:
                files.append(filename)
    
    return sorted(files)