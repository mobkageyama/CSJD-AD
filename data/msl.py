import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import os
import ast
import pickle as pk
from .default_ts import DefaultTimeSeriesDataset


# MSL dataset - User's implementation
class MSL_Dataset():
    def __init__(self, dataset_pth, entities):
        self.dims = 55
        with open(dataset_pth, 'rb') as file:
            self.dat = pk.load(file)
        if entities != 'all':
            print(entities)
            self.dat['x_trn'] = [self.dat['x_trn'][entity] for entity in entities]
            self.dat['x_tst'] = [self.dat['x_tst'][entity] for entity in entities]
            self.dat['lab_tst'] = [self.dat['lab_tst'][entity] for entity in entities]
            self.num_entity = len(entities)
        else:
            self.num_entity = 27

    def preprocess(self, params):
        # parameters
        dl = params.dl
        stride = params.stride
        tst_stride = dl if params.tst_stride == 'no_rep' else params.tst_stride

        # Import prep here to avoid circular imports
        try:
            import utils.preprocess as prep
        except ImportError:
            # Fallback for when utils.preprocess is not available
            print("Warning: utils.preprocess not available, using dummy preprocessing")
            return self._dummy_preprocess(params)

        if params.train_method == 'train_per_entity':
            if params.entity_id == self.num_entity:
                return None
            print(f'using entity {params.entity_id}/{self.num_entity-1}')
            dat = {}
            dat['x_trn'] = self.dat['x_trn'][params.entity_id]
            dat['x_tst'] = self.dat['x_tst'][params.entity_id]
            dat['lab_tst'] = self.dat['lab_tst'][params.entity_id]
            self.num_entity = 1
            dat = prep.preprocess(dat, params, self.dims, self.num_entity, None)
            return prep.window_stride(dat['x_trn'], dat['x_tst'], dat['lab_tst'], self.num_entity, dl, stride, tst_stride)
        else:
            # preprocess self.dat by each entity!
            x_trn_all, x_tst_all, lab_tst_all = [], [], []
            for entity_id in range(self.num_entity):
                dat_ent = {}
                for key in self.dat.keys():
                    dat_ent[key] = self.dat[key][entity_id]
                dat = prep.preprocess(dat_ent, params, self.dims, self.num_entity, entity_id, quiet=True)
                x_trn_all.append(dat['x_trn'])
                x_tst_all.append(dat['x_tst'])
                lab_tst_all.append(dat['lab_tst'])

            # No additional channels - keep original 55 features

            return prep.window_stride(x_trn_all, x_tst_all, lab_tst_all, self.num_entity, dl, stride, tst_stride)

    def _dummy_preprocess(self, params):
        """Dummy preprocessing when utils.preprocess is not available"""
        print("Using dummy preprocessing - returning raw data")
        
        if params.train_method == 'train_per_entity':
            if params.entity_id >= self.num_entity:
                return None
            print(f'Dummy preprocessing: using entity {params.entity_id}/{self.num_entity-1}')
            # Extract data for specific entity
            x_trn = self.dat['x_trn'][params.entity_id]
            x_tst = self.dat['x_tst'][params.entity_id] 
            lab_tst = self.dat['lab_tst'][params.entity_id]
            
            return x_trn, x_tst, lab_tst
        else:
            # Return all data for train_all mode
            return self.dat['x_trn'], self.dat['x_tst'], self.dat['lab_tst']


class MSLDataset(DefaultTimeSeriesDataset):
    """
    MSL (Mars Science Laboratory) Dataset 
    Contains spacecraft telemetry data with anomaly detection tasks
    
    Two types of files:
    - 25 features: 54 files (A, B, E, F, G, M, P, R, S, T series)
    - 55 features: 28 files (C, D series)
    
    This class maintains backward compatibility with the old interface
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
        val_ratio: float = 0.2,
        feature_dim: Optional[int] = None,
        normalization_method: str = 'z_score',
        global_data_stats: Optional[dict] = None,
        normalization_stats: Optional[dict] = None
    ):
        """
        Initialize MSL dataset
        
        Args:
            data_dir: Directory containing MSL data files
            filename: Data filename (e.g., 'A-1', 'C-1', 'P-1')
            window_size: Size of sliding window
            stride: Step size for sliding window
            transform: Data transformations
            split: Dataset split ('train', 'val', or 'test')
            mean: Normalization mean (for backward compatibility)
            std: Normalization std (for backward compatibility)
            val_ratio: Ratio of training data to use for validation
            feature_dim: Expected feature dimension (25 or 55), None for auto-detect
            normalization_method: Normalization method ('z_score', 'min_max', 'robust', 'global_z_score', 'global_min_max')
            global_data_stats: Global statistics for global normalization methods
            normalization_stats: Normalization statistics for train-only methods
        """
        self.data_dir = data_dir
        self.filename = filename
        self.split = split
        self.val_ratio = val_ratio
        self.feature_dim = feature_dim
        self.window_size = window_size  # Store window_size for dynamic val_ratio adjustment
        
        # Load anomaly label information
        self.anomaly_info = self._load_anomaly_info()
        
        # Load data
        data, labels = self._load_msl_data()
        
        # Initialize parent class with enhanced normalization
        super().__init__(
            data=data,
            labels=labels,
            window_size=window_size,
            stride=stride,
            transform=transform,
            split=split,
            mean=mean,
            std=std,
            normalization_method=normalization_method,
            normalization_stats=normalization_stats,
            global_data_stats=global_data_stats
        )
    
    def _load_anomaly_info(self) -> dict:
        """Load anomaly information from labeled_anomalies.csv"""
        anomaly_file = os.path.join(self.data_dir, 'labeled_anomalies.csv')
        
        if not os.path.exists(anomaly_file):
            return {}
        
        df = pd.read_csv(anomaly_file)
        anomaly_info = {}
        
        # Filter for MSL spacecraft only (55-feature files) and exclude T-10 (no labels)
        msl_df = df[df['spacecraft'] == 'MSL'].copy()
        
        for _, row in msl_df.iterrows():
            chan_id = row['chan_id']
            
            # Parse anomaly sequences using string manipulation (same as user's code)
            labs = row['anomaly_sequences']
            labs_s = labs.replace('[', '').replace(']', '').replace(' ', '').split(',')
            anomaly_sequences = [[int(labs_s[i]), int(labs_s[i+1])] for i in range(0, len(labs_s), 2)]
            
            num_values = row['num_values']
            
            anomaly_info[chan_id] = {
                'anomaly_sequences': anomaly_sequences,
                'num_values': num_values,
                'spacecraft': row['spacecraft'],
                'class': row.get('class', 'unknown')
            }
        
        return anomaly_info
    
    def _load_msl_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load MSL data from numpy files"""
        if self.filename is None:
            # Load all MSL files and aggregate by feature dimension
            return self._load_all_msl_files_aggregated()
        else:
            # Load specific file
            return self._load_single_msl_file()
    
    def _load_single_msl_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single MSL file"""
        if self.split == 'train':
            # Load training data
            data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.npy')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            # Load numpy data
            data = np.load(data_path).astype(np.float32)
            
            # Training data assumed to be normal
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Adjust val_ratio if file is too short for window_size
            adjusted_val_ratio = self._get_adjusted_val_ratio(len(data), self.window_size)
            
            # Split for training only (keep validation separate)
            val_split_idx = int(len(data) * (1 - adjusted_val_ratio))
            data = data[:val_split_idx]
            labels = labels[:val_split_idx]
            
        elif self.split == 'val':
            # Load training data for validation split
            data_path = os.path.join(self.data_dir, 'train', f'{self.filename}.npy')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            # Load numpy data
            data = np.load(data_path).astype(np.float32)
            
            # Training data assumed to be normal
            labels = np.zeros(len(data), dtype=np.int64)
            
            # Adjust val_ratio if file is too short for window_size
            adjusted_val_ratio = self._get_adjusted_val_ratio(len(data), self.window_size)
            
            # If val_ratio is 0 (file too short), create minimal validation data from end of training data
            if adjusted_val_ratio == 0.0:
                # Use last window_size samples for validation
                data = data[-self.window_size:]
                labels = labels[-self.window_size:]
                print(f"  Using last {self.window_size} samples for validation")
            else:
                # Split for validation only
                val_split_idx = int(len(data) * (1 - adjusted_val_ratio))
                data = data[val_split_idx:]
                labels = labels[val_split_idx:]
            
        else:  # test
            # Load test data and create labels from anomaly info
            data_path = os.path.join(self.data_dir, 'test', f'{self.filename}.npy')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Test data not found: {data_path}")
            
            # Load numpy data
            data = np.load(data_path).astype(np.float32)
            
            # Create labels from anomaly information
            labels = self._create_labels_from_anomaly_info(self.filename, len(data))
        
        return data, labels
    
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
        
        # Cap at 0.4 to ensure we still have enough training data
        adjusted_ratio = min(min_val_ratio, 0.4)
        
        # If even max val_ratio doesn't give enough samples, skip validation (use 0)
        max_possible_val_samples = int(data_length * adjusted_ratio)
        if max_possible_val_samples < min_val_samples:
            print(f"  File {self.filename} too short for validation with window_size={window_size}. "
                  f"Skipping validation (val_ratio=0)")
            return 0.0
        
        if adjusted_ratio != self.val_ratio:
            print(f"  Adjusting val_ratio for {self.filename}: {self.val_ratio:.3f} → {adjusted_ratio:.3f} "
                  f"(val samples: {int(data_length * adjusted_ratio)})")
        
        return adjusted_ratio
    
    def _create_labels_from_anomaly_info(self, filename: str, data_length: int) -> np.ndarray:
        """Create binary labels from anomaly sequence information"""
        labels = np.zeros(data_length, dtype=np.int64)
        
        if filename in self.anomaly_info:
            anomaly_sequences = self.anomaly_info[filename]['anomaly_sequences']
            
            for start, end in anomaly_sequences:
                # Ensure indices are within bounds
                start = max(0, min(start, data_length - 1))
                end = max(0, min(end, data_length - 1))
                
                if start <= end:
                    labels[start:end + 1] = 1
        
        return labels
    
    def _load_all_msl_files_aggregated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all MSL files and aggregate them by feature dimension"""
        # Get all available files
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        if self.split in ['train', 'val']:
            data_dir = train_dir
        else:
            data_dir = test_dir
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"MSL directory not found: {data_dir}")
        
        # Get all MSL files
        msl_files = [f.replace('.npy', '') for f in os.listdir(data_dir) if f.endswith('.npy')]
        msl_files.sort()
        
        # Group files by feature dimension if specified and exclude T-10
        if self.feature_dim is not None:
            filtered_files = []
            for filename in msl_files:
                # Exclude T-10 as it has no anomaly labels
                if filename == 'T-10':
                    continue
                file_path = os.path.join(data_dir, f'{filename}.npy')
                data_sample = np.load(file_path)
                if data_sample.shape[1] == self.feature_dim:
                    filtered_files.append(filename)
            msl_files = filtered_files
        else:
            # Still exclude T-10 even if not filtering by feature dimension
            msl_files = [f for f in msl_files if f != 'T-10']
        
        if not msl_files:
            raise ValueError(f"No MSL files found with feature dimension {self.feature_dim} in {data_dir}")
        
        print(f"Loading {len(msl_files)} MSL files for {self.split} split...")
        if self.feature_dim:
            print(f"Filtering for {self.feature_dim}-feature files...")
        
        all_data = []
        all_labels = []
        
        for filename in msl_files:
            try:
                if self.split == 'train':
                    # Load training data
                    data_path = os.path.join(train_dir, f'{filename}.npy')
                    data = np.load(data_path).astype(np.float32)
                    
                    # Training data assumed to be normal
                    labels = np.zeros(len(data), dtype=np.int64)
                    
                    # Adjust val_ratio if file is too short for window_size
                    # Temporarily set filename for adjustment calculation
                    temp_filename = self.filename
                    self.filename = filename
                    adjusted_val_ratio = self._get_adjusted_val_ratio(len(data), self.window_size)
                    self.filename = temp_filename
                    
                    # Split for training only (keep validation separate)
                    val_split_idx = int(len(data) * (1 - adjusted_val_ratio))
                    data = data[:val_split_idx]
                    labels = labels[:val_split_idx]
                    
                elif self.split == 'val':
                    # Load training data for validation split
                    data_path = os.path.join(train_dir, f'{filename}.npy')
                    data = np.load(data_path).astype(np.float32)
                    
                    # Training data assumed to be normal
                    labels = np.zeros(len(data), dtype=np.int64)
                    
                    # Adjust val_ratio if file is too short for window_size
                    # Temporarily set filename for adjustment calculation
                    temp_filename = self.filename
                    self.filename = filename
                    adjusted_val_ratio = self._get_adjusted_val_ratio(len(data), self.window_size)
                    self.filename = temp_filename
                    
                    # If val_ratio is 0 (file too short), create minimal validation data from end of training data
                    if adjusted_val_ratio == 0.0:
                        # Use last window_size samples for validation
                        data = data[-self.window_size:]
                        labels = labels[-self.window_size:]
                        print(f"  Using last {self.window_size} samples for validation from {filename}")
                    else:
                        # Split for validation only
                        val_split_idx = int(len(data) * (1 - adjusted_val_ratio))
                        data = data[val_split_idx:]
                        labels = labels[val_split_idx:]
                    
                else:  # test
                    # Load test data and create labels
                    data_path = os.path.join(test_dir, f'{filename}.npy')
                    data = np.load(data_path).astype(np.float32)
                    
                    # Create labels from anomaly information
                    labels = self._create_labels_from_anomaly_info(filename, len(data))
                
                # Check feature dimension consistency
                if len(all_data) > 0 and data.shape[1] != all_data[0].shape[1]:
                    print(f"Warning: {filename} has {data.shape[1]} features, expected {all_data[0].shape[1]}. Skipping...")
                    continue
                
                all_data.append(data)
                all_labels.append(labels)
                
                print(f"  Loaded {filename}: {data.shape[0]} samples, {data.shape[1]} features")
                if self.split == 'test':
                    anomaly_count = np.sum(labels)
                    anomaly_ratio = anomaly_count / len(labels) if len(labels) > 0 else 0
                    print(f"    Test anomalies: {anomaly_count} ({anomaly_ratio:.4f} ratio)")
                
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid MSL files could be loaded")
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total aggregated data points: {len(combined_data)}")
        print(f"Total features: {combined_data.shape[1]}")
        print(f"Total anomalies: {np.sum(combined_labels)}")
        print(f"Anomaly ratio: {np.sum(combined_labels)/len(combined_labels):.4f}")
        
        return combined_data, combined_labels
    
    @staticmethod
    def get_files_by_feature_dim(data_dir: str) -> dict:
        """Get MSL files grouped by feature dimension, excluding T-10 (no anomaly labels)"""
        train_dir = os.path.join(data_dir, 'train')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"MSL train directory not found: {train_dir}")
        
        files_by_dim = {}
        
        for filename in os.listdir(train_dir):
            if filename.endswith('.npy'):
                file_path = os.path.join(train_dir, filename)
                data = np.load(file_path)
                feature_dim = data.shape[1]
                filename_without_ext = filename.replace('.npy', '')
                
                # Exclude T-10 as it has no anomaly labels in labeled_anomalies.csv
                if filename_without_ext == 'T-10':
                    continue
                
                if feature_dim not in files_by_dim:
                    files_by_dim[feature_dim] = []
                
                files_by_dim[feature_dim].append(filename_without_ext)
        
        # Sort files within each dimension
        for dim in files_by_dim:
            files_by_dim[dim].sort()
        
        return files_by_dim


def load_msl_data(
    data_dir: str, 
    filename: str, 
    val_ratio: float = 0.2, 
    feature_dim: Optional[int] = None,
    normalization_method: str = 'z_score',
    window_size: int = 100,
    stride: int = 1
) -> Tuple[MSLDataset, MSLDataset, MSLDataset]:
    """
    Load MSL train, validation, and test datasets with enhanced normalization
    
    Args:
        data_dir: Directory containing MSL data
        filename: Data filename (e.g., 'A-1', 'C-1', 'P-1')
        val_ratio: Ratio of training data to use for validation
        feature_dim: Expected feature dimension (25 or 55), None for auto-detect
        normalization_method: Normalization method ('z_score', 'min_max', 'robust', 'global_z_score', 'global_min_max')
        window_size: Size of sliding window
        stride: Step size for sliding window
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # For global normalization methods, we need to load both train and test data first
    if normalization_method.startswith('global_'):
        print(f"Using global normalization method: {normalization_method}")
        
        # Load raw train and test data directly
        if filename is None:
            # For aggregated data, load all files
            train_data = _load_raw_msl_files(data_dir, 'train', feature_dim)
            test_data = _load_raw_msl_files(data_dir, 'test', feature_dim)
        else:
            # For single file, load the complete training and test data
            train_path = os.path.join(data_dir, 'train', f'{filename}.npy')
            test_path = os.path.join(data_dir, 'test', f'{filename}.npy')
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Required files not found: {train_path} or {test_path}")
            
            train_data = np.load(train_path).astype(np.float32)
            test_data = np.load(test_path).astype(np.float32)
        
        # Compute global statistics
        global_data_stats = DefaultTimeSeriesDataset.compute_global_stats(train_data, test_data)
        
        print(f"Global normalization stats computed:")
        print(f"  Train data range: [{np.min(train_data):.4f}, {np.max(train_data):.4f}]")
        print(f"  Test data range: [{np.min(test_data):.4f}, {np.max(test_data):.4f}]")
        print(f"  Combined data range: [{np.min(global_data_stats['global_min']):.4f}, {np.max(global_data_stats['global_max']):.4f}]")
        
        # Create datasets with global normalization
        train_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='train', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method,
            global_data_stats=global_data_stats
        )
        val_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='val', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method,
            global_data_stats=global_data_stats
        )
        test_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='test', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method,
            global_data_stats=global_data_stats
        )
        
    else:
        # Standard normalization (train-only statistics)
        train_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='train', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method
        )
        val_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='val', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method,
            normalization_stats=train_dataset.normalization_stats
        )
        test_dataset = MSLDataset(
            data_dir, filename, window_size=window_size, stride=stride,
            split='test', val_ratio=val_ratio, 
            feature_dim=feature_dim, normalization_method=normalization_method,
            normalization_stats=train_dataset.normalization_stats
        )
    
    return train_dataset, val_dataset, test_dataset


def _load_raw_msl_files(data_dir: str, split: str, feature_dim: Optional[int] = None) -> np.ndarray:
    """Load raw MSL files for global statistics computation."""
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"MSL {split} directory not found: {split_dir}")
    
    all_data = []
    msl_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
    
    # Filter by feature dimension if specified
    if feature_dim is not None:
        filtered_files = []
        for filename in msl_files:
            # Exclude T-10 as it has no anomaly labels
            if filename.replace('.npy', '') == 'T-10':
                continue
            file_path = os.path.join(split_dir, filename)
            data_sample = np.load(file_path)
            if data_sample.shape[1] == feature_dim:
                filtered_files.append(filename)
        msl_files = filtered_files
    else:
        # Still exclude T-10 even if not filtering by feature dimension
        msl_files = [f for f in msl_files if f.replace('.npy', '') != 'T-10']
    
    if not msl_files:
        raise ValueError(f"No MSL files found with feature dimension {feature_dim} in {split_dir}")
    
    print(f"Loading {len(msl_files)} MSL files for global stats from {split}...")
    for filename in msl_files:
        file_path = os.path.join(split_dir, filename)
        data = np.load(file_path).astype(np.float32)
        all_data.append(data)
        print(f"  Loaded {filename}: {data.shape[0]} samples, {data.shape[1]} features")
    
    if not all_data:
        raise ValueError(f"No MSL files found in {split_dir}")
    
    combined_data = np.concatenate(all_data, axis=0)
    print(f"Combined {split} data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")
    return combined_data


def load_msl_pickle(dataset_pth: str, entities='all'):
    """
    Load MSL data from pickle file exactly like the user's code
    
    Args:
        dataset_pth: Path to the MSL pickle file (e.g., 'data/datasets/MSL/MSL.pk')
        entities: 'all' to load all entities, or list of entity indices
        
    Returns:
        MSL_Dataset instance that can be used for preprocessing
    """
    return MSL_Dataset(dataset_pth, entities)


class MSLPickleDataset(Dataset):
    """
    PyTorch Dataset adapter for MSL_Dataset (pickle-based)
    Bridges between the user's MSL_Dataset and PyTorch Dataset interface
    """
    
    def __init__(self, x_data, y_data, window_size=100, stride=1, transform=None):
        """
        Initialize MSL PyTorch Dataset
        
        Args:
            x_data: Training/test data from MSL_Dataset preprocessing
            y_data: Labels data from MSL_Dataset preprocessing  
            window_size: Window size for sequences
            stride: Stride for window creation
            transform: Optional transforms
        """
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        
        # Handle both single entity and multi-entity data
        if isinstance(x_data, list):
            # Multi-entity data - concatenate all entities
            self.data = np.concatenate(x_data, axis=0)
            self.labels = np.concatenate(y_data, axis=0)
        else:
            # Single entity data
            self.data = x_data
            self.labels = y_data
            
        # Ensure data is float32
        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        
        # Create sliding windows
        self.windows = []
        self.window_labels = []
        
        for i in range(0, len(self.data) - window_size + 1, stride):
            window = self.data[i:i + window_size]
            # Use the last label in the window (or majority vote)
            window_label = self.labels[i + window_size - 1]
            
            self.windows.append(window)
            self.window_labels.append(window_label)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
        
        print(f"Created MSL pickle dataset with {len(self.windows)} windows")
        print(f"Data shape: {self.data.shape}, Window shape: {self.windows.shape}")
        print(f"Anomaly ratio: {np.sum(self.window_labels) / len(self.window_labels):.4f}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        
        if self.transform:
            window = self.transform(window)
            
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_msl_pickle_pytorch(config, split='train'):
    """
    Load MSL pickle data and create PyTorch datasets
    
    Args:
        config: Config dictionary containing MSL pickle settings
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        MSLPickleDataset instance
    """
    data_config = config['data']
    msl_config = data_config['msl_pickle']
    
    # Get pickle file path - use direct path to avoid duplication
    pickle_file = msl_config['pickle_file']
    if pickle_file.startswith('./') or os.path.isabs(pickle_file):
        pickle_path = pickle_file
    else:
        # Construct path: data_dir should be ./data/datasets, pickle_file should be MSL/MSL.pk
        pickle_path = os.path.join('./data/datasets', pickle_file)
    
    # If file doesn't exist, try to create it
    if not os.path.exists(pickle_path):
        print(f"Pickle file not found at {pickle_path}, attempting to create it...")
        try:
            import sys
            import os as os_mod
            sys.path.insert(0, os_mod.path.dirname(os_mod.path.dirname(os_mod.path.abspath(__file__))))
            from preprocess_msl_pickle import preprocess_msl_data
            preprocess_msl_data('./data/datasets/MSL')
            print(f"Created pickle file successfully!")
        except Exception as e:
            print(f"Failed to create pickle file: {e}")
    
    # Create parameters object for MSL_Dataset
    class MSLParams:
        def __init__(self, msl_config):
            self.dl = msl_config.get('dl', data_config['window_size'])
            self.stride = msl_config.get('stride', data_config['stride'])
            self.tst_stride = msl_config.get('tst_stride', 'no_rep')
            self.train_method = msl_config.get('train_method', 'train_per_entity')
            self.entity_id = msl_config.get('entity_id', 0)
    
    # Load MSL dataset
    msl_dataset = load_msl_pickle(pickle_path, msl_config.get('entities', 'all'))
    
    # Create parameters
    params = MSLParams(msl_config)
    
    # Preprocess data
    result = msl_dataset.preprocess(params)
    
    if result is None:
        raise ValueError(f"MSL preprocessing returned None for entity {params.entity_id}")
    
    x_trn, x_tst, lab_tst = result
    
    # Create PyTorch datasets based on split
    if split == 'train':
        # For training, use training data (assume labels are zeros for training)
        if isinstance(x_trn, list):
            # Multi-entity case
            train_labels = [np.zeros(len(x), dtype=np.int64) for x in x_trn]
        else:
            # Single entity case
            train_labels = np.zeros(len(x_trn), dtype=np.int64)
            
        return MSLPickleDataset(
            x_trn, 
            train_labels,
            window_size=data_config['window_size'],
            stride=data_config['stride']
        )
    elif split == 'val':
        # For validation, use a portion of training data
        if isinstance(x_trn, list):
            val_data = [x[-len(x)//5:] for x in x_trn]  # Use last 20% for validation
            val_labels = [np.zeros(len(x), dtype=np.int64) for x in val_data]
        else:
            val_data = x_trn[-len(x_trn)//5:]
            val_labels = np.zeros(len(val_data), dtype=np.int64)
        
        return MSLPickleDataset(
            val_data,
            val_labels,
            window_size=data_config['window_size'],
            stride=data_config['stride']
        )
    else:  # test
        # For testing, use test data with labels
        return MSLPickleDataset(
            x_tst,
            lab_tst,
            window_size=data_config['window_size'],
            stride=data_config['stride']
        )