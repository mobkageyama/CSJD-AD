#!/usr/bin/env python3
"""
CSJD-AD: Causal Stochastic Jump Diffusion Anomaly Detection
Main execution script for training and evaluation
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt
# Remove seaborn import that's causing issues
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.CSJD_AD import CausalVariationalCSJDAD, create_causal_variational_csjdad
from data.asd import ASDDataset
from data.ecg import ECGDataset
from data.smd import SMDDataset
from data.msl import MSLDataset
from data.msl import load_msl_data, load_msl_pickle_pytorch
from data.wadi import WADIDataset
from data.kpi import KPIDataset


def efficient_evaluation(all_scores, binary_labels, eval_config, logger):
    """
    Memory-efficient evaluation with improved threshold selection.
    Can be used for both point-wise and  evaluation.
    """
    logger.info(f"Starting efficient evaluation for {len(binary_labels)} samples...")
    
    # Calculate score statistics for better threshold selection
    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)
    score_median = np.median(all_scores)
    
    logger.info(f"Score statistics: mean={score_mean:.6f}, std={score_std:.6f}, median={score_median:.6f}")
    
    # Use sampling for threshold optimization on very large datasets
    if len(binary_labels) > 500000:
        logger.info("Large dataset detected. Using sampling for threshold optimization...")
        # Sample 10% of the data for threshold optimization
        sample_size = min(50000, len(binary_labels) // 10)
        sample_indices = np.random.choice(len(binary_labels), sample_size, replace=False)
        sample_scores = all_scores[sample_indices]
        sample_labels = binary_labels[sample_indices]
        
        # Find optimal threshold on sample
        if eval_config['threshold_method'] == 'best_f1':
            # Use percentile-based threshold candidates for better range
            thresholds = np.percentile(sample_scores, np.linspace(50, 99, 50))
            best_f1 = 0
            best_threshold = score_median
            
            for threshold in thresholds:
                pred_labels = (sample_scores > threshold).astype(int)
                try:
                    f1 = f1_score(sample_labels, pred_labels)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                except:
                    continue
            logger.info(f"Optimal threshold found on sample: {best_threshold:.6f} (F1: {best_f1:.4f})")
        else:
            # Use original threshold methods
            if eval_config['threshold_method'] == 'percentile':
                best_threshold = np.percentile(sample_scores, eval_config['threshold_percentile'])
            else:
                best_threshold = eval_config['fixed_threshold']
    else:
        # Improved threshold optimization for smaller datasets
        if eval_config['threshold_method'] == 'best_f1':
            # Use percentile-based threshold candidates instead of linear range
            thresholds = np.percentile(all_scores, np.linspace(50, 99, 100))
            best_f1 = 0
            best_threshold = score_median
            
            for threshold in thresholds:
                pred_labels = (all_scores > threshold).astype(int)
                try:
                    f1 = f1_score(binary_labels, pred_labels)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                except:
                    continue
            
            # Fallback to percentile if F1 optimization fails
            if best_f1 == 0:
                logger.warning("F1 optimization failed, using 95th percentile as fallback")
                best_threshold = np.percentile(all_scores, 95)
                
        elif eval_config['threshold_method'] == 'percentile':
            best_threshold = np.percentile(all_scores, eval_config['threshold_percentile'])
        else:
            best_threshold = eval_config['fixed_threshold']
    
    # Apply threshold to get final predictions
    pred_labels = (all_scores > best_threshold).astype(int)
    
    # Log threshold selection results
    logger.info(f"Selected threshold: {best_threshold:.6f}")
    logger.info(f"Predictions: {np.sum(pred_labels)}/{len(pred_labels)} anomalies ({np.sum(pred_labels)/len(pred_labels)*100:.1f}%)")
    
    # Compute final metrics
    try:
        auc_roc = roc_auc_score(binary_labels, all_scores)  # AUC-ROC
        auc_pr = average_precision_score(binary_labels, all_scores)  # AU-PR
        f1 = f1_score(binary_labels, pred_labels)
        precision = precision_score(binary_labels, pred_labels)
        recall = recall_score(binary_labels, pred_labels)
        
        # Calculate False Positive Rate (FPR)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(binary_labels, pred_labels).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        auc_roc = auc_pr = f1 = precision = recall = fpr = 0.0
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'threshold': best_threshold,
        'predictions': pred_labels
    }


from data.yahoo import YahooDataset


def cleanup_old_logs(log_dir, keep_latest=1):
    """Clean up old log files, keeping only the most recent ones"""
    log_files = list(log_dir.glob("csjdad_*.log"))
    if len(log_files) > keep_latest:
        # Sort by modification time, newest first
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        # Remove old files
        for old_file in log_files[keep_latest:]:
            old_file.unlink()


def setup_logging(config):
    """Setup logging configuration"""
    dataset_name = config['data']['dataset']
    base_results_dir = Path(config['logging']['log_dir']).parent / dataset_name
    log_dir = base_results_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old logs before creating new one
    cleanup_old_logs(log_dir)
    
    log_file = log_dir / f"csjdad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config):
    """Get computing device"""
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_dataset(config, split='train'):
    """Create dataset based on configuration"""
    data_config = config['data']
    dataset_name = data_config['dataset'].lower()
    
    # Common parameters
    val_ratio = data_config.get('val_ratio', 0.2)
    
    if dataset_name == 'asd':
        dataset = ASDDataset(
            data_dir=os.path.join(data_config['data_dir'], 'ASD'),
            filename=data_config['asd']['filename'],
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            val_ratio=val_ratio
        )
    elif dataset_name == 'ecg':
        dataset = ECGDataset(
            data_dir=os.path.join(data_config['data_dir'], 'ECG'),
            filename=data_config['ecg']['filename'],
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            val_ratio=val_ratio
        )

    elif dataset_name == 'smd':
        dataset = SMDDataset(
            data_dir=os.path.join(data_config['data_dir'], 'SMD'),
            filename=data_config['smd']['filename'],
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            val_ratio=val_ratio
        )
    elif dataset_name == 'msl':
        # Get normalization method from config
        normalization_method = data_config.get('normalization_method', 'z_score')
        
        # Use the proper load function that handles global normalization
        train_dataset, val_dataset, test_dataset = load_msl_data(
            data_dir=os.path.join(data_config['data_dir'], 'MSL'),
            filename=data_config['msl']['filename'],
            val_ratio=val_ratio,
            feature_dim=data_config['msl'].get('feature_dim', None),
            normalization_method=normalization_method,
            window_size=data_config['window_size'],
            stride=data_config['stride']
        )
        
        # Return the appropriate split
        if split == 'train':
            dataset = train_dataset
        elif split == 'val':
            dataset = val_dataset
        else:  # test
            dataset = test_dataset
    elif dataset_name == 'msl_pickle':
        # Use the new pickle-based MSL processing
        dataset = load_msl_pickle_pytorch(config, split)

    elif dataset_name == 'yahoo':
        yahoo_config = data_config.get('yahoo', {})
        dataset = YahooDataset(
            data_dir=os.path.join(data_config['data_dir'], 'Yahoo'),
            filename=yahoo_config.get('filename', None),
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            train_ratio=yahoo_config.get('train_ratio', 0.8),
            val_ratio=val_ratio
        )
    elif dataset_name == 'wadi':
        wadi_config = data_config.get('wadi', {})
        dataset = WADIDataset(
            data_dir=data_config['data_dir'],
            filename=wadi_config.get('filename', 'WADI_attackdataLABLE.csv'),
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            train_ratio=wadi_config.get('train_ratio', 0.6),
            val_ratio=val_ratio,
            feature_dim=wadi_config.get('feature_dim', 127)
        )
    elif dataset_name == 'kpi':
        kpi_config = data_config.get('kpi', {})
        dataset = KPIDataset(
            data_dir=data_config['data_dir'],
            kpi_id=kpi_config.get('kpi_id', None),
            window_size=data_config['window_size'],
            stride=data_config['stride'],
            split=split,
            val_ratio=val_ratio
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_model(config, input_dim):
    """Create CSJD-AD model"""
    model_config = config['model']
    model_config['input_dim'] = input_dim
    
    model = create_causal_variational_csjdad(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        num_env=model_config['num_env'],
        hidden_dim=model_config['hidden_dim'],
        lambda_causal=model_config['lambda_causal'],
        lambda_kl=model_config['lambda_kl'],
        dt=model_config['dt'],
        gamma=model_config['gamma']
    )
    
    return model


def train_model(model, train_loader, val_loader, config, device, logger):
    """Train the CSJD-AD model"""
    training_config = config['training']
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config['scheduler_step_size'],
        gamma=training_config['scheduler_gamma']
    )
    
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    min_improvement = training_config.get('early_stopping_min_improvement', 1e-4)
    
    model.train()
    
    for epoch in range(training_config['epochs']):
        # Training phase
        train_loss = 0.0
        train_metrics = {'recon_loss': 0.0, 'causal_loss': 0.0, 'kl_loss': 0.0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss_dict = model.compute_loss(data, outputs)
            
            # Backward pass
            loss = loss_dict['total_loss']
            
            # Skip if loss is NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_metrics['recon_loss'] += loss_dict['reconstruction_loss'].item()
            train_metrics['causal_loss'] += loss_dict['causal_loss'].item()
            train_metrics['kl_loss'] += (loss_dict['kl_u_loss'] + loss_dict['kl_e_loss']).item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        if val_loader is not None:
            val_loss, val_metrics = evaluate_model(model, val_loader, device)
            
            # Logging
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            # No validation set - use training loss for early stopping
            val_loss = train_loss
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} (no validation set)")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping with minimum improvement threshold
        if val_loss < best_val_loss - min_improvement:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if config['logging']['save_model']:
                dataset_name = config['data']['dataset']
                base_results_dir = Path(config['logging']['model_save_dir']).parent / dataset_name
                model_dir = base_results_dir / 'models'
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / 'model.pth')
            
            logger.info(f"New best validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs (best: {best_val_loss:.6f})")
            
        if patience_counter >= training_config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed!")


def evaluate_model(model, data_loader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    total_metrics = {'recon_loss': 0.0, 'causal_loss': 0.0, 'kl_loss': 0.0}
    num_batches = 0
    total_batches = len(data_loader)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            
            # Skip if data contains NaN or Inf
            if torch.isnan(data).any() or torch.isinf(data).any():
                continue
                
            outputs = model(data)
            loss_dict = model.compute_loss(data, outputs)
            
            # Skip if loss is NaN or Inf
            if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                continue
                
            total_loss += loss_dict['total_loss'].item()
            total_metrics['recon_loss'] += loss_dict['reconstruction_loss'].item()
            total_metrics['causal_loss'] += loss_dict['causal_loss'].item()
            total_metrics['kl_loss'] += (loss_dict['kl_u_loss'] + loss_dict['kl_e_loss']).item()
            num_batches += 1
    
    # Average metrics
    if num_batches > 0:
        total_loss /= num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
    else:
        total_loss = float('inf')
        for key in total_metrics:
            total_metrics[key] = float('inf')
    
    model.train()
    return total_loss, total_metrics


def test_model(model, test_loader, device, config, logger):
    """Test model and compute anomaly detection metrics using   evaluation"""
    model.eval()
    
    # Get dataset parameters
    dataset = test_loader.dataset
    window_size = dataset.window_size
    stride = dataset.stride
    original_length = len(dataset.labels)
    
    logger.info(f"Dataset info: Original length={original_length}, Window size={window_size}, Stride={stride}")
    logger.info(f"Using   evaluation (NO aggregation to original length)")
    
    # Collect  predictions and labels
    window_scores = []
    window_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, batch_window_labels) in enumerate(tqdm(test_loader, desc="Testing")):
            data = data.to(device)
            
            # Get anomaly scores for each window - aggregate across timesteps within window
            scores = model.detect_anomalies(data)  # [batch_size, seq_len]
            
            # For each window, compute a single anomaly score
     
            window_batch_scores = torch.mean(scores, dim=1)  # [batch_size] - mean across timesteps

            
            # Store  scores and labels
            window_scores.extend(window_batch_scores.cpu().numpy())
            window_labels.extend(batch_window_labels.cpu().numpy())
    
    # Convert to numpy arrays
    window_scores = np.array(window_scores)
    window_labels = np.array(window_labels)
    
    logger.info(f" evaluation: {len(window_scores)} windows")
    logger.info(f" anomaly rate: {np.mean(window_labels):.4f}")
    
    # Check if binary or multi-class
    unique_labels = np.unique(window_labels)
    logger.info(f"Unique window labels found: {unique_labels}")
    
    # For anomaly detection, convert to binary if needed (0=normal, 1=anomaly)
    if len(unique_labels) > 2:
        # Convert multi-class to binary: 0 stays 0, everything else becomes 1
        binary_labels = (window_labels > 0).astype(int)
        logger.info(f"Converting multi-class to binary: {len(unique_labels)} classes -> binary")
    else:
        binary_labels = window_labels
    
    # Use efficient evaluation for  predictions
    eval_config = config['evaluation']
    eval_results = efficient_evaluation(window_scores, binary_labels, eval_config, logger)
    
    auc_roc = eval_results['auc_roc']
    auc_pr = eval_results['auc_pr']
    f1 = eval_results['f1']
    precision = eval_results['precision']
    recall = eval_results['recall']
    fpr = eval_results['fpr']
    best_threshold = eval_results['threshold']
    pred_labels = eval_results['predictions']
    
    logger.info(f" Test Results:")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AU-PR: {auc_pr:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"FPR: {fpr:.4f}")
    logger.info(f"Best Threshold: {best_threshold:.4f}")
    
    # Save predictions if requested
    if config['logging']['save_predictions']:
        dataset_name = config['data']['dataset']
        base_results_dir = Path(config['logging']['prediction_save_dir']).parent / dataset_name
        pred_dir = base_results_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = pred_dir / 'predictions_window_level.csv'
        
        # Save  predictions
        results_df = pd.DataFrame({
            'window_id': range(len(binary_labels)),
                'true_label': binary_labels,
            'original_label': window_labels,
            'anomaly_score': window_scores,
                'predicted_label': pred_labels
            })
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(results_df)}  predictions to {csv_path}")
            
        # Save  evaluation summary
        summary_path = pred_dir / 'window_evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"CSJD-AD Evaluation Summary\n")
            f.write(f"=====================================\n")
            f.write(f"Evaluation Method:    \n")
            f.write(f"Original time series length: {original_length}\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Stride: {stride}\n")
            f.write(f"Number of windows evaluated: {len(window_scores)}\n")
            f.write(f" anomaly rate: {np.mean(binary_labels):.4f}\n")
            f.write(f"Window scoring method: Mean of timestep scores within window\n")
            f.write(f"Window labeling method: ANY anomaly rule (max of timestep labels)\n")
            f.write(f"\nMetrics:\n")
            f.write(f"AUC-ROC: {auc_roc:.4f}\n")
            f.write(f"AU-PR: {auc_pr:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"FPR: {fpr:.4f}\n")
            f.write(f"Threshold: {best_threshold:.4f}\n")
        logger.info(f"Saved  evaluation summary to {summary_path}")
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'threshold': best_threshold
    }


def main():
    parser = argparse.ArgumentParser(description='CSJD-AD: Causal Stochastic Jump Diffusion Anomaly Detection')
    parser.add_argument('--config', type=str, default='configs/ASD.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model (for test mode)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set random seed
    set_seed(config['random_seed'])
    
    # Get device
    device = get_device(config)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')
    test_dataset = create_dataset(config, 'test')
    
    # Debug dataset info
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )
    
    # Only create validation loader if validation dataset has data
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['device']['num_workers'],
            pin_memory=config['device']['pin_memory']
        )
    else:
        val_loader = None
        logger.info("No validation set - validation will be skipped during training")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )
    
    # Get input dimension from dataset
    sample_data, _ = train_dataset[0]
    input_dim = sample_data.shape[-1]
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, input_dim)
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.mode == 'train':
        # Train model
        logger.info("Starting training...")
        train_model(model, train_loader, val_loader, config, device, logger)
        
        # Load best model for testing
        if config['logging']['save_model']:
            dataset_name = config['data']['dataset']
            base_results_dir = Path(config['logging']['model_save_dir']).parent / dataset_name
            model_path = base_results_dir / 'models' / 'model.pth'
            model.load_state_dict(torch.load(model_path, weights_only=True))
            logger.info(f"Loaded best model from {model_path}")
    
    elif args.mode == 'test':
        # Load pretrained model
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path, weights_only=True))
            logger.info(f"Loaded model from {args.model_path}")
        else:
            logger.error("Model path required for test mode")
            return
    
    # Test model
    logger.info("Starting testing...")
    test_results = test_model(model, test_loader, device, config, logger)
    
    # Save results
    dataset_name = config['data']['dataset']
    base_results_dir = Path(config['logging']['log_dir']).parent / dataset_name
    results_dir = base_results_dir / 'logs'
    results_file = results_dir / 'test_results.yaml'
    
    # with open(results_file, 'w') as f:
    #     yaml.dump(test_results, f)
    
    # logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()