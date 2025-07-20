#!/usr/bin/env python3
"""
Multi-experiment runner for Yahoo dataset
Runs experiments on all 67 real_*.csv files and averages the results
"""

import os
import json
import pandas as pd
import numpy as np
import yaml
import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime
import argparse

def setup_logging(seed):
    """Setup logging for the multi-experiment runner"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"results/Yahoo/seed_{seed}/logs/yahoo_multi_experiment_{timestamp}.log"
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/Yahoo/seed_{seed}/logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_yahoo_files_legacy(data_dir):
    """Legacy function: Get all Yahoo real_*.csv files that have both normal and anomaly samples in test set (no normal training data filtering)"""
    yahoo_dir = os.path.join(data_dir, 'Yahoo')
    if not os.path.exists(yahoo_dir):
        raise FileNotFoundError(f"Yahoo directory not found: {yahoo_dir}")
    
    # Import here to avoid circular imports
    from data.yahoo import get_available_yahoo_files
    
    # Get all real_*.csv files, filtering only for test sets
    yahoo_files = get_available_yahoo_files(
        yahoo_dir, 
        filter_test_sets=True, 
        filter_train_normal=False  # Disable normal training data filtering
    )
    
    # Filter to only include real_* files (in case there are other files)
    yahoo_files = [f for f in yahoo_files if f.startswith('real_')]
    
    # Sort by number (real_1, real_2, ..., real_67)
    yahoo_files.sort(key=lambda x: int(x.split('_')[1]))
    
    return yahoo_files

def get_yahoo_files(data_dir, min_normal_ratio=0.1, min_normal_samples=400):
    """Get all Yahoo real_*.csv files that have both normal and anomaly samples in test set AND sufficient normal data in training set"""
    yahoo_dir = os.path.join(data_dir, 'Yahoo')
    if not os.path.exists(yahoo_dir):
        raise FileNotFoundError(f"Yahoo directory not found: {yahoo_dir}")
    
    # Import here to avoid circular imports
    from data.yahoo import get_available_yahoo_files
    
    # Get all real_*.csv files, filtering out those without:
    # 1. Both normal and anomaly samples in test set
    # 2. Sufficient normal data in training set
    yahoo_files = get_available_yahoo_files(
        yahoo_dir, 
        filter_test_sets=True, 
        filter_train_normal=True,
        min_normal_ratio=min_normal_ratio,
        min_normal_samples=min_normal_samples
    )
    
    # Filter to only include real_* files (in case there are other files)
    yahoo_files = [f for f in yahoo_files if f.startswith('real_')]
    
    # Sort by number (real_1, real_2, ..., real_67)
    yahoo_files.sort(key=lambda x: int(x.split('_')[1]))
    
    return yahoo_files

def run_single_experiment(yahoo_file, seed, logger):
    """Run experiment on a single Yahoo file"""
    logger.info(f"Starting experiment for {yahoo_file} with seed {seed}")
    
    # Create a temporary config file for this experiment
    with open('configs/Yahoo.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this specific Yahoo file and seed
    config['data']['yahoo']['filename'] = yahoo_file
    config['experiment']['name'] = f"yahoo_{yahoo_file}_seed_{seed}_experiment"
    config['random_seed'] = seed
    
    # Update result paths to include seed
    config['logging']['log_dir'] = f"./results/Yahoo/seed_{seed}/logs"
    config['logging']['model_save_dir'] = f"./results/Yahoo/seed_{seed}/models"
    config['logging']['prediction_save_dir'] = f"./results/Yahoo/seed_{seed}/predictions"
    
    # Save temporary config
    temp_config_path = f'configs/Yahoo_{yahoo_file}_seed_{seed}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Set environment variables to fix MKL threading issues
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'INTEL'
        
        # Run the experiment
        cmd = [sys.executable, 'run.py', '--config', temp_config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)  # 1 hour timeout
        
        if result.returncode != 0:
            logger.error(f"Experiment failed for {yahoo_file}")
            logger.error(f"Error output: {result.stderr}")
            return None
        
        logger.info(f"Experiment completed successfully for {yahoo_file}")
        
        # Immediately copy results to prevent overwriting by next experiment
        # The actual predictions are saved in a nested Yahoo directory due to how run.py handles paths
        predictions_dir = f"results/Yahoo/seed_{seed}/Yahoo/predictions"
        temp_predictions_dir = f"results/Yahoo/seed_{seed}/temp_predictions_{yahoo_file}"
        
        if os.path.exists(predictions_dir):
            import shutil
            if os.path.exists(temp_predictions_dir):
                shutil.rmtree(temp_predictions_dir)
            shutil.copytree(predictions_dir, temp_predictions_dir)
            logger.info(f"Copied results for {yahoo_file} to {temp_predictions_dir}")
        else:
            logger.error(f"Predictions directory not found: {predictions_dir}")
            logger.info(f"Looking for alternative prediction locations...")
            # Try alternative path without nested Yahoo
            alt_predictions_dir = f"results/Yahoo/seed_{seed}/predictions"
            if os.path.exists(alt_predictions_dir):
                logger.info(f"Found predictions in alternative location: {alt_predictions_dir}")
                shutil.copytree(alt_predictions_dir, temp_predictions_dir)
                logger.info(f"Copied results for {yahoo_file} to {temp_predictions_dir}")
            else:
                logger.error(f"No predictions found in either location")
                return None
        
        # Parse results from the copied location
        return parse_experiment_results_from_dir(yahoo_file, temp_predictions_dir, logger)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment timed out for {yahoo_file}")
        return None
    except Exception as e:
        logger.error(f"Error running experiment for {yahoo_file}: {e}")
        return None
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def parse_experiment_results_from_dir(yahoo_file, predictions_dir, logger):
    """Parse the results from a specific predictions directory"""
    try:
        # Find prediction files in the specified directory
        pred_files = [f for f in os.listdir(predictions_dir) 
                     if f.startswith("predictions_window_level") and f.endswith(".csv")]
        
        if not pred_files:
            logger.error(f"No prediction results found for {yahoo_file} in {predictions_dir}")
            return None
        
        # Use the most recent file (should be only one)
        pred_files.sort(key=lambda x: os.path.getmtime(os.path.join(predictions_dir, x)))
        latest_pred_file = pred_files[-1]
        
        # Load predictions
        predictions_df = pd.read_csv(os.path.join(predictions_dir, latest_pred_file))
        
        # Calculate confusion matrix and metrics
        y_true = predictions_df['true_label'].values
        y_scores = predictions_df['anomaly_score'].values
        y_pred_binary = predictions_df['predicted_label'].values
        
        from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                                    confusion_matrix, average_precision_score)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Calculate individual metrics for logging
        # Handle case where only one class is present
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            if "Only one class present" in str(e):
                # If only normal samples, AUC = 1.0 (perfect)
                # If only anomaly samples, AUC = 1.0 (perfect detection)
                unique_classes = set(y_true)
                if len(unique_classes) == 1:
                    if 0 in unique_classes:  # Only normal samples
                        auc_score = 1.0
                        logger.info(f"  {yahoo_file}: Only normal samples in test set, setting AUC=1.0")
                    else:  # Only anomaly samples
                        auc_score = 1.0
                else:
                    raise e
            else:
                raise e
        
        try:
            aupr_score = average_precision_score(y_true, y_scores)
        except ValueError as e:
            if "Only one class present" in str(e):
                unique_classes = set(y_true)
                if len(unique_classes) == 1:
                    if 0 in unique_classes:  # Only normal samples
                        aupr_score = 0.0  # No positive class to predict
                        logger.info(f"  {yahoo_file}: Only normal samples in test set, setting AU-PR=0.0")
                    else:  # Only anomaly samples  
                        aupr_score = 1.0  # All samples are positive
                else:
                    raise e
            else:
                raise e
        
        unique_classes = set(y_true)
        if len(unique_classes) == 1 and 1 in unique_classes:
            individual_metrics = {
                'auc': 1.0,
                'aupr': 1.0,
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0
            }
        else:
            individual_metrics = {
                'auc': auc_score,
                'aupr': aupr_score,
                'f1': f1_score(y_true, y_pred_binary, zero_division=0),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0)
            }
        
        logger.info(f"Individual results for {yahoo_file}: AUC={individual_metrics['auc']:.4f}, "
                   f"AU-PR={individual_metrics['aupr']:.4f}, "
                   f"F1={individual_metrics['f1']:.4f}, "
                   f"Precision={individual_metrics['precision']:.4f}, "
                   f"Recall={individual_metrics['recall']:.4f}")
        
        return {
            'yahoo_file': yahoo_file,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'predictions_for_auc': {'y_true': y_true, 'y_scores': y_scores},  # For AUC/AU-PR aggregation
            'individual_metrics': individual_metrics,
            'num_samples': len(predictions_df),
            'num_anomalies': int(predictions_df['true_label'].sum())
        }
        
    except Exception as e:
        logger.error(f"Error parsing results for {yahoo_file}: {e}")
        return None

def aggregate_results(results_list, logger):
    """Aggregate results from all experiments using confusion matrix summation"""
    logger.info("Aggregating results from all experiments")
    
    if not results_list:
        logger.error("No valid results to aggregate")
        return None
    
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        logger.error("No valid results found")
        return None
    
    # Sum confusion matrices across all experiments
    logger.info("Summing confusion matrices from all experiments...")
    
    total_tp = sum(result['confusion_matrix']['tp'] for result in valid_results)
    total_fp = sum(result['confusion_matrix']['fp'] for result in valid_results)
    total_tn = sum(result['confusion_matrix']['tn'] for result in valid_results)
    total_fn = sum(result['confusion_matrix']['fn'] for result in valid_results)
    
    logger.info(f"Combined Confusion Matrix:")
    logger.info(f"  TP: {total_tp}, FP: {total_fp}")
    logger.info(f"  FN: {total_fn}, TN: {total_tn}")
    
    # Calculate final metrics from combined confusion matrix
    logger.info("Calculating final metrics from combined confusion matrix...")
    
    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Aggregate predictions for AUC and AU-PR calculation
    logger.info("Calculating AUC and AU-PR from combined predictions...")
    all_y_true = np.concatenate([result['predictions_for_auc']['y_true'] for result in valid_results])
    all_y_scores = np.concatenate([result['predictions_for_auc']['y_scores'] for result in valid_results])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    auc = roc_auc_score(all_y_true, all_y_scores)
    aupr = average_precision_score(all_y_true, all_y_scores)
    
    final_metrics = {
        'auc': auc,
        'aupr': aupr,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    logger.info("FINAL AGGREGATED METRICS (from combined confusion matrix):")
    logger.info(f"  AUC: {final_metrics['auc']:.4f}")
    logger.info(f"  AU-PR: {final_metrics['aupr']:.4f}")
    logger.info(f"  F1: {final_metrics['f1']:.4f}")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  Recall: {final_metrics['recall']:.4f}")
    
    # Collect individual stats for comparison
    individual_stats = []
    total_samples = 0
    total_anomalies = 0
    
    for result in valid_results:
        individual_stats.append({
            'yahoo_file': result['yahoo_file'],
            'num_samples': result['num_samples'],
            'num_anomalies': result['num_anomalies'],
            'anomaly_ratio': result['num_anomalies'] / result['num_samples'],
            **result['individual_metrics']
        })
        
        total_samples += result['num_samples']
        total_anomalies += result['num_anomalies']
    
    # Calculate average of individual metrics for comparison
    avg_individual_metrics = {}
    for metric in ['auc', 'aupr', 'f1', 'precision', 'recall']:
        avg_individual_metrics[f'avg_{metric}'] = np.mean([stats[metric] for stats in individual_stats])
    
    logger.info("AVERAGE OF INDIVIDUAL METRICS (for comparison):")
    for metric, value in avg_individual_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return {
        'final_metrics': final_metrics,
        'confusion_matrix': {
            'tp': total_tp,
            'fp': total_fp,
            'tn': total_tn,
            'fn': total_fn
        },
        'individual_stats': individual_stats,
        'average_individual_metrics': avg_individual_metrics,
        'total_samples': total_samples,
        'total_anomalies': total_anomalies,
        'num_experiments': len(valid_results),
        'num_failed_experiments': len(results_list) - len(valid_results)
    }

def save_aggregated_results(aggregated_result, seed, logger):
    """Save aggregated results to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory
    results_dir = f"results/Yahoo/seed_{seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save final summary
    final_summary_path = f"{results_dir}/yahoo_final_summary_seed_{seed}_{timestamp}.csv"
    final_summary = pd.DataFrame([aggregated_result['final_metrics']])
    final_summary.to_csv(final_summary_path, index=False)
    logger.info(f"Final summary saved to {final_summary_path}")
    
    # Save confusion matrix
    confusion_matrix_path = f"{results_dir}/yahoo_confusion_matrix_seed_{seed}_{timestamp}.csv"
    confusion_matrix_df = pd.DataFrame([aggregated_result['confusion_matrix']])
    confusion_matrix_df.to_csv(confusion_matrix_path, index=False)
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
    
    # Save individual results
    individual_results_path = f"{results_dir}/yahoo_individual_results_seed_{seed}_{timestamp}.csv"
    individual_results_df = pd.DataFrame(aggregated_result['individual_stats'])
    individual_results_df.to_csv(individual_results_path, index=False)
    logger.info(f"Individual results saved to {individual_results_path}")
    
    # Save complete results as JSON
    json_results_path = f"{results_dir}/yahoo_multi_experiment_results_seed_{seed}_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Create JSON-serializable version of results
    json_results = {
        'final_metrics': {k: convert_numpy_types(v) for k, v in aggregated_result['final_metrics'].items()},
        'confusion_matrix': {k: convert_numpy_types(v) for k, v in aggregated_result['confusion_matrix'].items()},
        'average_individual_metrics': {k: convert_numpy_types(v) for k, v in aggregated_result['average_individual_metrics'].items()},
        'total_samples': convert_numpy_types(aggregated_result['total_samples']),
        'total_anomalies': convert_numpy_types(aggregated_result['total_anomalies']),
        'num_experiments': aggregated_result['num_experiments'],
        'num_failed_experiments': aggregated_result['num_failed_experiments'],
        'individual_stats': [
            {k: convert_numpy_types(v) for k, v in stats.items()} 
            for stats in aggregated_result['individual_stats']
        ]
    }
    
    with open(json_results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Complete results saved to {json_results_path}")

def cleanup_temp_directories(yahoo_files, seed, logger):
    """Clean up temporary prediction directories"""
    logger.info("Cleaning up temporary directories...")
    
    for yahoo_file in yahoo_files:
        temp_dir = f"results/Yahoo/seed_{seed}/temp_predictions_{yahoo_file}"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")

def main():
    """Main function to run Yahoo multi-experiment"""
    parser = argparse.ArgumentParser(description='Run Yahoo multi-experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./data/datasets', 
                       help='Directory containing Yahoo dataset')
    parser.add_argument('--skip-files', type=str, nargs='*', default=[], 
                       help='Yahoo files to skip (e.g., real_1 real_2)')
    parser.add_argument('--only-files', type=str, nargs='*', default=[], 
                       help='Only run these Yahoo files (e.g., real_1 real_2)')
    parser.add_argument('--min-normal-ratio', type=float, default=0.1, 
                       help='Minimum ratio of normal samples in training data (default: 0.1)')
    parser.add_argument('--min-normal-samples', type=int, default=400, 
                       help='Minimum absolute number of normal samples in training data (default: 400)')
    parser.add_argument('--disable-normal-filter', action='store_true', 
                       help='Disable filtering based on normal data in training set')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.seed)
    
    logger.info("Starting Yahoo multi-experiment")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Normal data filtering: {'disabled' if args.disable_normal_filter else 'enabled'}")
    if not args.disable_normal_filter:
        logger.info(f"  Min normal ratio: {args.min_normal_ratio}")
        logger.info(f"  Min normal samples: {args.min_normal_samples}")
    
    try:
        # Get all Yahoo files
        if args.disable_normal_filter:
            # Use old filtering (test sets only)
            yahoo_files = get_yahoo_files_legacy(args.data_dir)
        else:
            # Use new filtering (test sets + normal training data)
            yahoo_files = get_yahoo_files(args.data_dir, args.min_normal_ratio, args.min_normal_samples)
        
        # Filter files based on arguments
        if args.only_files:
            yahoo_files = [f for f in yahoo_files if f in args.only_files]
            logger.info(f"Running only specified files: {args.only_files}")
        
        if args.skip_files:
            yahoo_files = [f for f in yahoo_files if f not in args.skip_files]
            logger.info(f"Skipping files: {args.skip_files}")
        
        logger.info(f"Found {len(yahoo_files)} Yahoo files to process")
        logger.info(f"Files: {yahoo_files}")
        
        # Run experiments
        results = []
        for i, yahoo_file in enumerate(yahoo_files):
            logger.info(f"Processing file {i+1}/{len(yahoo_files)}: {yahoo_file}")
            result = run_single_experiment(yahoo_file, args.seed, logger)
            results.append(result)
        
        # Aggregate results
        aggregated_result = aggregate_results(results, logger)
        
        if aggregated_result is None:
            logger.error("Failed to aggregate results")
            return 1
        
        # Save results
        save_aggregated_results(aggregated_result, args.seed, logger)
        
        # Clean up temporary directories
        cleanup_temp_directories(yahoo_files, args.seed, logger)
        
        logger.info("Yahoo multi-experiment completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in multi-experiment: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 