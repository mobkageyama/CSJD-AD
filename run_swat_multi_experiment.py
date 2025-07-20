#!/usr/bin/env python3
"""
Multi-experiment runner for SWAT dataset
Runs multiple experiments with different random seeds and averages the results
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
    log_file = f"results/SWAT/seed_{seed}/logs/swat_multi_experiment_{timestamp}.log"
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/SWAT/seed_{seed}/logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_single_experiment(experiment_name, seed, logger):
    """Run a single experiment"""
    logger.info(f"Starting SWAT experiment {experiment_name} with seed {seed}")
    
    # Create a temporary config file for this experiment
    with open('configs/SWAT.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this specific experiment and seed
    config['experiment']['name'] = f"swat_{experiment_name}_seed_{seed}_experiment"
    config['random_seed'] = seed
    
    # Update result paths to include seed
    config['logging']['log_dir'] = f"./results/SWAT/seed_{seed}/logs"
    config['logging']['model_save_dir'] = f"./results/SWAT/seed_{seed}/models"
    config['logging']['prediction_save_dir'] = f"./results/SWAT/seed_{seed}/predictions"
    
    # Save temporary config
    temp_config_path = f'configs/SWAT_{experiment_name}_seed_{seed}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Set environment variables to fix MKL threading issues
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'INTEL'
        
        # Run the experiment
        cmd = [sys.executable, 'run.py', '--config', temp_config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)  # 2 hour timeout
        
        if result.returncode != 0:
            logger.error(f"Experiment failed for {experiment_name}")
            logger.error(f"Error output: {result.stderr}")
            return None
        
        logger.info(f"Experiment completed successfully for {experiment_name}")
        
        # Immediately copy results to prevent overwriting by next experiment
        predictions_dir = f"results/SWAT/seed_{seed}/SWAT/predictions"
        temp_predictions_dir = f"results/SWAT/seed_{seed}/temp_predictions_{experiment_name}"
        
        if os.path.exists(predictions_dir):
            import shutil
            if os.path.exists(temp_predictions_dir):
                shutil.rmtree(temp_predictions_dir)
            shutil.copytree(predictions_dir, temp_predictions_dir)
            logger.info(f"Copied results for {experiment_name} to {temp_predictions_dir}")
        else:
            logger.error(f"Predictions directory not found: {predictions_dir}")
            # Try alternative path without nested SWAT
            alt_predictions_dir = f"results/SWAT/seed_{seed}/predictions"
            if os.path.exists(alt_predictions_dir):
                logger.info(f"Found predictions in alternative location: {alt_predictions_dir}")
                import shutil
                if os.path.exists(temp_predictions_dir):
                    shutil.rmtree(temp_predictions_dir)
                shutil.copytree(alt_predictions_dir, temp_predictions_dir)
                logger.info(f"Copied results for {experiment_name} to {temp_predictions_dir}")
            else:
                logger.error(f"No predictions found in either location")
                return None
        
        # Parse results from the copied location
        return parse_experiment_results_from_dir(experiment_name, temp_predictions_dir, logger)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment timed out for {experiment_name}")
        return None
    except Exception as e:
        logger.error(f"Error running experiment for {experiment_name}: {e}")
        return None
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def parse_experiment_results_from_dir(experiment_name, predictions_dir, logger):
    """Parse the results from a specific predictions directory"""
    try:
        # Find prediction files in the specified directory
        pred_files = [f for f in os.listdir(predictions_dir) 
                     if f.startswith("predictions_window_level") and f.endswith(".csv")]
        
        if not pred_files:
            logger.error(f"No prediction results found for {experiment_name} in {predictions_dir}")
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
        individual_metrics = {
            'auc': roc_auc_score(y_true, y_scores),
            'aupr': average_precision_score(y_true, y_scores),  # AU-PR
            'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        }
        
        logger.info(f"Individual results for {experiment_name}: AUC={individual_metrics['auc']:.4f}, "
                   f"AU-PR={individual_metrics['aupr']:.4f}, "
                   f"F1={individual_metrics['f1']:.4f}, "
                   f"Precision={individual_metrics['precision']:.4f}, "
                   f"Recall={individual_metrics['recall']:.4f}")
        
        return {
            'experiment_name': experiment_name,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'predictions_for_auc': {'y_true': y_true, 'y_scores': y_scores},  # For AUC/AU-PR aggregation
            'individual_metrics': individual_metrics,
            'num_samples': len(predictions_df),
            'num_anomalies': int(predictions_df['true_label'].sum())
        }
        
    except Exception as e:
        logger.error(f"Error parsing results for {experiment_name}: {e}")
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
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    # Aggregate AUC and AU-PR by concatenating all predictions
    all_y_true = np.concatenate([result['predictions_for_auc']['y_true'] for result in valid_results])
    all_y_scores = np.concatenate([result['predictions_for_auc']['y_scores'] for result in valid_results])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(all_y_true, all_y_scores)
    aupr = average_precision_score(all_y_true, all_y_scores)
    
    # Log final results
    logger.info(f"Final Aggregated Results:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  AU-PR: {aupr:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  FPR: {fpr:.4f}")
    
    # Average individual metrics for comparison
    avg_individual_metrics = {}
    for metric in ['auc', 'aupr', 'f1', 'precision', 'recall', 'fpr']:
        avg_individual_metrics[f'avg_{metric}'] = np.mean([result['individual_metrics'][metric] for result in valid_results])
        std_individual_metrics = np.std([result['individual_metrics'][metric] for result in valid_results])
        logger.info(f"  Average {metric.upper()}: {avg_individual_metrics[f'avg_{metric}']:.4f} ± {std_individual_metrics:.4f}")
    
    return {
        'final_metrics': {
            'auc': auc,
            'aupr': aupr,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr
        },
        'confusion_matrix': {
            'tp': total_tp,
            'fp': total_fp,
            'tn': total_tn,
            'fn': total_fn
        },
        'individual_results': valid_results,
        'averaged_individual_metrics': avg_individual_metrics,
        'num_experiments': len(valid_results),
        'total_samples': sum(result['num_samples'] for result in valid_results),
        'total_anomalies': sum(result['num_anomalies'] for result in valid_results)
    }

def save_aggregated_results(aggregated_result, seed, logger):
    """Save aggregated results to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main results JSON
    results_file = f"results/SWAT/seed_{seed}/swat_multi_experiment_results_seed_{seed}_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_result = json.loads(json.dumps(aggregated_result, default=str))
        json.dump(json_result, f, indent=2)
    
    logger.info(f"Aggregated results saved to {results_file}")
    
    # Save summary CSV
    summary_file = f"results/SWAT/seed_{seed}/swat_final_summary_seed_{seed}_{timestamp}.csv"
    summary_data = {
        'Metric': ['AUC', 'AU-PR', 'F1', 'Precision', 'Recall', 'FPR'],
        'Value': [
            aggregated_result['final_metrics']['auc'],
            aggregated_result['final_metrics']['aupr'],
            aggregated_result['final_metrics']['f1'],
            aggregated_result['final_metrics']['precision'],
            aggregated_result['final_metrics']['recall'],
            aggregated_result['final_metrics']['fpr']
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")
    
    # Save confusion matrix
    confusion_file = f"results/SWAT/seed_{seed}/swat_confusion_matrix_seed_{seed}_{timestamp}.csv"
    confusion_data = {
        'Metric': ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
        'Value': [
            aggregated_result['confusion_matrix']['tp'],
            aggregated_result['confusion_matrix']['fp'],
            aggregated_result['confusion_matrix']['tn'],
            aggregated_result['confusion_matrix']['fn']
        ]
    }
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv(confusion_file, index=False)
    logger.info(f"Confusion matrix saved to {confusion_file}")
    
    # Save individual results
    individual_file = f"results/SWAT/seed_{seed}/swat_individual_results_seed_{seed}_{timestamp}.csv"
    individual_data = []
    for result in aggregated_result['individual_results']:
        individual_data.append({
            'experiment_name': result['experiment_name'],
            'auc': result['individual_metrics']['auc'],
            'aupr': result['individual_metrics']['aupr'],
            'f1': result['individual_metrics']['f1'],
            'precision': result['individual_metrics']['precision'],
            'recall': result['individual_metrics']['recall'],
            'fpr': result['individual_metrics']['fpr'],
            'num_samples': result['num_samples'],
            'num_anomalies': result['num_anomalies']
        })
    
    individual_df = pd.DataFrame(individual_data)
    individual_df.to_csv(individual_file, index=False)
    logger.info(f"Individual results saved to {individual_file}")

def cleanup_temp_directories(experiment_names, seed, logger):
    """Clean up temporary directories"""
    for exp_name in experiment_names:
        temp_dir = f"results/SWAT/seed_{seed}/temp_predictions_{exp_name}"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

def main():
    parser = argparse.ArgumentParser(description='SWAT Multi-Experiment Runner')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_experiments', type=int, default=3, help='Number of experiments to run')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.seed)
    
    logger.info(f"Starting SWAT multi-experiment run with seed {args.seed}")
    logger.info(f"Number of experiments: {args.num_experiments}")
    
    # Create experiment names
    experiment_names = [f"exp_{i+1}" for i in range(args.num_experiments)]
    
    # Run experiments
    results = []
    for exp_name in experiment_names:
        logger.info(f"Running experiment: {exp_name}")
        result = run_single_experiment(exp_name, args.seed, logger)
        if result:
            results.append(result)
            logger.info(f"Experiment {exp_name} completed successfully")
        else:
            logger.error(f"Experiment {exp_name} failed")
    
    # Aggregate results
    if results:
        logger.info(f"Aggregating results from {len(results)} successful experiments")
        aggregated_result = aggregate_results(results, logger)
        
        if aggregated_result:
            # Save results
            save_aggregated_results(aggregated_result, args.seed, logger)
            
            # Clean up temporary directories
            cleanup_temp_directories(experiment_names, args.seed, logger)
            
            logger.info("SWAT multi-experiment run completed successfully!")
            logger.info(f"Final AUC: {aggregated_result['final_metrics']['auc']:.4f}")
            logger.info(f"Final AU-PR: {aggregated_result['final_metrics']['aupr']:.4f}")
            logger.info(f"Final F1: {aggregated_result['final_metrics']['f1']:.4f}")
        else:
            logger.error("Failed to aggregate results")
    else:
        logger.error("No experiments completed successfully")

if __name__ == "__main__":
    main() 