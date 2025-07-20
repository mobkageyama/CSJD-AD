#!/usr/bin/env python3
"""
Multi-experiment runner for ASD dataset
Runs experiments on all 12 omi files and averages the results
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
    log_file = f"results/ASD/seed_{seed}/logs/asd_multi_experiment_{timestamp}.log"
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/ASD/seed_{seed}/logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_single_experiment(omi_file, seed, logger):
    """Run experiment on a single omi file"""
    logger.info(f"Starting experiment for {omi_file} with seed {seed}")
    
    # Create a temporary config file for this experiment
    with open('configs/ASD.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this specific omi file and seed
    config['data']['asd']['filename'] = omi_file
    config['experiment']['name'] = f"asd_{omi_file}_seed_{seed}_experiment"
    config['random_seed'] = seed
    
    # Update result paths to include seed
    config['logging']['log_dir'] = f"./results/ASD/seed_{seed}/logs"
    config['logging']['model_save_dir'] = f"./results/ASD/seed_{seed}/models"
    config['logging']['prediction_save_dir'] = f"./results/ASD/seed_{seed}/predictions"
    
    # Save temporary config
    temp_config_path = f'configs/ASD_{omi_file}_seed_{seed}.yaml'
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
            logger.error(f"Experiment failed for {omi_file}")
            logger.error(f"Error output: {result.stderr}")
            return None
        
        logger.info(f"Experiment completed successfully for {omi_file}")
        
        # Immediately copy results to prevent overwriting by next experiment
        # The actual predictions are saved in a nested ASD directory due to how run.py handles paths
        predictions_dir = f"results/ASD/seed_{seed}/ASD/predictions"
        temp_predictions_dir = f"results/ASD/seed_{seed}/temp_predictions_{omi_file}"
        
        if os.path.exists(predictions_dir):
            import shutil
            if os.path.exists(temp_predictions_dir):
                shutil.rmtree(temp_predictions_dir)
            shutil.copytree(predictions_dir, temp_predictions_dir)
            logger.info(f"Copied results for {omi_file} to {temp_predictions_dir}")
        else:
            logger.error(f"Predictions directory not found: {predictions_dir}")
            logger.info(f"Looking for alternative prediction locations...")
            # Try alternative path without nested ASD
            alt_predictions_dir = f"results/ASD/seed_{seed}/predictions"
            if os.path.exists(alt_predictions_dir):
                logger.info(f"Found predictions in alternative location: {alt_predictions_dir}")
                shutil.copytree(alt_predictions_dir, temp_predictions_dir)
                logger.info(f"Copied results for {omi_file} to {temp_predictions_dir}")
            else:
                logger.error(f"No predictions found in either location")
                return None
        
        # Parse results from the copied location
        return parse_experiment_results_from_dir(omi_file, temp_predictions_dir, logger)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment timed out for {omi_file}")
        return None
    except Exception as e:
        logger.error(f"Error running experiment for {omi_file}: {e}")
        return None
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def parse_experiment_results_from_dir(omi_file, predictions_dir, logger):
    """Parse the results from a specific predictions directory"""
    try:
        # Find prediction files in the specified directory
        pred_files = [f for f in os.listdir(predictions_dir) 
                     if f.startswith("predictions_window_level") and f.endswith(".csv")]
        
        if not pred_files:
            logger.error(f"No prediction results found for {omi_file} in {predictions_dir}")
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
            'recall': recall_score(y_true, y_pred_binary, zero_division=0)
        }
        
        logger.info(f"Individual results for {omi_file}: AUC={individual_metrics['auc']:.4f}, "
                   f"AU-PR={individual_metrics['aupr']:.4f}, "
                   f"F1={individual_metrics['f1']:.4f}, "
                   f"Precision={individual_metrics['precision']:.4f}, "
                   f"Recall={individual_metrics['recall']:.4f}")
        
        return {
            'omi_file': omi_file,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'predictions_for_auc': {'y_true': y_true, 'y_scores': y_scores},  # For AUC/AU-PR aggregation
            'individual_metrics': individual_metrics,
            'num_samples': len(predictions_df),
            'num_anomalies': int(predictions_df['true_label'].sum())
        }
        
    except Exception as e:
        logger.error(f"Error parsing results for {omi_file}: {e}")
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
            'omi_file': result['omi_file'],
            'num_samples': result['num_samples'],
            'num_anomalies': result['num_anomalies'],
            'anomaly_ratio': result['num_anomalies'] / result['num_samples'],
            **result['individual_metrics']
        })
        
        total_samples += result['num_samples']
        total_anomalies += result['num_anomalies']
    
    # Calculate individual metric statistics for comparison
    individual_metric_stats = {}
    for metric in ['auc', 'aupr', 'f1', 'precision', 'recall']:
        values = [stat[metric] for stat in individual_stats]
        individual_metric_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    aggregated_result = {
        'num_experiments': len(valid_results),
        'total_samples': total_samples,
        'total_anomalies': total_anomalies,
        'overall_anomaly_ratio': total_anomalies / total_samples if total_samples > 0 else 0,
        'combined_confusion_matrix': {
            'tp': int(total_tp), 'fp': int(total_fp), 
            'tn': int(total_tn), 'fn': int(total_fn)
        },
        'final_metrics': final_metrics,  # Metrics from combined confusion matrix
        'individual_metric_stats': individual_metric_stats,  # Stats of individual metrics
        'individual_results': individual_stats
    }
    
    return aggregated_result

def save_aggregated_results(aggregated_result, seed, logger):
    """Save the aggregated results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/ASD/seed_{seed}"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed JSON results
    json_file = f"{results_dir}/asd_multi_experiment_results_seed_{seed}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(aggregated_result, f, indent=2, default=str)
    
    # Save confusion matrix summary
    cm = aggregated_result['combined_confusion_matrix']
    confusion_matrix_data = [
        {'Metric': 'True Positives (TP)', 'Value': cm['tp']},
        {'Metric': 'False Positives (FP)', 'Value': cm['fp']},
        {'Metric': 'True Negatives (TN)', 'Value': cm['tn']},
        {'Metric': 'False Negatives (FN)', 'Value': cm['fn']},
        {'Metric': 'Total Samples', 'Value': cm['tp'] + cm['fp'] + cm['tn'] + cm['fn']},
        {'Metric': 'Total Anomalies', 'Value': cm['tp'] + cm['fn']},
        {'Metric': 'Total Normal', 'Value': cm['tn'] + cm['fp']}
    ]
    
    cm_df = pd.DataFrame(confusion_matrix_data)
    cm_file = f"{results_dir}/asd_confusion_matrix_seed_{seed}_{timestamp}.csv"
    cm_df.to_csv(cm_file, index=False)
    
    # Save final metrics summary with comparison
    final_summary_data = []
    for metric, value in aggregated_result['final_metrics'].items():
        # Also include individual stats for comparison
        individual_stats = aggregated_result['individual_metric_stats'][metric]
        final_summary_data.append({
            'Metric': metric.upper(),
            'Final_Aggregated': value,
            'Individual_Mean': individual_stats['mean'],
            'Individual_Std': individual_stats['std'],
            'Individual_Min': individual_stats['min'],
            'Individual_Max': individual_stats['max'],
            'Method': 'Confusion_Matrix_Sum' if metric in ['f1', 'precision', 'recall'] else 'Score_Concatenation',
            'Seed': seed
        })
    
    final_summary_df = pd.DataFrame(final_summary_data)
    final_summary_file = f"{results_dir}/asd_final_summary_seed_{seed}_{timestamp}.csv"
    final_summary_df.to_csv(final_summary_file, index=False)
    
    # Save detailed individual results
    individual_data = aggregated_result['individual_results'].copy()
    for item in individual_data:
        item['seed'] = seed  # Add seed to each individual result
    
    individual_df = pd.DataFrame(individual_data)
    individual_csv = f"{results_dir}/asd_individual_results_seed_{seed}_{timestamp}.csv"
    individual_df.to_csv(individual_csv, index=False)
    
    logger.info(f"Results saved:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  Final Summary: {final_summary_file}")
    logger.info(f"  Confusion Matrix: {cm_file}")
    logger.info(f"  Individual Results: {individual_csv}")
    
    return json_file, final_summary_file, individual_csv, cm_file

def cleanup_temp_directories(omi_files, seed, logger):
    """Clean up temporary prediction directories"""
    logger.info("Cleaning up temporary directories...")
    
    import shutil
    for omi_file in omi_files:
        temp_dir = f"results/ASD/seed_{seed}/temp_predictions_{omi_file}"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove {temp_dir}: {e}")
    
    logger.info("Temporary directory cleanup completed")

def main():
    """Main function to run multi-experiment"""
    parser = argparse.ArgumentParser(description="Run multi-experiment for ASD dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments.")
    args = parser.parse_args()

    logger = setup_logging(args.seed)
    logger.info(f"Starting ASD multi-experiment runner with seed {args.seed}")
    
    # List of all omi files
    omi_files = [f"omi-{i}" for i in range(1, 13)]
    
    logger.info(f"Will run experiments on {len(omi_files)} omi files: {omi_files}")
    logger.info(f"Using random seed: {args.seed}")
    logger.info(f"Results will be saved to: results/ASD/seed_{args.seed}/")
    
    # Run experiments
    results = []
    successful_experiments = 0
    
    for omi_file in omi_files:
        try:
            result = run_single_experiment(omi_file, args.seed, logger)
            results.append(result)
            if result is not None:
                successful_experiments += 1
            else:
                logger.warning(f"Failed to get results for {omi_file}")
        except Exception as e:
            logger.error(f"Unexpected error with {omi_file}: {e}")
            results.append(None)
    
    logger.info(f"Completed {successful_experiments}/{len(omi_files)} experiments successfully")
    
    if successful_experiments == 0:
        logger.error("No experiments completed successfully")
        cleanup_temp_directories(omi_files, args.seed, logger)
        return 1
    
    # Aggregate results
    aggregated_result = aggregate_results(results, logger)
    
    if aggregated_result is None:
        logger.error("Failed to aggregate results")
        cleanup_temp_directories(omi_files, args.seed, logger)
        return 1
    
    # Add seed information to aggregated results
    aggregated_result['experiment_seed'] = args.seed
    aggregated_result['experiment_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results
    json_file, final_summary_file, individual_csv, cm_file = save_aggregated_results(aggregated_result, args.seed, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ASD MULTI-EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Experiments completed: {aggregated_result['num_experiments']}/12")
    logger.info(f"Total samples: {aggregated_result['total_samples']}")
    logger.info(f"Total anomalies: {aggregated_result['total_anomalies']}")
    logger.info(f"Overall anomaly ratio: {aggregated_result['overall_anomaly_ratio']:.4f}")
    
    logger.info("\n🎯 FINAL AGGREGATED METRICS (calculated on combined dataset):")
    for metric, value in aggregated_result['final_metrics'].items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Display confusion matrix details
    cm = aggregated_result['combined_confusion_matrix']
    logger.info(f"\n🔢 COMBINED CONFUSION MATRIX:")
    logger.info(f"  TP: {cm['tp']:,}, FP: {cm['fp']:,}")
    logger.info(f"  FN: {cm['fn']:,}, TN: {cm['tn']:,}")
    logger.info(f"  Total: {cm['tp'] + cm['fp'] + cm['tn'] + cm['fn']:,} samples")
    
    logger.info("\n📊 Individual Experiment Statistics (for comparison):")
    for metric in ['auc', 'aupr', 'f1', 'precision', 'recall']:
        stats = aggregated_result['individual_metric_stats'][metric]
        logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"(range: {stats['min']:.4f} - {stats['max']:.4f})")
    
    logger.info(f"\nDetailed results saved to:")
    logger.info(f"  📋 Final Summary: {final_summary_file}")
    logger.info(f"  📊 Confusion Matrix: {cm_file}")
    logger.info(f"  📝 Individual Results: {individual_csv}")
    logger.info(f"  📄 Full JSON: {json_file}")
    logger.info("="*60)
    
    # Clean up temporary directories
    cleanup_temp_directories(omi_files, args.seed, logger)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 