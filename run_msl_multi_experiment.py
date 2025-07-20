#!/usr/bin/env python3
"""
Multi-experiment runner for MSL dataset (55-feature files only)
Runs experiments on all 27 MSL files with 55 features and averages the results
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
    log_file = f"results/MSL/seed_{seed}/logs/msl_multi_experiment_{timestamp}.log"
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/MSL/seed_{seed}/logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_msl_entities():
    """Get MSL entities from filesystem using standard MSL dataset approach"""
    try:
        # Import MSL dataset to use its file discovery utilities
        from data.msl import MSLDataset
        
        # Get MSL 55-feature files (C and D series) from the filesystem
        data_dir = './data/datasets/MSL'
        
        if not os.path.exists(data_dir):
            print(f"MSL data directory not found: {data_dir}")
            return []
        
        # Get files by feature dimension - we want 55-feature files
        files_by_dim = MSLDataset.get_files_by_feature_dim(data_dir)
        
        if 55 not in files_by_dim:
            print("No 55-feature MSL files found")
            return []
        
        msl_55_files = files_by_dim[55]
        print(f"Found {len(msl_55_files)} MSL 55-feature files: {msl_55_files}")
        
        return msl_55_files
        
    except Exception as e:
        print(f"Error getting MSL entities: {e}")
        return []

def run_single_experiment(filename, seed, logger):
    """Run experiment on a single MSL file using standard MSL processing"""
    logger.info(f"Starting experiment for MSL file {filename} with seed {seed}")
    
    # Use standard MSL config file
    config_file = 'configs/MSL.yaml'
    
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return None
    
    # Create a temporary config file for this experiment
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this specific MSL file and seed
    config['data']['msl']['filename'] = filename
    config['data']['msl']['feature_dim'] = 55  # Ensure 55-feature filtering
    config['experiment']['name'] = f"msl_{filename}_seed_{seed}_experiment"
    config['random_seed'] = seed
    
    # Update result paths to include seed and filename
    config['logging']['log_dir'] = f"./results/MSL/seed_{seed}/logs"
    config['logging']['model_save_dir'] = f"./results/MSL/seed_{seed}/file_{filename}/models"
    config['logging']['prediction_save_dir'] = f"./results/MSL/seed_{seed}/predictions"
    
    # Save temporary config
    temp_config_path = f'configs/MSL_{filename}_seed_{seed}.yaml'
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
            logger.error(f"Experiment failed for file {filename}")
            logger.error(f"Error output: {result.stderr}")
            return None
        
        logger.info(f"Experiment completed successfully for file {filename}")
        
        # Immediately copy results to prevent overwriting by next experiment
        # run.py creates nested directory structure: results/MSL/seed_XX/MSL/predictions/
        predictions_dir = f"results/MSL/seed_{seed}/MSL/predictions"
        temp_predictions_dir = f"results/MSL/seed_{seed}/temp_predictions_{filename}"
        
        if os.path.exists(predictions_dir):
            import shutil
            if os.path.exists(temp_predictions_dir):
                shutil.rmtree(temp_predictions_dir)
            shutil.copytree(predictions_dir, temp_predictions_dir)
            logger.info(f"Copied results for file {filename} to {temp_predictions_dir}")
        else:
            logger.error(f"Predictions directory not found: {predictions_dir}")
            logger.info(f"Looking for alternative prediction locations...")
            # Try alternative path without nested dataset directory
            alt_predictions_dir = f"results/MSL/seed_{seed}/predictions"
            if os.path.exists(alt_predictions_dir):
                logger.info(f"Found predictions in alternative location: {alt_predictions_dir}")
                import shutil
                if os.path.exists(temp_predictions_dir):
                    shutil.rmtree(temp_predictions_dir)
                shutil.copytree(alt_predictions_dir, temp_predictions_dir)
                logger.info(f"Copied results for file {filename} to {temp_predictions_dir}")
            else:
                logger.error(f"No predictions found in either location")
                return None
        
        # Parse results from the copied location
        return parse_experiment_results_from_dir(filename, temp_predictions_dir, logger)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment timed out for file {filename}")
        return None
    except Exception as e:
        logger.error(f"Error running experiment for file {filename}: {e}")
        return None
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def parse_experiment_results_from_dir(filename, predictions_dir, logger):
    """Parse the results from a specific predictions directory"""
    try:
        # Find prediction files in the specified directory
        pred_files = [f for f in os.listdir(predictions_dir) 
                     if f.startswith("predictions_window_level") and f.endswith(".csv")]
        
        if not pred_files:
            logger.error(f"No prediction results found for file {filename} in {predictions_dir}")
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
        
        logger.info(f"Individual results for file {filename} (55f): AUC={individual_metrics['auc']:.4f}, "
                   f"AU-PR={individual_metrics['aupr']:.4f}, "
                   f"F1={individual_metrics['f1']:.4f}, "
                   f"Precision={individual_metrics['precision']:.4f}, "
                   f"Recall={individual_metrics['recall']:.4f}")
        
        return {
            'filename': filename,
            'feature_dim': 55,  # MSL 55-feature files
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'predictions_for_auc': {'y_true': y_true, 'y_scores': y_scores},  # For AUC/AU-PR aggregation
            'individual_metrics': individual_metrics,
            'num_samples': len(predictions_df),
            'num_anomalies': int(predictions_df['true_label'].sum())
        }
        
    except Exception as e:
        logger.error(f"Error parsing results for file {filename}: {e}")
        return None

def aggregate_results(results_list, logger):
    """Aggregate results from all experiments using confusion matrix summation"""
    logger.info("Aggregating results from all experiments")
    
    if not results_list:
        logger.error("No valid results to aggregate")
        return None
    
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        logger.error("No valid results after filtering")
        return None
    
    logger.info(f"Aggregating {len(valid_results)} valid results")
    
    # Aggregate confusion matrices by summing
    total_tp = sum(r['confusion_matrix']['tp'] for r in valid_results)
    total_fp = sum(r['confusion_matrix']['fp'] for r in valid_results)
    total_tn = sum(r['confusion_matrix']['tn'] for r in valid_results)
    total_fn = sum(r['confusion_matrix']['fn'] for r in valid_results)
    
    # Calculate final metrics from combined confusion matrix
    final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    final_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    final_f1 = (2 * final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    
    # For AUC and AU-PR, concatenate all predictions and calculate on combined dataset
    all_y_true = np.concatenate([r['predictions_for_auc']['y_true'] for r in valid_results])
    all_y_scores = np.concatenate([r['predictions_for_auc']['y_scores'] for r in valid_results])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    final_auc = roc_auc_score(all_y_true, all_y_scores)
    final_aupr = average_precision_score(all_y_true, all_y_scores)
    
    # Calculate individual metric statistics for comparison
    individual_metrics_stats = {}
    for metric in ['auc', 'aupr', 'f1', 'precision', 'recall']:
        values = [r['individual_metrics'][metric] for r in valid_results]
        individual_metrics_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    # Summary statistics
    total_samples = sum(r['num_samples'] for r in valid_results)
    total_anomalies = sum(r['num_anomalies'] for r in valid_results)
    
    # Individual results for detailed analysis
    individual_results = []
    for r in valid_results:
        individual_results.append({
            'filename': r['filename'],
            'feature_dim': r['feature_dim'],
            'auc': r['individual_metrics']['auc'],
            'aupr': r['individual_metrics']['aupr'],
            'f1': r['individual_metrics']['f1'],
            'precision': r['individual_metrics']['precision'],
            'recall': r['individual_metrics']['recall'],
            'num_samples': r['num_samples'],
            'num_anomalies': r['num_anomalies'],
            'anomaly_ratio': r['num_anomalies'] / r['num_samples'] if r['num_samples'] > 0 else 0
        })
    
    return {
        'final_metrics': {
            'auc': final_auc,
            'aupr': final_aupr,
            'f1': final_f1,
            'precision': final_precision,
            'recall': final_recall
        },
        'combined_confusion_matrix': {
            'tp': total_tp,
            'fp': total_fp,
            'tn': total_tn,
            'fn': total_fn
        },
        'individual_metric_stats': individual_metrics_stats,
        'num_experiments': len(valid_results),
        'num_entities': len(valid_results),  # Number of files processed
        'total_samples': total_samples,
        'total_anomalies': total_anomalies,
        'overall_anomaly_ratio': total_anomalies / total_samples if total_samples > 0 else 0,
        'individual_results': individual_results
    }

def save_aggregated_results(aggregated_result, seed, logger):
    """Save the aggregated results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/MSL/seed_{seed}"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed JSON results
    json_file = f"{results_dir}/msl_multi_experiment_results_seed_{seed}_{timestamp}.json"
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
        {'Metric': 'Total Normal', 'Value': cm['tn'] + cm['fp']},
        {'Metric': '55-feature Files', 'Value': aggregated_result['num_entities']},
    ]
    
    cm_df = pd.DataFrame(confusion_matrix_data)
    cm_file = f"{results_dir}/msl_confusion_matrix_seed_{seed}_{timestamp}.csv"
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
    final_summary_file = f"{results_dir}/msl_final_summary_seed_{seed}_{timestamp}.csv"
    final_summary_df.to_csv(final_summary_file, index=False)
    

    
    # Save detailed individual results
    individual_data = aggregated_result['individual_results'].copy()
    for item in individual_data:
        item['seed'] = seed  # Add seed to each individual result
    
    individual_df = pd.DataFrame(individual_data)
    individual_csv = f"{results_dir}/msl_individual_results_seed_{seed}_{timestamp}.csv"
    individual_df.to_csv(individual_csv, index=False)
    
    logger.info(f"Results saved:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  Final Summary: {final_summary_file}")
    logger.info(f"  Confusion Matrix: {cm_file}")
    logger.info(f"  Individual Results: {individual_csv}")
    
    return json_file, final_summary_file, individual_csv, cm_file

def cleanup_temp_directories(filenames, seed, logger):
    """Clean up temporary prediction directories"""
    logger.info("Cleaning up temporary directories...")
    
    import shutil
    for filename in filenames:
        # Clean up temp prediction directories
        temp_dir = f"results/MSL/seed_{seed}/temp_predictions_{filename}"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove {temp_dir}: {e}")
        
        # Clean up file-specific result directories (optional - comment out if you want to keep them)
        # file_dir = f"results/MSL/seed_{seed}/file_{filename}"
        # if os.path.exists(file_dir):
        #     try:
        #         shutil.rmtree(file_dir)
        #         logger.debug(f"Removed file directory: {file_dir}")
        #     except Exception as e:
        #         logger.warning(f"Failed to remove file directory {file_dir}: {e}")
    
    logger.info("Temporary directory cleanup completed")

def main():
    """Main function to run multi-experiment"""
    parser = argparse.ArgumentParser(description="Run multi-experiment for MSL dataset using standard MSL processing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments.")
    args = parser.parse_args()

    logger = setup_logging(args.seed)
    logger.info(f"Starting MSL multi-experiment runner with seed {args.seed}")
    
    # Get MSL entities
    msl_entities = get_msl_entities()
    
    if not msl_entities:
        logger.error("No MSL entities found")
        return 1
    
    logger.info(f"Will run experiments on {len(msl_entities)} MSL files:")
    for i, filename in enumerate(msl_entities):
        logger.info(f"  File {i+1}: {filename}")
    
    logger.info(f"Using random seed: {args.seed}")
    logger.info(f"Results will be saved to: results/MSL/seed_{args.seed}/")
    
    # Run experiments
    results = []
    successful_experiments = 0
    
    for filename in msl_entities:
        try:
            result = run_single_experiment(filename, args.seed, logger)
            results.append(result)
            if result is not None:
                successful_experiments += 1
            else:
                logger.warning(f"Failed to get results for file {filename}")
        except Exception as e:
            logger.error(f"Unexpected error with file {filename}: {e}")
            results.append(None)
    
    logger.info(f"Completed {successful_experiments}/{len(msl_entities)} experiments successfully")
    
    if successful_experiments == 0:
        logger.error("No experiments completed successfully")
        cleanup_temp_directories(msl_entities, args.seed, logger)
        return 1
    
    # Aggregate results
    aggregated_result = aggregate_results(results, logger)
    
    if aggregated_result is None:
        logger.error("Failed to aggregate results")
        cleanup_temp_directories(msl_entities, args.seed, logger)
        return 1
    
    # Add seed information to aggregated results
    aggregated_result['experiment_seed'] = args.seed
    aggregated_result['experiment_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results
    json_file, final_summary_file, individual_csv, cm_file = save_aggregated_results(aggregated_result, args.seed, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("MSL MULTI-EXPERIMENT SUMMARY (Standard MSL Dataset)")
    logger.info("="*60)
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Experiments completed: {aggregated_result['num_experiments']}/{len(msl_entities)}")
    logger.info(f"  55-feature files: {aggregated_result['num_entities']}")
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
    cleanup_temp_directories(msl_entities, args.seed, logger)
    
    logger.info("MSL multi-experiment completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 