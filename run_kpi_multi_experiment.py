#!/usr/bin/env python3
"""
Multi-experiment runner for KPI dataset
Runs experiments on all 29 KPIs and aggregates the results using confusion matrix summation
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
import shutil

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def setup_logging(seed):
    """Setup logging for the multi-experiment runner"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"results/KPI/seed_{seed}/logs/kpi_multi_experiment_{timestamp}.log"
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/KPI/seed_{seed}/logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_single_experiment(kpi_id, seed, logger):
    """Run experiment on a single KPI"""
    logger.info(f"Starting experiment for KPI {kpi_id} with seed {seed}")
    
    # Create a temporary config file for this experiment
    with open('configs/KPI.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this specific KPI and seed
    config['data']['kpi']['kpi_id'] = kpi_id
    config['experiment']['name'] = f"kpi_{kpi_id[:8]}_seed_{seed}_experiment"
    config['random_seed'] = seed
    
    # Update result paths to include seed (following the pattern from other datasets)
    config['logging']['log_dir'] = f"./results/KPI/seed_{seed}/logs"
    config['logging']['model_save_dir'] = f"./results/KPI/seed_{seed}/models"
    config['logging']['prediction_save_dir'] = f"./results/KPI/seed_{seed}/predictions"
    
    # Create temporary config file
    temp_config_path = f"temp_config_kpi_{kpi_id[:8]}_seed_{seed}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Set environment variables to fix MKL threading issues
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'INTEL'
        
        # Run the experiment
        cmd = [
            sys.executable, "run.py", 
            "--config", temp_config_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)  # 1 hour timeout
        
        if result.returncode != 0:
            logger.error(f"Experiment failed for KPI {kpi_id}: {result.stderr}")
            return None
        
        logger.info(f"Experiment completed for KPI {kpi_id}")
        
        # Immediately copy results to prevent overwriting by next experiment
        # The actual predictions are saved in a nested KPI directory due to how run.py handles paths
        predictions_dir = f"results/KPI/seed_{seed}/KPI/predictions"
        temp_predictions_dir = f"results/KPI/seed_{seed}/temp_predictions_kpi_{kpi_id[:8]}"
        
        if os.path.exists(predictions_dir):
            import shutil
            if os.path.exists(temp_predictions_dir):
                shutil.rmtree(temp_predictions_dir)
            shutil.copytree(predictions_dir, temp_predictions_dir)
            logger.info(f"Copied results for KPI {kpi_id} to {temp_predictions_dir}")
        else:
            logger.error(f"Predictions directory not found: {predictions_dir}")
            logger.info(f"Looking for alternative prediction locations...")
            # Try alternative path without nested KPI
            alt_predictions_dir = f"results/KPI/seed_{seed}/predictions"
            if os.path.exists(alt_predictions_dir):
                logger.info(f"Found predictions in alternative location: {alt_predictions_dir}")
                import shutil
                if os.path.exists(temp_predictions_dir):
                    shutil.rmtree(temp_predictions_dir)
                shutil.copytree(alt_predictions_dir, temp_predictions_dir)
                logger.info(f"Copied results for KPI {kpi_id} to {temp_predictions_dir}")
            else:
                logger.error(f"No predictions found in either location")
                return None
        
        # Parse results from the copied location
        return parse_experiment_results_from_dir(kpi_id, temp_predictions_dir, logger)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment timed out for KPI {kpi_id}")
        return None
    except Exception as e:
        logger.error(f"Error running experiment for KPI {kpi_id}: {str(e)}")
        return None
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def get_available_kpis():
    """Get list of available KPI IDs"""
    import pandas as pd
    
    csv_path = "data/datasets/KPI/KPI.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KPI data file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df['KPI ID'].unique().tolist()

def check_kpi_anomaly_rate(kpi_id):
    """Check anomaly rate for a specific KPI in the test split"""
    import pandas as pd
    
    csv_path = "data/datasets/KPI/KPI.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KPI data file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter data for the specific KPI
    kpi_data = df[df['KPI ID'] == kpi_id].copy()
    
    if len(kpi_data) == 0:
        return 0, 0, 0  # No data found
    
    # Sort by timestamp
    kpi_data = kpi_data.sort_values('timestamp')
    
    # Extract test split (last 20% of data)
    split_idx = int(len(kpi_data) * 0.8)
    test_data = kpi_data.iloc[split_idx:]
    
    total_test_points = len(test_data)
    anomaly_points = (test_data['label'] == 1).sum()
    anomaly_rate = anomaly_points / total_test_points if total_test_points > 0 else 0
    
    return anomaly_rate, anomaly_points, total_test_points

def filter_kpis_with_anomalies(kpi_ids, logger):
    """Filter out KPIs with zero anomalies in test split"""
    valid_kpis = []
    skipped_kpis = []
    
    logger.info("Checking KPIs for anomalies in test split...")
    
    for kpi_id in kpi_ids:
        anomaly_rate, anomaly_points, total_points = check_kpi_anomaly_rate(kpi_id)
        
        if anomaly_rate > 0:
            valid_kpis.append(kpi_id)
            logger.info(f"KPI {kpi_id}: {anomaly_points}/{total_points} anomalies ({anomaly_rate:.4f} rate) - INCLUDED")
        else:
            skipped_kpis.append(kpi_id)
            logger.info(f"KPI {kpi_id}: {anomaly_points}/{total_points} anomalies ({anomaly_rate:.4f} rate) - SKIPPED")
    
    logger.info(f"Summary: {len(valid_kpis)} KPIs with anomalies, {len(skipped_kpis)} KPIs skipped (zero anomalies)")
    
    if skipped_kpis:
        logger.info(f"Skipped KPIs: {skipped_kpis}")
    
    return valid_kpis, skipped_kpis

def parse_experiment_results_from_dir(kpi_id, predictions_dir, logger):
    """Parse experiment results from the predictions directory using CSV files like other datasets"""
    try:
        # Find prediction files in the specified directory
        pred_files = [f for f in os.listdir(predictions_dir) 
                     if f.startswith("predictions_window_level") and f.endswith(".csv")]
        
        if not pred_files:
            logger.error(f"No prediction results found for KPI {kpi_id} in {predictions_dir}")
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
        
        # Calculate additional metrics for comprehensive analysis
        num_samples = len(predictions_df)
        num_anomalies = int(predictions_df['true_label'].sum())
        anomaly_rate = num_anomalies / num_samples if num_samples > 0 else 0
        
        # Create threshold from predicted labels (reverse-engineer from binary predictions)
        threshold = 0.5  # Default threshold, could be extracted from evaluation logic
        
        logger.info(f"Results for KPI {kpi_id}: AUC={individual_metrics['auc']:.4f}, "
                   f"AU-PR={individual_metrics['aupr']:.4f}, F1={individual_metrics['f1']:.4f}, "
                   f"Precision={individual_metrics['precision']:.4f}, Recall={individual_metrics['recall']:.4f}")
        
        # Create result dictionary compatible with aggregation
        result = {
            'Evaluation Method': 'CARLA-style   ',
            'Original time series length': float(num_samples + 99),  # Approximate original length
            'Window size': 100.0,
            'Stride': 1.0,
            'Number of windows evaluated': float(num_samples),
            ' anomaly rate': anomaly_rate,
            'Window scoring method': 'Mean of timestep scores within window',
            'Window labeling method': 'ANY anomaly rule (max of timestep labels)',
            'Metrics': '',
            'AUC-ROC': individual_metrics['auc'],
            'AU-PR': individual_metrics['aupr'],
            'F1': individual_metrics['f1'],
            'Precision': individual_metrics['precision'],
            'Recall': individual_metrics['recall'],
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'Threshold': threshold,
            'kpi_id': kpi_id,
            # Data needed for aggregation
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'predictions_for_auc': {'y_true': y_true, 'y_scores': y_scores},
            'individual_metrics': individual_metrics,
            'num_samples': num_samples,
            'num_anomalies': num_anomalies
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing results for KPI {kpi_id}: {str(e)}")
        return None

def aggregate_results(results_list, logger):
    """Aggregate results from multiple KPI experiments using confusion matrix summation"""
    logger.info("Aggregating results from all KPI experiments")
    
    if not results_list:
        logger.error("No results to aggregate")
        return None
    
    # Filter out None results
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        logger.error("No valid results to aggregate")
        return None
    
    logger.info(f"Aggregating {len(valid_results)} valid KPI results")
    
    # Aggregate confusion matrices by summing
    total_tp = sum(r['confusion_matrix']['tp'] for r in valid_results)
    total_fp = sum(r['confusion_matrix']['fp'] for r in valid_results)
    total_tn = sum(r['confusion_matrix']['tn'] for r in valid_results)
    total_fn = sum(r['confusion_matrix']['fn'] for r in valid_results)
    
    logger.info(f"Combined Confusion Matrix:")
    logger.info(f"  TP: {total_tp}, FP: {total_fp}")
    logger.info(f"  FN: {total_fn}, TN: {total_tn}")
    
    # Calculate final metrics from combined confusion matrix
    final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    final_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    final_f1 = (2 * final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    final_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    # For AUC and AU-PR, concatenate all predictions and calculate on combined dataset
    all_y_true = np.concatenate([r['predictions_for_auc']['y_true'] for r in valid_results])
    all_y_scores = np.concatenate([r['predictions_for_auc']['y_scores'] for r in valid_results])
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    final_auc = roc_auc_score(all_y_true, all_y_scores)
    final_aupr = average_precision_score(all_y_true, all_y_scores)
    
    logger.info("FINAL AGGREGATED METRICS (from combined confusion matrix):")
    logger.info(f"  AUC: {final_auc:.4f}")
    logger.info(f"  AU-PR: {final_aupr:.4f}")
    logger.info(f"  F1: {final_f1:.4f}")
    logger.info(f"  Precision: {final_precision:.4f}")
    logger.info(f"  Recall: {final_recall:.4f}")
    logger.info(f"  FPR: {final_fpr:.4f}")
    
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
    
    logger.info("INDIVIDUAL METRICS STATISTICS (for comparison):")
    for metric, stats in individual_metrics_stats.items():
        logger.info(f"  {metric.upper()}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    # Summary statistics
    total_samples = sum(r['num_samples'] for r in valid_results)
    total_anomalies = sum(r['num_anomalies'] for r in valid_results)
    
    # All KPIs should have anomalies (zero-anomaly KPIs are filtered out before experiments)
    logger.info(f"\nAll {len(valid_results)} KPIs have anomalies (zero-anomaly KPIs filtered out beforehand)")
    
    return {
        'final_metrics': {
            'auc': final_auc,
            'aupr': final_aupr,
            'f1': final_f1,
            'precision': final_precision,
            'recall': final_recall,
            'fpr': final_fpr
        },
        'combined_confusion_matrix': {
            'tp': total_tp,
            'fp': total_fp,
            'tn': total_tn,
            'fn': total_fn
        },
        'individual_metric_stats': individual_metrics_stats,
        'num_experiments': len(valid_results),
        'num_kpis': len(valid_results),
        'total_samples': total_samples,
        'total_anomalies': total_anomalies,
        'overall_anomaly_ratio': total_anomalies / total_samples if total_samples > 0 else 0,
        'individual_results': valid_results
    }

def save_aggregated_results(aggregated_result, individual_results, seed, logger):
    """Save aggregated results to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory
    results_dir = f"results/KPI/seed_{seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual results
    individual_df = pd.DataFrame(individual_results)
    individual_file = os.path.join(results_dir, f"kpi_individual_results_seed_{seed}_{timestamp}.csv")
    individual_df.to_csv(individual_file, index=False)
    logger.info(f"Individual results saved to {individual_file}")
    
    # Save final summary with aggregated metrics
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
    summary_file = os.path.join(results_dir, f"kpi_final_summary_seed_{seed}_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Final aggregated summary saved to {summary_file}")
    
    # Save confusion matrix
    confusion_data = {
        'Metric': ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
        'Value': [
            aggregated_result['combined_confusion_matrix']['tp'],
            aggregated_result['combined_confusion_matrix']['fp'],
            aggregated_result['combined_confusion_matrix']['tn'],
            aggregated_result['combined_confusion_matrix']['fn']
        ]
    }
    confusion_df = pd.DataFrame(confusion_data)
    confusion_file = os.path.join(results_dir, f"kpi_confusion_matrix_seed_{seed}_{timestamp}.csv")
    confusion_df.to_csv(confusion_file, index=False)
    logger.info(f"Confusion matrix saved to {confusion_file}")
    
    # Save as JSON for programmatic access
    json_file = os.path.join(results_dir, f"kpi_multi_experiment_results_seed_{seed}_{timestamp}.json")
    results_dict = {
        'aggregated_results': aggregated_result,
        'individual_results': individual_results,
        'timestamp': timestamp,
        'seed': seed
    }
    
    # Convert numpy types to native Python types for JSON serialization
    results_dict_serializable = convert_numpy_types(results_dict)
    
    with open(json_file, 'w') as f:
        json.dump(results_dict_serializable, f, indent=2)
    
    logger.info(f"JSON results saved to {json_file}")

def cleanup_temp_directories(kpi_ids, seed, logger):
    """Clean up ALL temporary directories in the KPI results directory"""
    import glob
    import shutil
    
    results_dir = f"results/KPI/seed_{seed}"
    if not os.path.exists(results_dir):
        return
    
    # Find all temp directories (both current and leftover from previous runs)
    temp_pattern = os.path.join(results_dir, "temp_predictions_*")
    temp_dirs = glob.glob(temp_pattern)
    
    logger.info(f"Found {len(temp_dirs)} temporary directories to clean up")
    
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")

def main():
    """Main function to run multi-experiment"""
    parser = argparse.ArgumentParser(description='Run multi-experiment on KPI dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--kpi-limit', type=int, default=None, help='Limit number of KPIs to test')
    parser.add_argument('--start-from', type=int, default=0, help='Start from specific KPI index')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.seed)
    
    logger.info(f"Starting KPI multi-experiment with seed {args.seed}")
    logger.info("Using aggregated F1 calculation from combined confusion matrices")
    
    try:
        # Get available KPI IDs
        all_kpi_ids = get_available_kpis()
        logger.info(f"Found {len(all_kpi_ids)} total KPIs")
        
        # Apply filtering if specified
        if args.start_from > 0:
            all_kpi_ids = all_kpi_ids[args.start_from:]
            logger.info(f"Starting from KPI index {args.start_from}")
        
        if args.kpi_limit is not None:
            all_kpi_ids = all_kpi_ids[:args.kpi_limit]
            logger.info(f"Limited to {args.kpi_limit} KPIs")
        
        # Filter out KPIs with zero anomalies in test split
        kpi_ids, skipped_kpis = filter_kpis_with_anomalies(all_kpi_ids, logger)
        
        if len(kpi_ids) == 0:
            logger.error("No KPIs with anomalies found. Cannot proceed.")
            return 1
        
        logger.info(f"Running experiments on {len(kpi_ids)} KPIs (skipped {len(skipped_kpis)} zero-anomaly KPIs)")
        
        # Run experiments on each KPI
        results = []
        for i, kpi_id in enumerate(kpi_ids):
            logger.info(f"Progress: {i+1}/{len(kpi_ids)} - Processing KPI {kpi_id}")
            result = run_single_experiment(kpi_id, args.seed, logger)
            results.append(result)
        
        # Aggregate results
        aggregated_result = aggregate_results(results, logger)
        
        if aggregated_result is None:
            logger.error("Failed to aggregate results")
            return 1
        
        # Add skipped KPIs information to aggregated result
        aggregated_result['skipped_kpis'] = skipped_kpis
        aggregated_result['total_kpis_found'] = len(all_kpi_ids)
        
        # Save results
        save_aggregated_results(aggregated_result, aggregated_result['individual_results'], args.seed, logger)
        
        # Clean up temporary directories
        cleanup_temp_directories(kpi_ids, args.seed, logger)
        
        logger.info("KPI multi-experiment completed successfully")
        
        # Print summary
        print(f"\nExperiment Summary:")
        print(f"Total KPIs found: {aggregated_result['total_kpis_found']}")
        print(f"KPIs with anomalies: {len(kpi_ids)}")
        print(f"KPIs skipped (zero anomalies): {len(skipped_kpis)}")
        if skipped_kpis:
            print(f"Skipped KPI IDs: {skipped_kpis}")
        print(f"Successful experiments: {aggregated_result['num_experiments']}")
        print(f"Success rate: {aggregated_result['num_experiments']/len(kpi_ids)*100:.1f}%")
        print(f"\nAggregated Results (from combined confusion matrix):")
        print(f"  AUC: {aggregated_result['final_metrics']['auc']:.4f}")
        print(f"  AU-PR: {aggregated_result['final_metrics']['aupr']:.4f}")
        print(f"  F1: {aggregated_result['final_metrics']['f1']:.4f}")
        print(f"  Precision: {aggregated_result['final_metrics']['precision']:.4f}")
        print(f"  Recall: {aggregated_result['final_metrics']['recall']:.4f}")
        print(f"  FPR: {aggregated_result['final_metrics']['fpr']:.4f}")
        
        # Compare with individual metrics average
        individual_stats = aggregated_result['individual_metric_stats']
        print(f"\nComparison - Average of Individual F1 scores: {individual_stats['f1']['mean']:.4f} ± {individual_stats['f1']['std']:.4f}")
        print(f"Aggregated F1 from combined confusion matrix: {aggregated_result['final_metrics']['f1']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 