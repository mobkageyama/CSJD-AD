# CSJD-AD: Causal Stochastic Jump Diffusion Anomaly Detection (in submission)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1+cu121-orange.svg)





##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd CSJD-AD
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare datasets**: Place your datasets in the `data/datasets/` directory following the expected structure:
```
data/datasets/
├── ASD/
├── ECG/
├── KPI/
├── MSL/
├── SMD/
├── WADI/
└── Yahoo/
```

## Usage

Run comprehensive experiments across all files in a dataset:

```bash
# ASD dataset (all omi-1 to omi-12 files)
python run_asd_multi_experiment.py 

# ECG dataset (all ECG files)
python run_ecg_multi_experiment.py 

# SMD dataset (all machine files)
python run_smd_multi_experiment.py

# Yahoo dataset (all real/synthetic files)
python run_yahoo_multi_experiment.py 

# KPI dataset (all 29 KPIs)
python run_kpi_multi_experiment.py 

# MSL dataset (55-feature files)
python run_msl_multi_experiment.py 

# WADI dataset
./run_wadi.sh
```


## Results and Evaluation

### Metrics

The framework evaluates models using multiple metrics:
- **AUC**: Area Under the ROC Curve
- **AU-PR**: Area Under the Precision-Recall Curve  
- **F1**: F1-Score
- **Precision**: Precision score
- **Recall**: Recall score

### Result Files

Results are saved in structured format under `results/`:

```
results/
├── ASD/seed_42/
│   ├── asd_final_summary_seed_42_*.csv      # Aggregated metrics
│   ├── asd_individual_results_seed_42_*.csv # Per-file results
│   ├── asd_confusion_matrix_seed_42_*.csv   # Confusion matrices
│   └── asd_multi_experiment_results_seed_42_*.json # Full results
└── [Other datasets...]
```
