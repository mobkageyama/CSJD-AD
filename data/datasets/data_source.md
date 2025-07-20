# Anomaly Detection Datasets

This document provides information about the datasets used in the CSJD-AD anomaly detection experiments.

## Dataset Sources and Descriptions

| Dataset | Features | Source | Notes |
|---------|----------|--------|-------|
| **ASD** | 19 features | https://github.com/zhhlee/InterFusion/tree/main/data | Publicly available |
| **ECG** | 2 features | https://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip | Publicly available |
| **MSL** | 55 features | https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl | Publicly available |
| **SMD** | 38 features | https://github.com/NetManAIOps/OmniAnomaly/tree/master/ | Publicly available |
| **WADI** | 127 features | https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ | Request form needed |
| **Yahoo** | 1 feature | https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 | Request form needed |
| **KPI** | Variable features | https://github.com/NetManAIOps/KPI-Anomaly-Detection | Publicly available |

## Usage Notes

- **Request Required**: WADI and Yahoo datasets require filling out request forms from their respective organizations, which do not allow redistribution.
- **Dataset Placement**: Place downloaded datasets under the following directory structure:
  ```
  data/datasets/
  ├── ASD/
  ├── ECG/
  ├── MSL/
  ├── SMD/
  ├── WADI/
  ├── Yahoo/
  └── KPI/
  ```