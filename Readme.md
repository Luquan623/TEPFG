# TEPFG

This repository provides the official implementation of the paper  
**“TEPFG: A Robust Traffic Flow Prediction Model for Extreme Events.”**

TEPFG is designed for extreme traffic flow prediction and introduces a unified framework with hierarchical transfer fine-tuning and probability-driven gating fusion, achieving significant improvements under extreme traffic conditions on the PeMS04 and PeMS08 datasets.

---

## Data

You need to **download the PeMS04 and PeMS08 datasets** from the official sources, as they are not included in this repository due to licensing constraints. After downloading, place the datasets into the `data` directory as shown below:

```text
data/
 ├── PEMS04/
 │    └── PEMS04.npz
 ├── PEMS08/
 │    └── PEMS08.npz
```

## Training

### PeMS04

```bash
python run.py \
  -f ts_forecasting_traffic/config/TEPFG_class/PEMS04.yaml

python run.py \
  -f ts_forecasting_traffic/config/TEPFG_finetune/PEMS04.yaml
```  

### PeMS08

```bash
python run.py \
  -f ts_forecasting_traffic/config/TEPFG_class/PEMS08.yaml

python run.py \
  -f ts_forecasting_traffic/config/TEPFG_finetune/PEMS08.yaml
```


