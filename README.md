# WindowProof

**Completeness-Aware Commitment Anchoring for Tamper-Evident Mobility Anomaly Detection**

> Submitted to BlockSys 2026 (Springer LNCS)

## Overview

WindowProof is a framework that integrates blockchain-based data integrity verification with trajectory anomaly detection. It commits compact sketches of raw GPS windows on-chain before untrusted processing, enabling a three-way decision that distinguishes **integrity failures** (data tampering) from **behavioral anomalies** (unusual movement patterns).

### Key Features

- **Window-Sketch Consistency Detection**: Commits compact sketches on-chain, then detects omission/tampering via sketch residual comparison
- **WindowChain Commitment Protocol**: Smart contract-enforced contiguity, deadlines, and minimum checkpoint density
- **Three-Way Decision**: Normal / Integrity Failure / Behavioral Anomaly classification
- **CPU-only**: No GPU required; uses Isolation Forest, LOF, OC-SVM, TRAOD, iBAT as baselines

## Project Structure

```
Blocksys2026/
├── code/
│   ├── windowproof/               # Core framework
│   │   ├── capture/               # Secure capture module (window split, sketch, checkpoints)
│   │   ├── blockchain/            # WindowChain protocol (Merkle tree, smart contract)
│   │   ├── detection/             # Three-way detector + baselines (IF, LOF, OC-SVM, TRAOD, iBAT)
│   │   ├── attacks/               # Attack simulator (9 attack types)
│   │   └── utils/                 # Data loaders, metrics
│   ├── experiments/               # Experiment scripts
│   │   ├── download_data.py       # Download and preprocess datasets
│   │   ├── run_main_experiment.py # Synthetic dataset experiment
│   │   ├── run_tdrive_quick.py    # T-Drive dataset experiment
│   │   ├── run_geolife_quick.py   # GeoLife dataset experiment
│   │   └── generate_figures.py    # Generate paper figures
│   ├── configs/
│   │   └── default.yaml           # Default hyperparameters
│   └── requirements.txt
├── paper/                         # LaTeX source (Springer LNCS format)
│   ├── main.tex
│   ├── references.bib
│   ├── figures/                   # Paper figures (PDF + PNG)
│   ├── llncs.cls                  # LNCS document class
│   └── splncs04.bst               # LNCS bibliography style
├── .gitignore
└── README.md
```

## Installation

```bash
cd code
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, no GPU needed.

## Data Preparation

### Option 1: Automated Download

```bash
cd code/experiments

# Download all datasets
python download_data.py --dataset all

# Or download individually
python download_data.py --dataset tdrive
python download_data.py --dataset geolife
python download_data.py --dataset porto
```

### Option 2: Manual Download

#### T-Drive (Beijing Taxi GPS)

- **Source**: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
- **Description**: ~10,357 taxis in Beijing, 1 week, ~15M GPS points
- **Format**: `taxi_id, datetime, longitude, latitude`
- **Steps**:
  1. Download zip files from the Microsoft Research page
  2. Extract to `code/data/tdrive/`
  3. Run preprocessing: `python download_data.py --dataset tdrive` (skips download if files exist, runs preprocessing only)

#### GeoLife (Multi-modal GPS)

- **Source**: [Microsoft Research GeoLife](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/)
- **Download URL**: `https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip`
- **Description**: 17,621 trajectories from 182 users in Beijing (2007-2012), multi-modal (walking, driving, bus, bike)
- **Format**: `.plt` files with `lat, lon, 0, altitude, date_days, date, time`
- **Steps**:
  1. Download and extract to `code/data/geolife/`
  2. Run preprocessing: `python download_data.py --dataset geolife`

#### Porto Taxi (Optional)

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/339/) or [Kaggle](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)
- **Description**: 1.7M taxi trajectories in Porto over 1 year
- **Steps**:
  1. Download `train.csv` from the source
  2. Place in `code/data/porto/train.csv`

After preprocessing, sample CSV files will be generated at:
- `code/data/tdrive_sample.csv`
- `code/data/geolife_sample.csv`

## Running Experiments

```bash
cd code/experiments

# Synthetic dataset (no download needed)
python run_main_experiment.py

# T-Drive (requires data download first)
python run_tdrive_quick.py

# GeoLife (requires data download first)
python run_geolife_quick.py

# Generate paper figures from results
python generate_figures.py
```

Results are saved to `code/results*/results.json`.

## Baselines

| Method | Type | Reference |
|--------|------|-----------|
| IF-Full | Isolation Forest (all 24 features) | Liu et al., ICDM 2008 |
| LOF | Local Outlier Factor | Breunig et al., SIGMOD 2000 |
| OC-SVM | One-Class SVM (RBF kernel) | - |
| TRAOD | Trajectory outlier detection | Lee et al., ICDE 2008 |
| iBAT | Isolation-based trajectory detection | Chen et al., IEEE TITS 2013 |
| IF-Sparse | IF on checkpoint features only | - |
| IF-Sketch | IF on sketch features only | - |

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a LaTeX distribution (e.g., MiKTeX or TeX Live).

## Citation

```bibtex
@inproceedings{windowproof2026,
  title={WindowProof: Completeness-Aware Commitment Anchoring for Tamper-Evident Mobility Anomaly Detection},
  author={Anonymous},
  booktitle={BlockSys 2026, Springer LNCS},
  year={2026}
}
```

## License

This project is for academic research purposes.
