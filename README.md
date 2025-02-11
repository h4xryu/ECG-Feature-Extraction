# ECG Wave Delineation and Feature Extraction

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


ECG wave delineation and feature extraction system implementation, focusing on accurate detection of P, QRS, and T waves using optimized signal processing techniques.

## Overview

This project implements advanced ECG signal processing algorithms for:
- Automatic wave delineation (P, QRS, T waves)
- Feature extraction using Hermite functions
- Signal quality assessment
- Beat segmentation and analysis

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
```

## Project Structure

```
ECG_delineation/
├── Processing/
│   ├── config/          # Configuration settings
│   ├── features/        # Feature extraction
│   ├── optimize/        # Optimization algorithms
│   ├── process/         # Signal processing
│   └── utils/           # Utility functions
├── ecg_data.mat         # Sample ECG data
└── main.py             # Main execution script
```

## Quick Start

```python
# Load ECG data and initialize processor
data = loadmat('ecg_data.mat')
signal = data['xx'].squeeze()
qrs_locs = data['QRS_locs'].squeeze()
fs = float(data['fs'][0][0])

# Create config and processor
config = ECGConfig(fs=fs, p_wave=True, u_wave=False)
processor = ECGProcessor(config)

# Process ECG signal
mean_beat, test_beats = processor.prepare_data(signal, qrs_locs)
coeffs, opt_pars = processor.extract_features(test_beats)
```

## Processing Pipeline

1. **Data Preparation**
   - Load ECG signal
   - QRS complex detection
   - Beat segmentation

2. **Wave Delineation**
   - P wave detection
   - QRS complex analysis
   - T wave detection

3. **Feature Extraction**
   - Morphological features
   - Temporal features
   - Signal quality metrics

