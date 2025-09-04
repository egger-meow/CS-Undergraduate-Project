# Elevator Re-leveling Anomaly Detection System

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

A dual autoencoder system for detecting elevator re-leveling events using motor current and vibration data, implementing the methodology described in `docs/README.md`.

## Project Overview

This system detects "re-leveling" events in elevators - when an elevator performs a second alignment after the first leveling attempt has excessive error. This serves as an early warning system for potential maintenance issues.

### Key Features

- **Dual Autoencoder Architecture**: Separate models for normal and abnormal data
- **Multi-channel Analysis**: Motor current (high precision) + door vibration (high recall)
- **OR Fusion Strategy**: Optimizes for recall over precision (maintenance-friendly)
- **10-fold Cross-validation**: Robust evaluation with repeated experiments
- **Clean Configuration**: Centralized config management with dataclasses

## Architecture

### Data Channels (Background.md Findings)
- **Channel 0**: Motor current (amp) - High precision (0.94), lower recall (0.62)
- **Channels 1-3**: Door vibration XYZ - High recall (0.84), lower precision (0.82) 
- **Channels 4-6**: Car vibration XYZ - Less informative (excluded from optimal combination)

### Methodology
1. **Dual Training**: Train separate 1D-CNN autoencoders on normal and abnormal data
2. **Feature Extraction**: Use reconstruction losses as 2D features (normal_loss, abnormal_loss)
3. **Classification**: Apply SVM/LogReg/kNN on 2D feature space
4. **Fusion**: OR logic between amp and vib predictions for final decision

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

The system uses a centralized configuration in `config.py`. Key settings:

```python
# Data configuration  
data_version = 'v3'
amp_channels = [0]          # Motor current
vib_channels = [1, 2, 3]    # Door vibration XYZ

# Model configuration
architecture = 'CNN1D'      # 1D-CNN as per background.md
embedding_size = 8          # For amp model
embedding_size_vib = 16     # For vib model (multi-channel)

# Evaluation configuration
n_repeats = 10              # 10-fold cross-validation
fusion_method = 'OR'        # OR fusion strategy
```

### Basic Usage

```bash
# Display current configuration
python main.py --config-info

# Train individual models
python main.py --train-amp              # Train amplitude model
python main.py --train-vib              # Train vibration model  
python main.py --train-both             # Train both models

# Evaluate with fusion
python main.py --evaluate               # Single evaluation with fusion

# Complete pipeline (background.md methodology)
python main.py --complete               # 10-repeat cross-validation
```

### Example Workflow

```bash
# 1. Train both models
python main.py --train-both

# 2. Run evaluation with fusion
python main.py --evaluate

# 3. Run complete evaluation pipeline (matches background.md results)
python main.py --complete
```

## Project Structure

```
pytorch-AE/
├── core/                   # Core system components
│   ├── data_processor.py   # Data preprocessing and loading
│   ├── dual_autoencoder.py # Dual AE implementation  
│   ├── fusion_engine.py    # OR fusion and evaluation
│   └── evaluation_pipeline.py # 10-fold cross-validation
├── models/                 # Model architectures
│   ├── architectures/
│   │   ├── CNN1D.py       # 1D-CNN encoder/decoder
│   │   ├── MLP.py         # MLP architecture
│   │   └── LSTM.py        # LSTM architecture
│   ├── AE.py              # Legacy autoencoder (deprecated)
│   └── VAE.py             # Legacy VAE (deprecated)
├── docs/                  # Documentation
│   └── background.md      # Detailed methodology and findings
├── checkpoints/           # Saved models and pipelines
├── results/               # Evaluation results
├── logs/                  # System logs
├── config.py              # Centralized configuration
├── main.py                # Main execution script
└── requirements.txt       # Dependencies
```

## Key Components

### DataProcessor (`core/data_processor.py`)
- Handles downsampling from 8192Hz to ~256Hz
- Channel selection based on background.md findings
- Consistent data length normalization
- Separate preparation for amp and vib data

### DualAutoEncoder (`core/dual_autoencoder.py`)  
- Implements dual AE training (normal + abnormal)
- Extracts 2D reconstruction loss features
- Supports different embedding sizes for amp/vib
- Includes sparsity regularization

### FusionEngine (`core/fusion_engine.py`)
- Trains classifiers on 2D features
- Implements OR fusion strategy
- Evaluates individual and fused performance
- Generates results in background.md format

### EvaluationPipeline (`core/evaluation_pipeline.py`)
- Runs 10-repeat complete pipeline evaluation
- Computes averaged metrics across experiments
- Generates final reports matching background.md methodology

## Results Format

The system outputs results in the format shown in background.md:

| Model          | Accuracy | Precision | Recall   | F1-Score |
| -------------- | -------- | --------- | -------- | -------- |
| Amp (電流0)    | 0.79     | **0.94**  | 0.62     | 0.75     |
| Vib (1,2,3)    | 0.83     | 0.82      | **0.84** | 0.83     |
| Mix (OR)       | **0.86** | 0.82      | **0.92** | **0.87** |

## Configuration Management

The refactored system uses dataclasses for clean configuration:

- `DataConfig`: Data paths, channels, sampling parameters
- `ModelConfig`: Architecture, training parameters, embedding sizes  
- `EvaluationConfig`: Cross-validation, fusion strategy, classifiers
- `SystemConfig`: Device, paths, logging, random seeds

## Migration from Legacy Code

The refactored system replaces:
- `pipeline.py` → `main.py` (cleaner CLI interface)
- `trainTest_I.py` → `core/dual_autoencoder.py` (focused dual AE)
- `trainTest_II.py` → `core/fusion_engine.py` (fusion and evaluation)
- `mix.py` → `core/fusion_engine.py` (integrated OR fusion)
- `settings.py` → `config.py` (structured configuration)

Legacy files are preserved but deprecated.

## Advanced Usage

### Custom Configurations

Modify `config.py` for different experimental setups:

```python
# Use different channels
config.data.vib_channels = [1, 2, 3, 4, 5, 6]  # Include car vibration

# Adjust model parameters
config.model.embedding_size_vib = 32           # Larger embedding for more channels

# Change fusion strategy  
config.evaluation.fusion_method = 'AND'       # More conservative fusion
```

### Custom Evaluation

```python
from core.evaluation_pipeline import EvaluationPipeline

# Run with different number of repeats
config.evaluation.n_repeats = 5
pipeline = EvaluationPipeline()
results = pipeline.run_complete_evaluation()
```

## Logging

All operations are logged to `logs/dual_ae_pipeline_TIMESTAMP.log` with configurable levels.

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- PyTorch (deep learning framework)
- scikit-learn (classifiers and evaluation)
- pandas/numpy (data processing)
- joblib (model persistence)

## Contributing

When adding new features:
1. Follow the modular architecture in `core/`
2. Update configuration in `config.py`
3. Add logging statements for debugging
4. Maintain backward compatibility with background.md methodology

## License

[Add appropriate license]
