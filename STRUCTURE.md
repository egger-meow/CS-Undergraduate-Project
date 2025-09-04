# Repository Structure

This document describes the cleaned up repository structure for the Elevator Re-leveling Anomaly Detection System.

## Current Structure

```
pytorch-AE/
├── 📁 core/                    # Core system components (NEW)
│   ├── data_processor.py       # Data preprocessing and loading
│   ├── dual_autoencoder.py     # Dual AE implementation
│   ├── fusion_engine.py        # OR fusion and evaluation
│   └── evaluation_pipeline.py  # 10-fold cross-validation
│
├── 📁 models/                  # Model architectures
│   ├── architectures/
│   │   ├── CNN1D.py           # 1D-CNN encoder/decoder (UPDATED)
│   │   ├── MLP.py             # MLP architecture
│   │   └── LSTM.py            # LSTM architecture
│   ├── AE.py                  # Legacy autoencoder (kept for reference)
│   └── VAE.py                 # Legacy VAE (kept for reference)
│
├── 📁 docs/                   # Documentation
│   ├── background.md          # Detailed methodology and findings
│   └── img/                   # Images and figures
│
├── 📁 legacy/                 # Legacy files (MOVED HERE)
│   ├── README.md             # Explanation of legacy files
│   └── trainTest_II_dataSets/ # Old training data sets
│
├── 📄 config.py              # Centralized configuration (NEW)
├── 📄 main.py                # Main execution script (NEW)
├── 📄 README.md              # Project documentation (CLEANED)
├── 📄 requirements.txt       # Dependencies
├── 📄 .gitignore            # Updated git ignore rules
└── 📄 STRUCTURE.md           # This file
```

## Generated During Runtime

The following directories will be created when running the system:

```
├── 📁 checkpoints/           # Saved models (auto-created)
├── 📁 results/               # Evaluation results (auto-created)
└── 📁 logs/                  # System logs (auto-created)
```

## Key Changes Made

### ✅ Cleaned Up
- **README.md**: Removed legacy autoencoder content, focused on elevator project
- **File organization**: Moved legacy result files to `legacy/` folder
- **Configuration**: Replaced scattered settings with centralized `config.py`
- **Git ignore**: Updated to properly handle generated files

### 🆕 New Components
- **`core/`**: Modular system components implementing background.md methodology
- **`config.py`**: Structured configuration with dataclasses
- **`main.py`**: Clean CLI interface replacing complex pipeline scripts

### 📦 Legacy (Preserved but Organized)
- **`models/AE.py`, `models/VAE.py`**: Old implementations kept for reference
- **`legacy/`**: Contains old result files and datasets

### 🗑️ Removed/Deprecated
- **`pipeline.py`**: Replaced by `main.py`
- **`trainTest_I.py`, `trainTest_II.py`**: Functionality moved to `core/`
- **`mix.py`**: Integrated into `core/fusion_engine.py`
- **`settings.py`**: Replaced by `config.py`
- **`datasets.py`**: Functionality moved to `core/data_processor.py`

## Usage

The repository is now focused and clean. Use:

```bash
# Display current configuration
python main.py --config-info

# Train and evaluate (new clean interface)
python main.py --train-both --evaluate

# Complete 10-fold evaluation (background.md methodology)
python main.py --complete
```

## Benefits of New Structure

1. **🎯 Focused**: Repository is dedicated to elevator re-leveling detection
2. **🧩 Modular**: Clear separation of concerns in `core/` components  
3. **⚙️ Configurable**: Centralized configuration management
4. **📚 Documented**: Comprehensive documentation and clean README
5. **🔄 Maintainable**: Easy to extend and modify individual components
6. **🚀 Production-ready**: Proper logging, error handling, and deployment structure
