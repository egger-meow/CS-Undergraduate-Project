# config.py - Centralized configuration management
import os
import torch
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data-related configuration"""
    # Data versions and paths
    data_version: str = 'v3'
    base_data_dir: str = 'D:/leveling/leveling_data'
    
    # Sampling configuration
    sample_rate: int = 256
    sample_rate_origin: int = 8192
    time_stamps: int = 1024
    stride: int = 128
    
    # Channel configuration - aligned with background.md findings
    # Channel 0: Motor current (amp)
    # Channels 1-3: Door vibration XYZ (most informative)
    # Channels 4-6: Car/Motor back vibration XYZ (less informative)
    amp_channels: List[int] = None  # [0]
    vib_channels: List[int] = None  # [1, 2, 3]
    all_channels: List[int] = None  # [0, 1, 2, 3, 4, 5, 6]
    
    # Data split configuration
    train_ratio: float = 0.85
    test_file_num: int = 50
    
    def __post_init__(self):
        if self.amp_channels is None:
            self.amp_channels = [0]
        if self.vib_channels is None:
            self.vib_channels = [1, 2, 3]
        if self.all_channels is None:
            self.all_channels = [0, 1, 2, 3, 4, 5, 6]

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Architecture selection
    autoencoder_type: str = 'AE'  # AE or VAE (background.md prefers AE)
    architecture: str = 'CNN1D'   # CNN1D, MLP, or LSTM (background.md uses 1D-CNN)
    
    # Network parameters
    embedding_size: int = 8
    embedding_size_vib: int = 16  # Different size for vibration (multi-channel)
    
    # Training parameters
    epochs: int = 300
    lr: float = 0.005
    scheduler_step_size: int = 8
    scheduler_gamma: float = 0.85
    
    # Batch sizes
    batch_size_normal: int = 32
    batch_size_abnormal: int = 4
    
    # Loss function parameters
    sparsity_param: float = 0.05
    sparsity_weight: float = 0.01

@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration"""
    # Cross-validation settings (background.md mentions 10-fold)
    n_folds: int = 10
    n_repeats: int = 10  # 10 complete pipeline repeats as mentioned
    
    # Fusion strategy (background.md uses OR fusion)
    fusion_method: str = 'OR'  # OR, AND, WEIGHTED
    
    # Classifiers to use
    classifiers: List[str] = None
    
    # Evaluation metrics focus
    optimize_for: str = 'recall'  # background.md emphasizes recall over precision
    
    def __post_init__(self):
        if self.classifiers is None:
            self.classifiers = ['SVM', 'LogisticRegression', 'kNN']

@dataclass
class SystemConfig:
    """System and runtime configuration"""
    # Device configuration
    use_cuda: bool = True
    device: str = None
    
    # Paths
    project_root: str = 'D:/course/pytorch-AE'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    logs_dir: str = 'logs'
    
    # Logging
    log_level: str = 'INFO'
    
    # Random seeds for reproducibility
    random_seed: int = 42
    torch_seed: int = 42
    
    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'

class Config:
    """Main configuration class combining all configs"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()
        
        # Create necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            os.path.join(self.system.project_root, self.system.checkpoint_dir),
            os.path.join(self.system.project_root, self.system.results_dir),
            os.path.join(self.system.project_root, self.system.logs_dir),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths based on configuration"""
        base_path = f"{self.data.base_data_dir}/{self.data.data_version}"
        return {
            'norm_train': f"{base_path}/Normal/train/",
            'abnorm_train': f"{base_path}/Abnormal/train/",
            'norm_test': f"{base_path}/Normal/test/",
            'abnorm_test': f"{base_path}/Abnormal/test/",
        }
    
    def get_model_paths(self, mode: str, date: str = None) -> Dict[str, str]:
        """Get model checkpoint paths"""
        if date is None:
            date = datetime.now().strftime("%y%m%d")
        
        checkpoint_dir = os.path.join(self.system.project_root, self.system.checkpoint_dir)
        return {
            'normal_ae': f"{checkpoint_dir}/aeNorm_{mode}_{self.data.data_version}_{self.model.architecture}_{date}.pth",
            'abnormal_ae': f"{checkpoint_dir}/aeAbnorm_{mode}_{self.data.data_version}_{self.model.architecture}_{date}.pth",
        }
    
    def get_pipeline_paths(self, mode: str) -> Dict[str, str]:
        """Get trained pipeline paths"""
        checkpoint_dir = os.path.join(self.system.project_root, self.system.checkpoint_dir)
        return {
            'svm': f"{checkpoint_dir}/svm_pipeline_{mode}.joblib",
            'knn': f"{checkpoint_dir}/knn_pipeline_{mode}.joblib", 
            'logreg': f"{checkpoint_dir}/logreg_pipeline_{mode}.joblib",
        }

# Global configuration instance
config = Config()
