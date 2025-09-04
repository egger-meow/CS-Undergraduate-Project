# core/data_processor.py - Clean data preprocessing pipeline
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import logging

from config import config

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data preprocessing for elevator re-leveling detection.
    Implements the methodology described in background.md:
    - Downsampling from 8192Hz to ~256Hz
    - Channel selection (amp: 0, vib: 1,2,3)
    - Normalization
    - Stage 3-5 focus
    """
    
    def __init__(self):
        self.config = config
        self.data_paths = self.config.get_data_paths()
        
    def find_smallest_file(self, directory: str) -> Tuple[str, int]:
        """Find the smallest file in directory to determine minimum length"""
        smallest_file = None
        smallest_size = float('inf')
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    if file_size < smallest_size:
                        smallest_size = file_size
                        smallest_file = file_path
                        
        return smallest_file, smallest_size
    
    def find_shortest_data_length(self, directories: List[str]) -> int:
        """Find shortest data length across all directories for consistent preprocessing"""
        shortest_len = None
        downsample_factor = self.config.data.sample_rate_origin // self.config.data.sample_rate
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
                
            smallest_file, _ = self.find_smallest_file(directory)
            if smallest_file:
                df = pd.read_csv(smallest_file)
                # Use all channels initially to determine length
                data = df.iloc[::downsample_factor, :].values
                length = data.shape[0]
                shortest_len = length if shortest_len is None else min(length, shortest_len)
                
        if shortest_len is None:
            logger.warning("No valid data found, using default time_stamps")
            return self.config.data.time_stamps
            
        return shortest_len
    
    def load_channel_data(self, directory: str, channels: List[int], 
                         is_normal: bool = True) -> np.ndarray:
        """
        Load data from directory for specified channels.
        Based on background.md findings:
        - Channel 0: Motor current (amp) - high precision, lower recall
        - Channels 1-3: Door vibration XYZ - high recall, lower precision  
        - Channels 4-6: Car vibration XYZ - less informative
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        directory_path = Path(directory)
        files = [f for f in directory_path.iterdir() 
                if f.is_file() and f.suffix == '.csv' and f.name != '.DS_Store']
        
        if not files:
            raise ValueError(f"No CSV files found in {directory}")
            
        # Find consistent data length
        all_dirs = [self.data_paths['norm_train'], self.data_paths['abnorm_train'],
                   self.data_paths['norm_test'], self.data_paths['abnorm_test']]
        shortest_len = self.find_shortest_data_length(all_dirs)
        
        data_list = []
        downsample_factor = self.config.data.sample_rate_origin // self.config.data.sample_rate
        
        logger.info(f"Loading {len(files)} files from {directory}")
        logger.info(f"Target channels: {channels}")
        
        for file_path in tqdm(files, desc=f"Loading {'normal' if is_normal else 'abnormal'} data"):
            try:
                df = pd.read_csv(file_path)
                
                # Downsample and select channels
                data = df.iloc[::downsample_factor, channels].values
                
                # Ensure consistent length
                data = data[:shortest_len, :]
                
                # Normalize per sample
                data = self._normalize_data(data)
                
                # Interpolate to desired time_stamps if needed
                if data.shape[0] != self.config.data.time_stamps:
                    data = self._interpolate_data(data, self.config.data.time_stamps)
                
                # Transpose to (channels, time_stamps) format
                data = data.T
                data_list.append(data)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
                
        if not data_list:
            raise ValueError(f"No valid data loaded from {directory}")
            
        # Stack all samples
        data_array = np.stack(data_list, axis=0)  # (n_samples, channels, time_stamps)
        
        logger.info(f"Loaded data shape: {data_array.shape}")
        return data_array
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using MinMaxScaler per sample"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    
    def _interpolate_data(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate data to target length"""
        channels, timestamps = data.shape[1], data.shape[0]
        x_old = np.linspace(0, 1, timestamps)
        x_new = np.linspace(0, 1, target_length)
        
        interpolated = np.zeros((target_length, channels))
        for i in range(channels):
            interpolated[:, i] = np.interp(x_new, x_old, data[:, i])
            
        return interpolated
    
    def create_dataloader(self, data: np.ndarray, batch_size: int, 
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader from numpy array"""
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def prepare_amp_data(self) -> Dict[str, DataLoader]:
        """
        Prepare amplitude (motor current) data - Channel 0
        Background.md: High precision (0.94), lower recall (0.62)
        """
        logger.info("Preparing amplitude (motor current) data")
        
        # Load training data
        norm_train_data = self.load_channel_data(
            self.data_paths['norm_train'], 
            self.config.data.amp_channels, 
            is_normal=True
        )
        abnorm_train_data = self.load_channel_data(
            self.data_paths['abnorm_train'], 
            self.config.data.amp_channels, 
            is_normal=False
        )
        
        # Load test data
        norm_test_data = self.load_channel_data(
            self.data_paths['norm_test'], 
            self.config.data.amp_channels, 
            is_normal=True
        )
        abnorm_test_data = self.load_channel_data(
            self.data_paths['abnorm_test'], 
            self.config.data.amp_channels, 
            is_normal=False
        )
        
        # Create data loaders
        dataloaders = {
            'norm_train': self.create_dataloader(
                norm_train_data, 
                self.config.model.batch_size_normal, 
                shuffle=True
            ),
            'abnorm_train': self.create_dataloader(
                abnorm_train_data, 
                self.config.model.batch_size_abnormal, 
                shuffle=True
            ),
            'norm_test': self.create_dataloader(
                norm_test_data, 
                batch_size=1, 
                shuffle=False
            ),
            'abnorm_test': self.create_dataloader(
                abnorm_test_data, 
                batch_size=1, 
                shuffle=False
            ),
        }
        
        return dataloaders
    
    def prepare_vib_data(self) -> Dict[str, DataLoader]:
        """
        Prepare vibration data - Channels 1,2,3 (door vibration XYZ)
        Background.md: Higher recall (0.84), lower precision (0.82)
        """
        logger.info("Preparing vibration (door XYZ) data")
        
        # Load training data
        norm_train_data = self.load_channel_data(
            self.data_paths['norm_train'], 
            self.config.data.vib_channels, 
            is_normal=True
        )
        abnorm_train_data = self.load_channel_data(
            self.data_paths['abnorm_train'], 
            self.config.data.vib_channels, 
            is_normal=False
        )
        
        # Load test data
        norm_test_data = self.load_channel_data(
            self.data_paths['norm_test'], 
            self.config.data.vib_channels, 
            is_normal=True
        )
        abnorm_test_data = self.load_channel_data(
            self.data_paths['abnorm_test'], 
            self.config.data.vib_channels, 
            is_normal=False
        )
        
        # Create data loaders
        dataloaders = {
            'norm_train': self.create_dataloader(
                norm_train_data, 
                self.config.model.batch_size_normal, 
                shuffle=True
            ),
            'abnorm_train': self.create_dataloader(
                abnorm_train_data, 
                self.config.model.batch_size_abnormal, 
                shuffle=True
            ),
            'norm_test': self.create_dataloader(
                norm_test_data, 
                batch_size=1, 
                shuffle=False
            ),
            'abnorm_test': self.create_dataloader(
                abnorm_test_data, 
                batch_size=1, 
                shuffle=False
            ),
        }
        
        return dataloaders
