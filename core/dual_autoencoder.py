# core/dual_autoencoder.py - Dual Autoencoder implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from config import config
from models.architectures.CNN1D import CNN_Encoder, CNN_Decoder

logger = logging.getLogger(__name__)

class AutoEncoder(nn.Module):
    """
    Single Autoencoder implementation for dual-AE approach.
    Based on background.md methodology:
    - 1D-CNN architecture (not VAE, not using FFT)
    - Reconstruction loss + sparsity regularization
    - Separate models for normal and abnormal data
    """
    
    def __init__(self, input_channels: int, embedding_size: int):
        super(AutoEncoder, self).__init__()
        self.input_channels = input_channels
        self.embedding_size = embedding_size
        
        # Create encoder and decoder
        self.encoder = CNN_Encoder(
            input_size=(input_channels, config.data.time_stamps),
            embedding_size=embedding_size
        )
        self.decoder = CNN_Decoder(
            input_size=(input_channels, config.data.time_stamps),
            embedding_size=embedding_size
        )
        
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        encoded, activations = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, activations

class DualAutoEncoder:
    """
    Dual Autoencoder system implementing the methodology from background.md:
    1. Train separate AEs on normal and abnormal data
    2. Use reconstruction losses as 2D features
    3. Apply classifiers (SVM, LogReg, kNN) for final classification
    """
    
    def __init__(self, mode: str = 'amp'):
        """
        Initialize dual autoencoder system.
        
        Args:
            mode: 'amp' for amplitude/current data, 'vib' for vibration data
        """
        self.mode = mode
        self.config = config
        self.device = torch.device(self.config.system.device)
        
        # Determine input channels and embedding size based on mode
        if mode == 'amp':
            self.input_channels = len(self.config.data.amp_channels)
            self.embedding_size = self.config.model.embedding_size
        elif mode == 'vib':
            self.input_channels = len(self.config.data.vib_channels)
            self.embedding_size = self.config.model.embedding_size_vib
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Initialize autoencoders
        self.normal_ae = AutoEncoder(self.input_channels, self.embedding_size).to(self.device)
        self.abnormal_ae = AutoEncoder(self.input_channels, self.embedding_size).to(self.device)
        
        # Initialize optimizers
        self.normal_optimizer = optim.Adam(
            self.normal_ae.parameters(), 
            lr=self.config.model.lr
        )
        self.abnormal_optimizer = optim.Adam(
            self.abnormal_ae.parameters(), 
            lr=self.config.model.lr
        )
        
        # Initialize schedulers
        self.normal_scheduler = StepLR(
            self.normal_optimizer,
            step_size=self.config.model.scheduler_step_size,
            gamma=self.config.model.scheduler_gamma
        )
        self.abnormal_scheduler = StepLR(
            self.abnormal_optimizer,
            step_size=self.config.model.scheduler_step_size,
            gamma=self.config.model.scheduler_gamma
        )
        
        # Track training history
        self.train_losses = {'normal': [], 'abnormal': []}
        self.val_losses = {'normal': [], 'abnormal': []}
        
    def compute_loss(self, reconstructed: torch.Tensor, original: torch.Tensor, 
                    activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss with sparsity regularization.
        Based on background.md: focus on reconstruction error for anomaly detection.
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(
            reconstructed.reshape(-1, self.input_channels * self.config.data.time_stamps),
            original.reshape(-1, self.input_channels * self.config.data.time_stamps),
            reduction='sum'
        )
        
        # Sparsity loss
        sparsity_loss = sum(
            torch.mean(torch.abs(activation - self.config.model.sparsity_param)) 
            for activation in activations
        )
        
        total_loss = recon_loss + self.config.model.sparsity_weight * sparsity_loss
        return total_loss
        
    def train_autoencoder(self, autoencoder: AutoEncoder, optimizer: optim.Optimizer,
                         scheduler: StepLR, dataloader: torch.utils.data.DataLoader,
                         val_dataloader: torch.utils.data.DataLoader = None,
                         ae_type: str = 'normal') -> None:
        """Train a single autoencoder"""
        
        logger.info(f"Training {ae_type} autoencoder for {self.mode} mode")
        
        for epoch in range(1, self.config.model.epochs + 1):
            # Training phase
            autoencoder.train()
            epoch_train_loss = 0.0
            
            with tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.model.epochs}") as pbar:
                for batch_data in pbar:
                    data = batch_data[0].to(self.device)
                    
                    optimizer.zero_grad()
                    reconstructed, activations = autoencoder(data)
                    loss = self.compute_loss(reconstructed, data, activations)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item() / len(data)})
            
            # Average training loss
            avg_train_loss = epoch_train_loss / len(dataloader.dataset)
            self.train_losses[ae_type].append(avg_train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                autoencoder.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for batch_data in val_dataloader:
                        data = batch_data[0].to(self.device)
                        reconstructed, activations = autoencoder(data)
                        loss = self.compute_loss(reconstructed, data, activations)
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_dataloader.dataset)
                self.val_losses[ae_type].append(avg_val_loss)
                
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}")
            
            scheduler.step()
    
    def train_dual_system(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Train both normal and abnormal autoencoders.
        
        Args:
            dataloaders: Dictionary with keys 'norm_train', 'abnorm_train', 
                        'norm_test', 'abnorm_test'
        """
        logger.info(f"Starting dual autoencoder training for {self.mode} mode")
        
        # Train normal autoencoder
        self.train_autoencoder(
            self.normal_ae,
            self.normal_optimizer,
            self.normal_scheduler,
            dataloaders['norm_train'],
            dataloaders.get('norm_test'),
            'normal'
        )
        
        # Train abnormal autoencoder  
        self.train_autoencoder(
            self.abnormal_ae,
            self.abnormal_optimizer,
            self.abnormal_scheduler,
            dataloaders['abnorm_train'],
            dataloaders.get('abnorm_test'),
            'abnormal'
        )
        
        logger.info(f"Dual autoencoder training completed for {self.mode} mode")
    
    def extract_features(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D features (normal_loss, abnormal_loss) for classification.
        This implements the core methodology from background.md.
        """
        logger.info(f"Extracting 2D features for {self.mode} mode")
        
        self.normal_ae.eval()
        self.abnormal_ae.eval()
        
        normal_losses = []
        abnormal_losses = []
        labels = []
        
        with torch.no_grad():
            # Process normal test data
            for batch_data in dataloaders['norm_test']:
                data = batch_data[0].to(self.device)
                
                # Get reconstruction losses from both AEs
                normal_recon, normal_activations = self.normal_ae(data)
                abnormal_recon, abnormal_activations = self.abnormal_ae(data)
                
                normal_loss = self.compute_loss(normal_recon, data, normal_activations).item()
                abnormal_loss = self.compute_loss(abnormal_recon, data, abnormal_activations).item()
                
                normal_losses.append(normal_loss)
                abnormal_losses.append(abnormal_loss)
                labels.extend([0] * len(data))  # Normal = 0
            
            # Process abnormal test data
            for batch_data in dataloaders['abnorm_test']:
                data = batch_data[0].to(self.device)
                
                # Get reconstruction losses from both AEs
                normal_recon, normal_activations = self.normal_ae(data)
                abnormal_recon, abnormal_activations = self.abnormal_ae(data)
                
                normal_loss = self.compute_loss(normal_recon, data, normal_activations).item()
                abnormal_loss = self.compute_loss(abnormal_recon, data, abnormal_activations).item()
                
                normal_losses.append(normal_loss)
                abnormal_losses.append(abnormal_loss)
                labels.extend([1] * len(data))  # Abnormal = 1
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        features = np.column_stack([normal_losses, abnormal_losses])
        normalized_features = scaler.fit_transform(features)
        
        logger.info(f"Extracted features shape: {normalized_features.shape}")
        return normalized_features, np.array(labels)
    
    def save_models(self, model_paths: Dict[str, str]) -> None:
        """Save trained autoencoder models"""
        # Save normal AE
        torch.save({
            'model_state_dict': self.normal_ae.state_dict(),
            'optimizer_state_dict': self.normal_optimizer.state_dict(),
            'scheduler_state_dict': self.normal_scheduler.state_dict(),
            'mode': self.mode,
            'input_channels': self.input_channels,
            'embedding_size': self.embedding_size,
            'train_losses': self.train_losses['normal'],
            'val_losses': self.val_losses['normal'],
        }, model_paths['normal_ae'])
        
        # Save abnormal AE
        torch.save({
            'model_state_dict': self.abnormal_ae.state_dict(),
            'optimizer_state_dict': self.abnormal_optimizer.state_dict(),
            'scheduler_state_dict': self.abnormal_scheduler.state_dict(),
            'mode': self.mode,
            'input_channels': self.input_channels,
            'embedding_size': self.embedding_size,
            'train_losses': self.train_losses['abnormal'],
            'val_losses': self.val_losses['abnormal'],
        }, model_paths['abnormal_ae'])
        
        logger.info(f"Models saved: {model_paths}")
    
    def load_models(self, model_paths: Dict[str, str]) -> None:
        """Load trained autoencoder models"""
        # Load normal AE
        normal_checkpoint = torch.load(model_paths['normal_ae'], map_location=self.device)
        self.normal_ae.load_state_dict(normal_checkpoint['model_state_dict'])
        self.normal_optimizer.load_state_dict(normal_checkpoint['optimizer_state_dict'])
        self.normal_scheduler.load_state_dict(normal_checkpoint['scheduler_state_dict'])
        
        # Load abnormal AE
        abnormal_checkpoint = torch.load(model_paths['abnormal_ae'], map_location=self.device)
        self.abnormal_ae.load_state_dict(abnormal_checkpoint['model_state_dict'])
        self.abnormal_optimizer.load_state_dict(abnormal_checkpoint['optimizer_state_dict'])
        self.abnormal_scheduler.load_state_dict(abnormal_checkpoint['scheduler_state_dict'])
        
        logger.info(f"Models loaded: {model_paths}")
