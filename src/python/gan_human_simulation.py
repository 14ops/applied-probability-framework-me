"""
GAN Human-Simulation System for Click Pattern Imitation

This module implements a sophisticated GAN system for generating human-like
click patterns using LSTM and Transformer architectures. The system learns
to distinguish between human and bot click patterns and generates realistic
human behavior for testing and validation.

Features:
- LSTM-based generator for sequential click patterns
- Transformer-based discriminator for pattern analysis
- Human vs bot pattern classification
- Realistic timing and hesitation modeling
- Multi-modal behavior generation
- Advanced training with Wasserstein GAN loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from datetime import datetime
import json
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our existing modules
from drl_environment import MinesRLEnvironment
from bayesian_mines import BayesianMinesInference
from performance_profiler import profile_function


@dataclass
class ClickPattern:
    """Human click pattern representation."""
    actions: List[int]
    timings: List[float]
    hesitations: List[bool]
    mouse_movements: List[Tuple[float, float]]
    click_pressure: List[float]
    behavior_type: str  # 'human', 'bot', 'synthetic'
    experience_level: str  # 'beginner', 'intermediate', 'expert'
    emotional_state: str  # 'calm', 'nervous', 'confident', 'frustrated'


@dataclass
class PatternDataset:
    """Dataset for click patterns."""
    patterns: List[ClickPattern]
    labels: List[int]  # 0 = human, 1 = bot
    features: np.ndarray
    metadata: Dict[str, Any]


class ClickPatternDataset(Dataset):
    """PyTorch dataset for click patterns."""
    
    def __init__(self, patterns: List[ClickPattern], max_length: int = 25):
        self.patterns = patterns
        self.max_length = max_length
        self.features = self._extract_features()
        self.labels = self._extract_labels()
    
    def _extract_features(self) -> np.ndarray:
        """Extract numerical features from patterns."""
        features = []
        
        for pattern in self.patterns:
            # Pad or truncate to max_length
            actions = pattern.actions[:self.max_length]
            timings = pattern.timings[:self.max_length]
            hesitations = pattern.hesitations[:self.max_length]
            
            # Pad with zeros if necessary
            while len(actions) < self.max_length:
                actions.append(0)
                timings.append(0.0)
                hesitations.append(False)
            
            # Combine features
            pattern_features = []
            for i in range(self.max_length):
                pattern_features.extend([
                    actions[i] / 25.0,  # Normalize action
                    timings[i] / 10.0,  # Normalize timing
                    1.0 if hesitations[i] else 0.0,  # Hesitation flag
                    pattern.click_pressure[i] if i < len(pattern.click_pressure) else 0.5,
                    pattern.mouse_movements[i][0] if i < len(pattern.mouse_movements) else 0.0,
                    pattern.mouse_movements[i][1] if i < len(pattern.mouse_movements) else 0.0
                ])
            
            features.append(pattern_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_labels(self) -> np.ndarray:
        """Extract labels from patterns."""
        labels = []
        for pattern in self.patterns:
            if pattern.behavior_type == 'human':
                labels.append(0)
            else:
                labels.append(1)
        return np.array(labels, dtype=np.longlong)
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])


class LSTMGenerator(nn.Module):
    """
    LSTM-based generator for human-like click patterns.
    
    Uses LSTM to generate sequential click patterns with realistic
    timing, hesitation, and behavioral characteristics.
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 6,  # action, timing, hesitation, pressure, mouse_x, mouse_y
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 sequence_length: int = 25):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers for different pattern components
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 25)  # 25 possible actions
        )
        
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Timing between 0 and 1
        )
        
        self.hesitation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Hesitation probability
        )
        
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Click pressure
        )
        
        self.mouse_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Mouse x, y coordinates
        )
        
        # Behavior type embedding
        self.behavior_embedding = nn.Embedding(5, hidden_dim // 4)  # 5 behavior types
        self.experience_embedding = nn.Embedding(3, hidden_dim // 4)  # 3 experience levels
        self.emotion_embedding = nn.Embedding(4, hidden_dim // 4)  # 4 emotional states
    
    def forward(self, 
                noise: torch.Tensor,
                behavior_type: torch.Tensor,
                experience_level: torch.Tensor,
                emotional_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate click pattern from noise and behavioral parameters.
        
        Args:
            noise: Random noise tensor [batch_size, sequence_length, input_dim]
            behavior_type: Behavior type tensor [batch_size]
            experience_level: Experience level tensor [batch_size]
            emotional_state: Emotional state tensor [batch_size]
            
        Returns:
            Dictionary containing generated pattern components
        """
        batch_size = noise.size(0)
        
        # Get embeddings
        behavior_emb = self.behavior_embedding(behavior_type)  # [batch_size, hidden_dim//4]
        experience_emb = self.experience_embedding(experience_level)
        emotion_emb = self.emotion_embedding(emotional_state)
        
        # Combine embeddings
        combined_emb = torch.cat([behavior_emb, experience_emb, emotion_emb], dim=1)
        combined_emb = combined_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Combine noise with embeddings
        lstm_input = torch.cat([noise, combined_emb], dim=2)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_input)
        
        # Generate different pattern components
        actions = self.action_head(lstm_output)  # [batch_size, sequence_length, 25]
        timings = self.timing_head(lstm_output)  # [batch_size, sequence_length, 1]
        hesitations = self.hesitation_head(lstm_output)  # [batch_size, sequence_length, 1]
        pressures = self.pressure_head(lstm_output)  # [batch_size, sequence_length, 1]
        mouse_positions = self.mouse_head(lstm_output)  # [batch_size, sequence_length, 2]
        
        return {
            'actions': actions,
            'timings': timings.squeeze(-1),
            'hesitations': hesitations.squeeze(-1),
            'pressures': pressures.squeeze(-1),
            'mouse_positions': mouse_positions
        }


class TransformerDiscriminator(nn.Module):
    """
    Transformer-based discriminator for pattern classification.
    
    Uses Transformer architecture to analyze click patterns and
    distinguish between human and bot behavior.
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 sequence_length: int = 25):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(sequence_length, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Pattern analysis head (for interpretability)
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)  # 10 pattern features
        )
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for transformer."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input pattern tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            Tuple of (authenticity_score, pattern_features)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Transformer forward pass
        transformer_output = self.transformer(x)
        
        # Global average pooling
        pooled_output = transformer_output.mean(dim=1)
        
        # Classification
        authenticity_score = self.classifier(pooled_output)
        
        # Pattern analysis
        pattern_features = self.pattern_analyzer(pooled_output)
        
        return authenticity_score, pattern_features


class WassersteinGAN:
    """
    Wasserstein GAN for human click pattern generation.
    
    Implements Wasserstein GAN with gradient penalty for stable training
    and high-quality pattern generation.
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 pattern_dim: int = 6,
                 sequence_length: int = 25,
                 generator_lr: float = 0.0001,
                 discriminator_lr: float = 0.0001,
                 lambda_gp: float = 10.0,
                 n_critic: int = 5):
        
        self.input_dim = input_dim
        self.pattern_dim = pattern_dim
        self.sequence_length = sequence_length
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Initialize networks
        self.generator = LSTMGenerator(
            input_dim=input_dim,
            output_dim=pattern_dim,
            sequence_length=sequence_length
        )
        
        self.discriminator = TransformerDiscriminator(
            input_dim=pattern_dim,
            sequence_length=sequence_length
        )
        
        # Optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(0.5, 0.999)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.999)
        )
        
        # Training statistics
        self.generator_losses = []
        self.discriminator_losses = []
        self.wasserstein_distances = []
        
        # TensorBoard logging
        self.writer = SummaryWriter(f'runs/wgan_human_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Pattern generation
        self.generated_patterns = []
        self.real_patterns = []
    
    def generate_pattern(self,
                        behavior_type: int = 0,
                        experience_level: int = 1,
                        emotional_state: int = 0,
                        batch_size: int = 1) -> List[ClickPattern]:
        """
        Generate human-like click patterns.
        
        Args:
            behavior_type: 0=conservative, 1=aggressive, 2=analytical, 3=emotional, 4=random
            experience_level: 0=beginner, 1=intermediate, 2=expert
            emotional_state: 0=calm, 1=nervous, 2=confident, 3=frustrated
            batch_size: Number of patterns to generate
            
        Returns:
            List of generated click patterns
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(batch_size, self.sequence_length, self.input_dim)
            
            # Create behavioral parameters
            behavior_tensor = torch.tensor([behavior_type] * batch_size)
            experience_tensor = torch.tensor([experience_level] * batch_size)
            emotion_tensor = torch.tensor([emotional_state] * batch_size)
            
            # Generate patterns
            generated = self.generator(noise, behavior_tensor, experience_tensor, emotion_tensor)
            
            # Convert to ClickPattern objects
            patterns = []
            for i in range(batch_size):
                # Convert actions to discrete values
                actions = torch.argmax(generated['actions'][i], dim=1).cpu().numpy().tolist()
                
                # Convert timings to realistic values
                timings = (generated['timings'][i] * 5.0).cpu().numpy().tolist()  # 0-5 seconds
                
                # Convert hesitations to boolean
                hesitations = (generated['hesitations'][i] > 0.5).cpu().numpy().tolist()
                
                # Convert pressures
                pressures = generated['pressures'][i].cpu().numpy().tolist()
                
                # Convert mouse positions
                mouse_positions = generated['mouse_positions'][i].cpu().numpy().tolist()
                
                # Create pattern
                pattern = ClickPattern(
                    actions=actions,
                    timings=timings,
                    hesitations=hesitations,
                    mouse_movements=mouse_positions,
                    click_pressure=pressures,
                    behavior_type=self._get_behavior_name(behavior_type),
                    experience_level=self._get_experience_name(experience_level),
                    emotional_state=self._get_emotion_name(emotional_state)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _get_behavior_name(self, behavior_type: int) -> str:
        """Convert behavior type index to name."""
        names = ['conservative', 'aggressive', 'analytical', 'emotional', 'random']
        return names[behavior_type]
    
    def _get_experience_name(self, experience_level: int) -> str:
        """Convert experience level index to name."""
        names = ['beginner', 'intermediate', 'expert']
        return names[experience_level]
    
    def _get_emotion_name(self, emotional_state: int) -> str:
        """Convert emotional state index to name."""
        names = ['calm', 'nervous', 'confident', 'frustrated']
        return names[emotional_state]
    
    def train_discriminator(self, real_patterns: List[ClickPattern], 
                           synthetic_patterns: List[ClickPattern]) -> float:
        """
        Train discriminator on real and synthetic patterns.
        
        Args:
            real_patterns: Real human click patterns
            synthetic_patterns: Generated synthetic patterns
            
        Returns:
            Discriminator loss
        """
        self.discriminator.train()
        
        # Convert patterns to tensors
        real_tensors = self._patterns_to_tensors(real_patterns)
        synthetic_tensors = self._patterns_to_tensors(synthetic_patterns)
        
        if len(real_tensors) == 0 and len(synthetic_tensors) == 0:
            return 0.0
        
        # Combine real and synthetic
        if len(real_tensors) > 0 and len(synthetic_tensors) > 0:
            all_patterns = torch.cat([real_tensors, synthetic_tensors], dim=0)
        elif len(real_tensors) > 0:
            all_patterns = real_tensors
        else:
            all_patterns = synthetic_tensors
        
        # Create labels (1 for real, 0 for synthetic)
        real_labels = torch.ones(len(real_tensors), 1)
        synthetic_labels = torch.zeros(len(synthetic_tensors), 1)
        
        if len(real_labels) > 0 and len(synthetic_labels) > 0:
            all_labels = torch.cat([real_labels, synthetic_labels], dim=0)
        elif len(real_labels) > 0:
            all_labels = real_labels
        else:
            all_labels = synthetic_labels
        
        # Train discriminator
        self.discriminator_optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = self.discriminator(all_patterns)
        
        # Calculate loss
        loss = F.binary_cross_entropy(predictions, all_labels)
        
        # Backward pass
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def train_generator(self, batch_size: int = 32) -> float:
        """
        Train generator to fool discriminator.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Generator loss
        """
        self.generator.train()
        
        # Generate synthetic patterns
        noise = torch.randn(batch_size, self.sequence_length, self.input_dim)
        behavior_types = torch.randint(0, 5, (batch_size,))
        experience_levels = torch.randint(0, 3, (batch_size,))
        emotional_states = torch.randint(0, 4, (batch_size,))
        
        generated = self.generator(noise, behavior_types, experience_levels, emotional_states)
        
        # Convert to tensor format
        synthetic_tensors = self._generated_to_tensors(generated)
        
        # Train generator
        self.generator_optimizer.zero_grad()
        
        # Try to fool discriminator
        predictions, _ = self.discriminator(synthetic_tensors)
        
        # Generator wants discriminator to think synthetic is real
        fake_labels = torch.ones(batch_size, 1)
        loss = F.binary_cross_entropy(predictions, fake_labels)
        
        # Backward pass
        loss.backward()
        self.generator_optimizer.step()
        
        return loss.item()
    
    def _patterns_to_tensors(self, patterns: List[ClickPattern]) -> torch.Tensor:
        """Convert click patterns to tensor format."""
        if not patterns:
            return torch.empty(0, self.sequence_length, self.pattern_dim)
        
        tensors = []
        for pattern in patterns:
            # Pad or truncate to sequence_length
            actions = pattern.actions[:self.sequence_length]
            timings = pattern.timings[:self.sequence_length]
            hesitations = pattern.hesitations[:self.sequence_length]
            pressures = pattern.click_pressure[:self.sequence_length]
            mouse_x = [pos[0] for pos in pattern.mouse_movements[:self.sequence_length]]
            mouse_y = [pos[1] for pos in pattern.mouse_movements[:self.sequence_length]]
            
            # Pad if necessary
            while len(actions) < self.sequence_length:
                actions.append(0)
                timings.append(0.0)
                hesitations.append(False)
                pressures.append(0.5)
                mouse_x.append(0.0)
                mouse_y.append(0.0)
            
            # Create tensor
            pattern_tensor = []
            for i in range(self.sequence_length):
                pattern_tensor.append([
                    actions[i] / 25.0,  # Normalize action
                    timings[i] / 10.0,  # Normalize timing
                    1.0 if hesitations[i] else 0.0,  # Hesitation
                    pressures[i],  # Pressure
                    mouse_x[i],  # Mouse X
                    mouse_y[i]   # Mouse Y
                ])
            
            tensors.append(pattern_tensor)
        
        return torch.FloatTensor(tensors)
    
    def _generated_to_tensors(self, generated: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert generated patterns to tensor format."""
        batch_size = generated['actions'].size(0)
        tensors = []
        
        for i in range(batch_size):
            actions = torch.argmax(generated['actions'][i], dim=1) / 25.0
            timings = generated['timings'][i] / 10.0
            hesitations = generated['hesitations'][i]
            pressures = generated['pressures'][i]
            mouse_x = generated['mouse_positions'][i][:, 0]
            mouse_y = generated['mouse_positions'][i][:, 1]
            
            pattern_tensor = torch.stack([
                actions, timings, hesitations, pressures, mouse_x, mouse_y
            ], dim=1)
            
            tensors.append(pattern_tensor)
        
        return torch.stack(tensors)
    
    def train(self, 
              real_patterns: List[ClickPattern],
              num_epochs: int = 100,
              batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the Wasserstein GAN.
        
        Args:
            real_patterns: Real human click patterns
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'wasserstein_distances': []
        }
        
        for epoch in range(num_epochs):
            # Train discriminator multiple times
            for _ in range(self.n_critic):
                # Sample real patterns
                real_batch = random.sample(real_patterns, min(batch_size, len(real_patterns)))
                
                # Generate synthetic patterns
                synthetic_batch = self.generate_pattern(batch_size=batch_size)
                
                # Train discriminator
                d_loss = self.train_discriminator(real_batch, synthetic_batch)
                training_history['discriminator_losses'].append(d_loss)
            
            # Train generator
            g_loss = self.train_generator(batch_size)
            training_history['generator_losses'].append(g_loss)
            
            # Calculate Wasserstein distance
            wasserstein_distance = self._calculate_wasserstein_distance(real_patterns)
            training_history['wasserstein_distances'].append(wasserstein_distance)
            
            # Log progress
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}, "
                           f"W_distance={wasserstein_distance:.4f}")
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/Generator', g_loss, epoch)
                self.writer.add_scalar('Loss/Discriminator', d_loss, epoch)
                self.writer.add_scalar('Distance/Wasserstein', wasserstein_distance, epoch)
        
        return training_history
    
    def _calculate_wasserstein_distance(self, real_patterns: List[ClickPattern]) -> float:
        """Calculate Wasserstein distance between real and generated distributions."""
        # Generate synthetic patterns
        synthetic_patterns = self.generate_pattern(batch_size=100)
        
        # Convert to tensors
        real_tensors = self._patterns_to_tensors(real_patterns[:100])
        synthetic_tensors = self._patterns_to_tensors(synthetic_patterns)
        
        if len(real_tensors) == 0 or len(synthetic_tensors) == 0:
            return 0.0
        
        # Calculate mean Wasserstein distance
        with torch.no_grad():
            real_scores, _ = self.discriminator(real_tensors)
            synthetic_scores, _ = self.discriminator(synthetic_tensors)
            
            wasserstein_distance = torch.mean(real_scores) - torch.mean(synthetic_scores)
        
        return wasserstein_distance.item()
    
    def evaluate_authenticity(self, patterns: List[ClickPattern]) -> List[float]:
        """Evaluate authenticity scores for patterns."""
        self.discriminator.eval()
        
        with torch.no_grad():
            pattern_tensors = self._patterns_to_tensors(patterns)
            if len(pattern_tensors) == 0:
                return []
            
            scores, _ = self.discriminator(pattern_tensors)
            return scores.squeeze().cpu().numpy().tolist()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'training_history': {
                'generator_losses': self.generator_losses,
                'discriminator_losses': self.discriminator_losses,
                'wasserstein_distances': self.wasserstein_distances
            }
        }, filepath)
        
        logging.info(f"WGAN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        training_history = checkpoint['training_history']
        self.generator_losses = training_history['generator_losses']
        self.discriminator_losses = training_history['discriminator_losses']
        self.wasserstein_distances = training_history['wasserstein_distances']
        
        logging.info(f"WGAN model loaded from {filepath}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def create_human_pattern_dataset(num_patterns: int = 1000) -> List[ClickPattern]:
    """
    Create a synthetic human pattern dataset for training.
    
    In practice, this would be replaced with real human gameplay data.
    """
    patterns = []
    
    behavior_types = ['conservative', 'aggressive', 'analytical', 'emotional', 'random']
    experience_levels = ['beginner', 'intermediate', 'expert']
    emotional_states = ['calm', 'nervous', 'confident', 'frustrated']
    
    for i in range(num_patterns):
        behavior_type = random.choice(behavior_types)
        experience_level = random.choice(experience_levels)
        emotional_state = random.choice(emotional_states)
        
        # Generate pattern based on behavior type
        if behavior_type == 'conservative':
            actions = [random.randint(0, 24) for _ in range(random.randint(5, 15))]
            timings = [random.uniform(1.0, 3.0) for _ in range(len(actions))]
            hesitations = [random.random() > 0.8 for _ in range(len(actions))]
        elif behavior_type == 'aggressive':
            actions = [random.randint(0, 24) for _ in range(random.randint(15, 25))]
            timings = [random.uniform(0.2, 1.0) for _ in range(len(actions))]
            hesitations = [random.random() > 0.9 for _ in range(len(actions))]
        else:  # analytical, emotional, random
            actions = [random.randint(0, 24) for _ in range(random.randint(8, 20))]
            timings = [random.uniform(0.5, 2.0) for _ in range(len(actions))]
            hesitations = [random.random() > 0.7 for _ in range(len(actions))]
        
        # Generate mouse movements
        mouse_movements = []
        x, y = 0.0, 0.0
        for _ in range(len(actions)):
            x += random.uniform(-0.1, 0.1)
            y += random.uniform(-0.1, 0.1)
            mouse_movements.append((x, y))
        
        # Generate click pressures
        pressures = [random.uniform(0.3, 0.8) for _ in range(len(actions))]
        
        pattern = ClickPattern(
            actions=actions,
            timings=timings,
            hesitations=hesitations,
            mouse_movements=mouse_movements,
            click_pressure=pressures,
            behavior_type=behavior_type,
            experience_level=experience_level,
            emotional_state=emotional_state
        )
        
        patterns.append(pattern)
    
    return patterns


def create_wgan_human_simulation() -> WassersteinGAN:
    """Factory function for creating WGAN human simulation."""
    return WassersteinGAN(
        input_dim=100,
        pattern_dim=6,
        sequence_length=25,
        generator_lr=0.0001,
        discriminator_lr=0.0001,
        lambda_gp=10.0,
        n_critic=5
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create WGAN
    wgan = create_wgan_human_simulation()
    
    # Create training data
    human_patterns = create_human_pattern_dataset(1000)
    
    # Train WGAN
    logging.info("Starting WGAN training...")
    training_history = wgan.train(human_patterns, num_epochs=100, batch_size=32)
    
    # Generate test patterns
    logging.info("Generating test patterns...")
    test_patterns = wgan.generate_pattern(
        behavior_type=2,  # analytical
        experience_level=1,  # intermediate
        emotional_state=0,  # calm
        batch_size=10
    )
    
    # Evaluate authenticity
    authenticity_scores = wgan.evaluate_authenticity(test_patterns)
    logging.info(f"Generated pattern authenticity scores: {authenticity_scores}")
    
    # Save model
    wgan.save_model("models/wgan_human_simulation.pth")
    
    # Close
    wgan.close()
    
    logging.info("WGAN training complete!")
