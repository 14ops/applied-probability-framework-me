"""
Adversarial GAN for Human-like Behavior Simulation

This module implements a Generative Adversarial Network (GAN) system designed
to generate human-like playing patterns for the Applied Probability Framework.
The GAN learns from human gameplay data to create realistic behavioral patterns
that can be used for testing, training, and adversarial evaluation.

Features:
- Generator network for human-like action sequences
- Discriminator network for pattern authenticity detection
- Behavioral pattern analysis and synthesis
- Multi-modal human behavior modeling
- Adversarial training with strategy-specific patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
import random
from datetime import datetime

from drl_environment import MinesRLEnvironment, create_rl_environment


@dataclass
class HumanBehaviorPattern:
    """Human behavior pattern representation."""
    action_sequence: List[int]
    timing_pattern: List[float]
    hesitation_markers: List[bool]
    risk_preferences: List[float]
    emotional_state: str
    experience_level: str


@dataclass
class GANTrainingData:
    """Training data for GAN."""
    real_patterns: List[HumanBehaviorPattern]
    synthetic_patterns: List[HumanBehaviorPattern]
    discriminator_labels: List[bool]
    generator_losses: List[float]
    discriminator_losses: List[float]


class HumanBehaviorGenerator(nn.Module):
    """
    Generator network for creating human-like behavior patterns.
    
    This network learns to generate realistic human playing patterns including:
    - Action sequences with human-like timing
    - Hesitation patterns and decision delays
    - Risk preference variations
    - Emotional state influences
    """
    
    def __init__(self, 
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 50,
                 sequence_length: int = 25,
                 num_behavior_types: int = 5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.num_behavior_types = num_behavior_types
        
        # Encoder for context and behavior type
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Behavior type embedding
        self.behavior_embedding = nn.Embedding(num_behavior_types, hidden_dim // 4)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2 + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Action sequence generator
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Timing pattern generator
        self.timing_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, sequence_length)
        )
        
        # Hesitation pattern generator
        self.hesitation_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, sequence_length),
            nn.Sigmoid()
        )
        
        # Risk preference generator
        self.risk_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, sequence_length),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor, behavior_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate human-like behavior pattern.
        
        Args:
            context: Context tensor [batch_size, input_dim]
            behavior_type: Behavior type tensor [batch_size]
            
        Returns:
            Dictionary containing generated behavior components
        """
        batch_size = context.size(0)
        
        # Encode context
        context_encoded = self.context_encoder(context)
        
        # Get behavior type embedding
        behavior_embedding = self.behavior_embedding(behavior_type)
        
        # Combine context and behavior type
        combined_input = torch.cat([context_encoded, behavior_embedding], dim=1)
        
        # Expand for sequence generation
        sequence_input = combined_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Generate sequence through LSTM
        lstm_output, _ = self.lstm(sequence_input)
        
        # Generate different behavior components
        actions = self.action_generator(lstm_output)
        timing = self.timing_generator(lstm_output)
        hesitation = self.hesitation_generator(lstm_output)
        risk_preferences = self.risk_generator(lstm_output)
        
        return {
            'actions': actions,
            'timing': timing,
            'hesitation': hesitation,
            'risk_preferences': risk_preferences
        }


class HumanBehaviorDiscriminator(nn.Module):
    """
    Discriminator network for detecting human vs synthetic behavior patterns.
    
    This network learns to distinguish between real human behavior patterns
    and synthetic patterns generated by the generator.
    """
    
    def __init__(self, 
                 input_dim: int = 50,
                 sequence_length: int = 25,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for pattern recognition
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM for temporal pattern analysis
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, behavior_pattern: torch.Tensor) -> torch.Tensor:
        """
        Classify behavior pattern as real or synthetic.
        
        Args:
            behavior_pattern: Behavior pattern tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            Probability of being real human behavior
        """
        batch_size = behavior_pattern.size(0)
        
        # Convolutional analysis
        conv_input = behavior_pattern.transpose(1, 2)  # [batch, input_dim, sequence_length]
        conv_features = self.conv_layers(conv_input).squeeze(-1)  # [batch, 256]
        
        # LSTM analysis
        lstm_output, (hidden, _) = self.lstm(behavior_pattern)
        lstm_features = hidden[-1]  # [batch, hidden_dim]
        
        # Combine features
        combined_features = torch.cat([conv_features, lstm_features], dim=1)
        
        # Classification
        authenticity_score = self.classifier(combined_features)
        
        return authenticity_score


class AdversarialGAN:
    """
    Adversarial GAN system for human-like behavior generation.
    
    This class coordinates the training of the generator and discriminator
    networks to create realistic human behavior patterns.
    """
    
    def __init__(self, 
                 input_dim: int = 100,
                 output_dim: int = 50,
                 sequence_length: int = 25,
                 num_behavior_types: int = 5,
                 learning_rate: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.num_behavior_types = num_behavior_types
        
        # Initialize networks
        self.generator = HumanBehaviorGenerator(
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            num_behavior_types=num_behavior_types
        )
        
        self.discriminator = HumanBehaviorDiscriminator(
            input_dim=output_dim,
            sequence_length=sequence_length
        )
        
        # Optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training data
        self.training_data = GANTrainingData(
            real_patterns=[],
            synthetic_patterns=[],
            discriminator_labels=[],
            generator_losses=[],
            discriminator_losses=[]
        )
        
        # Behavior type mapping
        self.behavior_types = {
            0: 'conservative',
            1: 'aggressive', 
            2: 'analytical',
            3: 'emotional',
            4: 'random'
        }
    
    def generate_behavior_pattern(self, 
                                 context: torch.Tensor,
                                 behavior_type: Optional[int] = None) -> HumanBehaviorPattern:
        """
        Generate a human-like behavior pattern.
        
        Args:
            context: Context tensor
            behavior_type: Optional behavior type (0-4)
            
        Returns:
            Generated behavior pattern
        """
        self.generator.eval()
        
        if behavior_type is None:
            behavior_type = random.randint(0, self.num_behavior_types - 1)
        
        behavior_tensor = torch.tensor([behavior_type], dtype=torch.long)
        
        with torch.no_grad():
            generated = self.generator(context.unsqueeze(0), behavior_tensor)
        
        # Convert to behavior pattern
        actions = generated['actions'].squeeze(0).cpu().numpy()
        timing = generated['timing'].squeeze(0).cpu().numpy()
        hesitation = generated['hesitation'].squeeze(0).cpu().numpy()
        risk_preferences = generated['risk_preferences'].squeeze(0).cpu().numpy()
        
        # Convert to discrete actions
        action_sequence = np.argmax(actions, axis=1).tolist()
        timing_pattern = timing.tolist()
        hesitation_markers = (hesitation > 0.5).tolist()
        risk_preferences_list = risk_preferences.tolist()
        
        return HumanBehaviorPattern(
            action_sequence=action_sequence,
            timing_pattern=timing_pattern,
            hesitation_markers=hesitation_markers,
            risk_preferences=risk_preferences_list,
            emotional_state=self.behavior_types[behavior_type],
            experience_level='intermediate'
        )
    
    def train_discriminator(self, 
                           real_patterns: List[HumanBehaviorPattern],
                           synthetic_patterns: List[HumanBehaviorPattern]) -> float:
        """
        Train discriminator on real and synthetic patterns.
        
        Args:
            real_patterns: Real human behavior patterns
            synthetic_patterns: Synthetic behavior patterns
            
        Returns:
            Discriminator loss
        """
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        
        # Prepare real patterns
        real_tensors = []
        for pattern in real_patterns:
            # Convert pattern to tensor
            pattern_tensor = self._pattern_to_tensor(pattern)
            real_tensors.append(pattern_tensor)
        
        if real_tensors:
            real_batch = torch.stack(real_tensors)
            real_labels = torch.ones(real_batch.size(0), 1)
        else:
            real_batch = torch.empty(0, self.sequence_length, self.output_dim)
            real_labels = torch.empty(0, 1)
        
        # Prepare synthetic patterns
        synthetic_tensors = []
        for pattern in synthetic_patterns:
            pattern_tensor = self._pattern_to_tensor(pattern)
            synthetic_tensors.append(pattern_tensor)
        
        if synthetic_tensors:
            synthetic_batch = torch.stack(synthetic_tensors)
            synthetic_labels = torch.zeros(synthetic_batch.size(0), 1)
        else:
            synthetic_batch = torch.empty(0, self.sequence_length, self.output_dim)
            synthetic_labels = torch.empty(0, 1)
        
        # Combine batches
        if real_batch.size(0) > 0 and synthetic_batch.size(0) > 0:
            all_patterns = torch.cat([real_batch, synthetic_batch], dim=0)
            all_labels = torch.cat([real_labels, synthetic_labels], dim=0)
        elif real_batch.size(0) > 0:
            all_patterns = real_batch
            all_labels = real_labels
        elif synthetic_batch.size(0) > 0:
            all_patterns = synthetic_batch
            all_labels = synthetic_labels
        else:
            return 0.0
        
        # Forward pass
        predictions = self.discriminator(all_patterns)
        loss = self.bce_loss(predictions, all_labels)
        
        # Backward pass
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def train_generator(self, 
                       context_batch: torch.Tensor,
                       behavior_types: torch.Tensor) -> float:
        """
        Train generator to fool discriminator.
        
        Args:
            context_batch: Batch of context tensors
            behavior_types: Batch of behavior type labels
            
        Returns:
            Generator loss
        """
        self.generator.train()
        self.generator_optimizer.zero_grad()
        
        # Generate synthetic patterns
        generated = self.generator(context_batch, behavior_types)
        
        # Convert to pattern format
        synthetic_patterns = []
        for i in range(context_batch.size(0)):
            actions = generated['actions'][i].cpu().numpy()
            timing = generated['timing'][i].cpu().numpy()
            hesitation = generated['hesitation'][i].cpu().numpy()
            risk_preferences = generated['risk_preferences'][i].cpu().numpy()
            
            action_sequence = np.argmax(actions, axis=1).tolist()
            timing_pattern = timing.tolist()
            hesitation_markers = (hesitation > 0.5).tolist()
            risk_preferences_list = risk_preferences.tolist()
            
            pattern = HumanBehaviorPattern(
                action_sequence=action_sequence,
                timing_pattern=timing_pattern,
                hesitation_markers=hesitation_markers,
                risk_preferences=risk_preferences_list,
                emotional_state=self.behavior_types[behavior_types[i].item()],
                experience_level='intermediate'
            )
            synthetic_patterns.append(pattern)
        
        # Convert to tensors for discriminator
        synthetic_tensors = [self._pattern_to_tensor(p) for p in synthetic_patterns]
        synthetic_batch = torch.stack(synthetic_tensors)
        
        # Try to fool discriminator
        fake_labels = torch.ones(synthetic_batch.size(0), 1)
        predictions = self.discriminator(synthetic_batch)
        
        # Generator loss (want discriminator to think synthetic is real)
        loss = self.bce_loss(predictions, fake_labels)
        
        # Backward pass
        loss.backward()
        self.generator_optimizer.step()
        
        return loss.item()
    
    def _pattern_to_tensor(self, pattern: HumanBehaviorPattern) -> torch.Tensor:
        """Convert behavior pattern to tensor format."""
        # Create one-hot encoding for actions
        action_tensor = torch.zeros(self.sequence_length, self.output_dim)
        for i, action in enumerate(pattern.action_sequence[:self.sequence_length]):
            if i < self.sequence_length:
                action_tensor[i, action] = 1.0
        
        # Add timing, hesitation, and risk information as additional features
        # This is a simplified conversion - in practice, you'd want more sophisticated encoding
        return action_tensor
    
    def train_gan(self, 
                  real_patterns: List[HumanBehaviorPattern],
                  num_epochs: int = 100,
                  batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the GAN system.
        
        Args:
            real_patterns: Real human behavior patterns for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'generator_acc': [],
            'discriminator_acc': []
        }
        
        for epoch in range(num_epochs):
            # Shuffle real patterns
            random.shuffle(real_patterns)
            
            # Train discriminator
            d_losses = []
            for i in range(0, len(real_patterns), batch_size):
                batch_real = real_patterns[i:i + batch_size]
                
                # Generate synthetic patterns for this batch
                context_batch = torch.randn(len(batch_real), self.input_dim)
                behavior_types = torch.randint(0, self.num_behavior_types, (len(batch_real),))
                
                synthetic_patterns = []
                for j in range(len(batch_real)):
                    pattern = self.generate_behavior_pattern(context_batch[j], behavior_types[j].item())
                    synthetic_patterns.append(pattern)
                
                # Train discriminator
                d_loss = self.train_discriminator(batch_real, synthetic_patterns)
                d_losses.append(d_loss)
            
            # Train generator
            g_losses = []
            for i in range(0, len(real_patterns), batch_size):
                context_batch = torch.randn(min(batch_size, len(real_patterns) - i), self.input_dim)
                behavior_types = torch.randint(0, self.num_behavior_types, (context_batch.size(0),))
                
                g_loss = self.train_generator(context_batch, behavior_types)
                g_losses.append(g_loss)
            
            # Record losses
            avg_d_loss = np.mean(d_losses) if d_losses else 0.0
            avg_g_loss = np.mean(g_losses) if g_losses else 0.0
            
            training_history['generator_losses'].append(avg_g_loss)
            training_history['discriminator_losses'].append(avg_d_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
        
        return training_history
    
    def save_model(self, filepath: str):
        """Save GAN model to file."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'training_data': self.training_data
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load GAN model from file."""
        checkpoint = torch.load(filepath)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.training_data = checkpoint['training_data']


def create_human_behavior_dataset(num_samples: int = 1000) -> List[HumanBehaviorPattern]:
    """
    Create a synthetic human behavior dataset for training.
    
    In practice, this would be replaced with real human gameplay data.
    
    Args:
        num_samples: Number of behavior patterns to generate
        
    Returns:
        List of human behavior patterns
    """
    patterns = []
    
    behavior_types = ['conservative', 'aggressive', 'analytical', 'emotional', 'random']
    
    for i in range(num_samples):
        behavior_type = random.choice(behavior_types)
        
        # Generate action sequence
        sequence_length = random.randint(10, 25)
        action_sequence = [random.randint(0, 24) for _ in range(sequence_length)]
        
        # Generate timing pattern
        timing_pattern = [random.uniform(0.5, 3.0) for _ in range(sequence_length)]
        
        # Generate hesitation markers
        hesitation_markers = [random.random() > 0.7 for _ in range(sequence_length)]
        
        # Generate risk preferences
        if behavior_type == 'conservative':
            risk_preferences = [random.uniform(0.1, 0.4) for _ in range(sequence_length)]
        elif behavior_type == 'aggressive':
            risk_preferences = [random.uniform(0.6, 0.9) for _ in range(sequence_length)]
        else:
            risk_preferences = [random.uniform(0.3, 0.7) for _ in range(sequence_length)]
        
        pattern = HumanBehaviorPattern(
            action_sequence=action_sequence,
            timing_pattern=timing_pattern,
            hesitation_markers=hesitation_markers,
            risk_preferences=risk_preferences,
            emotional_state=behavior_type,
            experience_level=random.choice(['beginner', 'intermediate', 'expert'])
        )
        
        patterns.append(pattern)
    
    return patterns


def create_adversarial_gan(input_dim: int = 100,
                          output_dim: int = 50,
                          sequence_length: int = 25) -> AdversarialGAN:
    """
    Factory function for creating adversarial GAN.
    
    Args:
        input_dim: Input dimension for context
        output_dim: Output dimension for actions
        sequence_length: Length of behavior sequences
        
    Returns:
        Configured adversarial GAN
    """
    return AdversarialGAN(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        num_behavior_types=5,
        learning_rate=0.0002
    )
