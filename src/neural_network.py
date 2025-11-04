"""
Q-Network Architecture for DQN Agent

Neural network that approximates Q-values for Atari games.
Input: 210x160x3 pixel images
Output: Q-values for each action

ATTRIBUTION:
- Architecture inspired by DeepMind DQN paper (Mnih et al., 2013)
- Convolutional layers process pixel input
- Fully connected layers output action values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for Atari games.
    
    Uses convolutional layers to process pixel input and outputs Q-values
    for each possible action.
    
    Architecture:
    - Input: 210x160x3 (Atari frame)
    - Conv1: 32 filters, 8x8 kernel, stride 4
    - Conv2: 64 filters, 4x4 kernel, stride 2
    - Conv3: 64 filters, 3x3 kernel, stride 1
    - FC1: 512 units
    - Output: action_size (Q-values for each action)
    """
    
    def __init__(self, state_size, action_size, seed=42):
        """
        Initialize Q-Network.
        
        Args:
            state_size: Tuple (height, width, channels) of input state
            action_size: Number of possible actions
            seed: Random seed for reproducibility
        """
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Calculate flattened size after convolutions
        self._calculate_conv_output_size(state_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
    
    def _calculate_conv_output_size(self, state_size):
        """
        Calculate the flattened output size after convolutional layers.
        
        Args:
            state_size: Tuple (height, width, channels)
        """
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, 3, state_size[0], state_size[1])
        dummy_output = self.conv1(dummy_input)
        dummy_output = self.conv2(dummy_output)
        dummy_output = self.conv3(dummy_output)
        self.conv_output_size = dummy_output.numel()
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Args:
            state: Input state (batch_size, 3, 210, 160)
        
        Returns:
            Q-values for each action (batch_size, action_size)
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values
