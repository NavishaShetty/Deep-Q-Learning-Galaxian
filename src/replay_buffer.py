"""
Experience Replay Buffer for DQN Training

Stores agent experiences and samples batches for training.
Breaks correlation between consecutive experiences for stable training.

ATTRIBUTION:
- Experience replay concept from DeepMind DQN paper
- Implementation adapted for PyTorch and Gymnasium
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling agent experiences.
    
    Stores tuples of (state, action, reward, next_state, done) and provides
    random sampling for training the Q-network.
    
    Benefits:
    1. Breaks correlation between consecutive experiences
    2. Improves sample efficiency
    3. Stabilizes training by sampling from diverse experiences
    """
    
    def __init__(self, max_size=10000, seed=42):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state (observation)
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All as numpy arrays suitable for PyTorch
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        
        experiences = random.sample(self.memory, batch_size)
        
        # Separate into individual components
        states = np.array([e[0] for e in experiences], dtype=np.float32)
        actions = np.array([e[1] for e in experiences], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        next_states = np.array([e[3] for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.memory)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough experiences for sampling."""
        return len(self.memory) >= batch_size
