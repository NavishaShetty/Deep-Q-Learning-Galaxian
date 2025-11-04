"""
Deep Q-Network (DQN) Agent

Implements the DQN algorithm for learning to play Atari games.

ATTRIBUTION:
- Core algorithm from DeepMind DQN paper (Mnih et al., 2013)
- Adapted for PyTorch and Gymnasium
- Includes experience replay and ε-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .neural_network import QNetwork
from .replay_buffer import ReplayBuffer
from .utils import preprocess_state, compute_epsilon_decay


class DQNAgent:
    """
    Deep Q-Network Agent for Atari games.
    
    Learns Q-values using neural network and experience replay.
    
    Key components:
    1. Q-Network: Neural network that approximates Q-values
    2. Experience Replay: Stores and samples past experiences
    3. ε-Greedy Exploration: Balances exploration and exploitation
    4. Bellman Update: Q-learning update rule
    """
    
    def __init__(self, 
                 state_size=(210, 160, 3),
                 action_size=6,
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon=1.0,
                 min_epsilon=0.01,
                 decay_rate=0.995,
                 device='cpu',
                 seed=42):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Tuple (height, width, channels) of input state
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer (alpha)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            min_epsilon: Minimum exploration rate
            decay_rate: Exploration decay rate
            device: 'cpu' or 'cuda'
            seed: Random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.device = torch.device(device)
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Neural networks
        self.q_network = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(max_size=10000, seed=seed)
        
        # Training parameters
        self.batch_size = 32
        self.update_frequency = 1  # Update network every N steps
        self.step_count = 0
    
    def act(self, state, epsilon=None, training=True):
        """
        Choose action using ε-greedy policy.
        
        Args:
            state: Current state (pixel array)
            epsilon: Exploration rate (uses self.epsilon if None)
            training: Whether in training mode
        
        Returns:
            Action to take
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Exploration: random action
        if training and np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        # Exploitation: best known action
        state_tensor = torch.from_numpy(preprocess_state(state)).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)  # Reorder to (batch, channels, height, width)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size=None):
        """
        Train network using experience replay.
        
        Samples random batch from replay buffer and performs Bellman update.
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Loss value if training performed, None otherwise
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Check if enough experiences in memory
        if not self.memory.is_ready(batch_size):
            return None
        
        # Sample random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Reorder dimensions for CNN (batch, channels, height, width)
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        
        # Compute Q-values for actions taken
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Bellman equation
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            
            # Q-target = reward + gamma * max(Q(s', a'))  if not done
            # Q-target = reward                            if done
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones)
        
        # Compute loss and update network
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.step_count += 1
        
        return loss.item()
    
    def decay_exploration(self, episode):
        """
        Decay exploration rate.
        
        Args:
            episode: Current episode number
        """
        self.epsilon = compute_epsilon_decay(
            initial_epsilon=1.0,
            min_epsilon=self.min_epsilon,
            decay_rate=self.decay_rate,
            episode=episode
        )
    
    def save_model(self, filepath):
        """
        Save network weights to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load network weights from file.
        
        Args:
            filepath: Path to load model from
        """
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from: {filepath}")
    
    def set_learning_rate(self, learning_rate):
        """
        Update learning rate.
        
        Args:
            learning_rate: New learning rate
        """
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    def set_gamma(self, gamma):
        """
        Update discount factor.
        
        Args:
            gamma: New discount factor
        """
        self.gamma = gamma
