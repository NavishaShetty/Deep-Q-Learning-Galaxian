"""
Utility Functions for DQN Training

Includes preprocessing, metrics tracking, and visualization helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque


class MetricsTracker:
    """Track training metrics during agent training."""
    
    def __init__(self, window_size=100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window for computing moving averages
        """
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
        self.episode_numbers = []
        
    def add_episode(self, episode_number, reward, length, epsilon):
        """
        Add metrics for completed episode.
        
        Args:
            episode_number: Episode number
            reward: Total reward for episode
            length: Number of steps in episode
            epsilon: Epsilon value used in episode
        """
        self.episode_numbers.append(episode_number)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilons.append(epsilon)
    
    def get_moving_average_reward(self):
        """Get moving average of rewards over last window."""
        if len(self.episode_rewards) < self.window_size:
            return np.mean(self.episode_rewards)
        return np.mean(self.episode_rewards[-self.window_size:])
    
    def get_moving_average_length(self):
        """Get moving average of episode lengths."""
        if len(self.episode_lengths) < self.window_size:
            return np.mean(self.episode_lengths)
        return np.mean(self.episode_lengths[-self.window_size:])
    
    def to_dataframe(self):
        """Convert metrics to pandas DataFrame."""
        df = pd.DataFrame({
            'episode': self.episode_numbers,
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'epsilon': self.epsilons,
        })
        
        # Add moving averages
        df['moving_avg_reward'] = df['reward'].rolling(window=self.window_size, min_periods=1).mean()
        df['moving_avg_length'] = df['length'].rolling(window=self.window_size, min_periods=1).mean()
        
        return df
    
    def save_to_csv(self, filepath):
        """Save metrics to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to: {filepath}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.episode_rewards:
            print("No episodes tracked yet.")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Max reward: {max(self.episode_rewards):.2f}")
        print(f"Min reward: {min(self.episode_rewards):.2f}")
        print(f"Mean reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Moving avg reward (last 100): {self.get_moving_average_reward():.2f}")
        print(f"Mean episode length: {np.mean(self.episode_lengths):.2f}")
        print("="*50 + "\n")


def preprocess_state(state):
    """
    Preprocess state for network input.
    
    Converts uint8 pixel values [0, 255] to float32 [0, 1].
    
    Args:
        state: Raw state from environment
    
    Returns:
        Preprocessed state
    """
    return np.array(state, dtype=np.float32) / 255.0


def compute_epsilon_decay(initial_epsilon, min_epsilon, decay_rate, episode):
    """
    Compute epsilon value for ε-greedy exploration.
    
    Args:
        initial_epsilon: Starting epsilon value
        min_epsilon: Minimum epsilon value
        decay_rate: Decay rate per episode
        episode: Current episode number
    
    Returns:
        Epsilon value for current episode
    """
    epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return max(epsilon, min_epsilon)


def plot_training_results(metrics_df, save_path=None):
    """
    Plot training metrics.
    
    Args:
        metrics_df: DataFrame with training metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Reward over time
    axes[0, 0].plot(metrics_df['episode'], metrics_df['reward'], alpha=0.5, label='Episode Reward')
    axes[0, 0].plot(metrics_df['episode'], metrics_df['moving_avg_reward'], 'r-', label='Moving Average (100 episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode length over time
    axes[0, 1].plot(metrics_df['episode'], metrics_df['length'], alpha=0.5)
    axes[0, 1].plot(metrics_df['episode'], metrics_df['moving_avg_length'], 'r-')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length (steps)')
    axes[0, 1].set_title('Episode Length Over Training')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    axes[1, 0].plot(metrics_df['episode'], metrics_df['epsilon'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution (histogram)
    axes[1, 1].hist(metrics_df['reward'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    
    plt.show()
