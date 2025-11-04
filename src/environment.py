"""
Atari Environment Wrapper

Wrapper around Gymnasium Atari environment for consistent interface.
Handles preprocessing and state management.

ATTRIBUTION:
- Uses Gymnasium API (https://gymnasium.farama.org/)
- Atari Learning Environment (ALE) wrapper
"""

import gymnasium as gym
import numpy as np


class AtariEnvironment:
    """
    Wrapper for Atari Galaxian environment from Gymnasium.
    
    Provides consistent interface for training DQN agent.
    """
    
    def __init__(self, game_name="ALE/Galaxian-v5", render_mode=None):
        """
        Initialize Atari environment.
        
        Args:
            game_name: Name of Atari game (e.g., "ALE/Galaxian-v5")
            render_mode: Optional rendering mode ('human' for display)
        """
        self.env = gym.make(game_name, render_mode=render_mode)
        self.game_name = game_name
        self.state = None
        self.info = None
    
    def reset(self):
        """
        Reset environment and return initial state.
        
        Returns:
            Initial state (numpy array of pixel values)
        """
        self.state, self.info = self.env.reset()
        return self.state
    
    def step(self, action):
        """
        Take action in environment.
        
        Args:
            action: Action to take
        
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state = next_state
        self.info = info
        
        return next_state, reward, terminated, truncated, info
    
    def is_done(self, terminated, truncated):
        """
        Check if episode is done.
        
        Args:
            terminated: Whether episode ended naturally
            truncated: Whether episode timed out
        
        Returns:
            True if episode is done
        """
        return terminated or truncated
    
    def get_action_space(self):
        """Return number of possible actions."""
        return self.env.action_space.n
    
    def get_observation_space(self):
        """Return observation space shape."""
        return self.env.observation_space.shape
    
    def render(self):
        """Render current frame (if render_mode='human')."""
        self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def get_current_state(self):
        """Return current state."""
        return self.state
    
    def sample_action(self):
        """Sample random action from action space."""
        return self.env.action_space.sample()


def get_environment_info(env):
    """
    Extract environment information for documentation.
    
    Args:
        env: AtariEnvironment instance
    
    Returns:
        Dictionary with environment details
    """
    info = {
        'game': env.game_name,
        'action_space': env.get_action_space(),
        'observation_space': env.get_observation_space(),
        'possible_actions': list(range(env.get_action_space())),
    }
    return info
