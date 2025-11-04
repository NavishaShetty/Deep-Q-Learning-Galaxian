"""
Policy Exploration Experiments

Test alternative exploration policies instead of ε-greedy.

Policies:
- Baseline: ε-greedy
- Variation: Boltzmann (softmax) exploration
- Variation: UCB (Upper Confidence Bound)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pandas as pd
from src.agent import DQNAgent
from src.environment import AtariEnvironment
from src.utils import MetricsTracker, preprocess_state


class BoltzmannAgent(DQNAgent):
    """DQN Agent with Boltzmann exploration policy."""
    
    def __init__(self, *args, temperature=1.0, **kwargs):
        """
        Initialize Boltzmann agent.
        
        Args:
            temperature: Temperature for softmax (higher = more random)
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    
    def act(self, state, training=True):
        """
        Choose action using Boltzmann exploration.
        
        Uses softmax over Q-values to select action probabilistically.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Action to take
        """
        state_tensor = torch.from_numpy(preprocess_state(state)).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
        
        # Softmax with temperature
        q_values_scaled = q_values / self.temperature
        probabilities = torch.softmax(q_values_scaled, dim=0)
        
        # Sample action according to probabilities
        action = torch.multinomial(probabilities, num_samples=1).item()
        
        return action


class UCBAgent(DQNAgent):
    """DQN Agent with UCB exploration policy."""
    
    def __init__(self, *args, **kwargs):
        """Initialize UCB agent."""
        super().__init__(*args, **kwargs)
        self.action_counts = np.zeros(self.action_size)
    
    def act(self, state, training=True):
        """
        Choose action using UCB exploration.
        
        Balances exploitation of high Q-values with exploration of rarely-tried actions.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Action to take
        """
        state_tensor = torch.from_numpy(preprocess_state(state)).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
        
        # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
        total_count = np.sum(self.action_counts)
        if total_count == 0:
            total_count = 1
        
        ucb_values = q_values + np.sqrt(2 * np.log(total_count + 1) / (self.action_counts + 1))
        action = np.argmax(ucb_values)
        
        self.action_counts[action] += 1
        
        return action


def train_with_policy(agent_class, agent_name, agent_kwargs, num_episodes=5000, max_steps=99):
    """
    Train agent with specific exploration policy.
    
    Args:
        agent_class: Agent class (DQNAgent, BoltzmannAgent, or UCBAgent)
        agent_name: Name of policy
        agent_kwargs: Keyword arguments for agent initialization
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
    
    Returns:
        MetricsTracker with training results
    """
    
    print(f"\n{'='*60}")
    print(f"Policy: {agent_name}")
    print(f"{'='*60}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize environment
    env = AtariEnvironment(game_name="ALE/Galaxian-v5")
    state_size = env.get_observation_space()
    action_size = env.get_action_space()
    
    # Initialize agent
    agent_kwargs.update({
        'state_size': state_size,
        'action_size': action_size,
        'device': device,
    })
    agent = agent_class(**agent_kwargs)
    
    # Training
    metrics = MetricsTracker(window_size=100)
    
    print(f"Training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Choose action with specific policy
            if isinstance(agent, DQNAgent) and not isinstance(agent, (BoltzmannAgent, UCBAgent)):
                # Standard ε-greedy
                action = agent.act(state, epsilon=agent.epsilon, training=True)
            else:
                # Boltzmann or UCB
                action = agent.act(state, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay(batch_size=32)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Decay epsilon for standard agent
        if isinstance(agent, DQNAgent) and not isinstance(agent, (BoltzmannAgent, UCBAgent)):
            agent.decay_exploration(episode)
        
        epsilon_val = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
        metrics.add_episode(episode, episode_reward, episode_length, epsilon_val)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = metrics.get_moving_average_reward()
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f}")
            
        # Memory cleanup
        if episode % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    env.close()
    
    print(f"Training complete!")
    metrics.print_summary()
    
    return metrics, agent


def run_policy_experiments():
    """Run all policy exploration experiments."""
    
    print("\n" + "="*60)
    print("POLICY EXPLORATION EXPERIMENTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("results/policy", exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        {
            "agent_class": BoltzmannAgent,
            "agent_name": "Boltzmann (Temperature=0.5)",
            "agent_kwargs": {
                "learning_rate": 0.7,
                "gamma": 0.8,
                "epsilon": 1.0,
                "min_epsilon": 0.01,
                "decay_rate": 0.01,
                "temperature": 0.5,
            }
        },
        {
            "agent_class": BoltzmannAgent,
            "agent_name": "Boltzmann (Temperature=2.0)",
            "agent_kwargs": {
                "learning_rate": 0.7,
                "gamma": 0.8,
                "epsilon": 1.0,
                "min_epsilon": 0.01,
                "decay_rate": 0.01,
                "temperature": 2.0,
            }
        },
        {
            "agent_class": UCBAgent,
            "agent_name": "UCB (Upper Confidence Bound)",
            "agent_kwargs": {
                "learning_rate": 0.7,
                "gamma": 0.8,
                "epsilon": 0.0,  # UCB doesn't use epsilon
                "min_epsilon": 0.0,
                "decay_rate": 0.0,
            }
        },
    ]
    
    results_summary = []
    
    for exp in experiments:
        agent_class = exp["agent_class"]
        agent_name = exp["agent_name"]
        agent_kwargs = exp["agent_kwargs"]
        
        metrics, agent = train_with_policy(
            agent_class=agent_class,
            agent_name=agent_name,
            agent_kwargs=agent_kwargs,
            num_episodes=5000,
            max_steps=99
        )
        
        # Save individual experiment results
        safe_name = agent_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        metrics.save_to_csv(f"results/policy/{safe_name}_metrics.csv")
        agent.save_model(f"results/policy/{safe_name}_model.pt")
        
        # Collect summary stats
        avg_reward = np.mean(metrics.episode_rewards[-100:])
        max_reward = np.max(metrics.episode_rewards)
        results_summary.append({
            'policy': agent_name,
            'final_avg_reward': avg_reward,
            'max_reward': max_reward,
            'num_episodes': len(metrics.episode_rewards),
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/policy/summary.csv", index=False)
    
    print("\n" + "="*60)
    print("POLICY EXPERIMENTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")
    
    print("Results saved to: results/policy/")


if __name__ == "__main__":
    run_policy_experiments()
