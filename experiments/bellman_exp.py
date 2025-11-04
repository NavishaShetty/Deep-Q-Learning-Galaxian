"""
Bellman Equation Parameter Experiments

Experiments with different alpha (learning_rate) and gamma (discount_factor) values.

Test cases:
- Baseline: alpha=0.7, gamma=0.8
- Variation 1: alpha=0.5, gamma=0.8 (lower learning rate)
- Variation 2: alpha=0.9, gamma=0.8 (higher learning rate)
- Variation 3: alpha=0.7, gamma=0.6 (lower discount)
- Variation 4: alpha=0.7, gamma=0.95 (higher discount)
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


def train_with_params(alpha, gamma, experiment_name, num_episodes=5000, max_steps=99):
    """
    Train agent with specific Bellman parameters.
    
    Args:
        alpha: Learning rate
        gamma: Discount factor
        experiment_name: Name for this experiment
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
    
    Returns:
        MetricsTracker with training results
    """
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Alpha (learning rate): {alpha}")
    print(f"Gamma (discount factor): {gamma}")
    print(f"{'='*60}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize environment
    env = AtariEnvironment(game_name="ALE/Galaxian-v5")
    state_size = env.get_observation_space()
    action_size = env.get_action_space()
    
    # Initialize agent with specific parameters
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=alpha,
        gamma=gamma,
        epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.01,
        device=device
    )
    
    # Training
    metrics = MetricsTracker(window_size=100)
    
    print(f"Training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.act(state, epsilon=agent.epsilon, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay(batch_size=32)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        agent.decay_exploration(episode)
        metrics.add_episode(episode, episode_reward, episode_length, agent.epsilon)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = metrics.get_moving_average_reward()
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    print(f"Training complete!")
    metrics.print_summary()
    
    return metrics, agent


def run_bellman_experiments():
    """Run all Bellman parameter experiments."""
    
    print("\n" + "="*60)
    print("BELLMAN EQUATION PARAMETER EXPERIMENTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("results/bellman", exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        {"alpha": 0.5, "gamma": 0.8, "name": "Low Alpha (0.5)"},
        {"alpha": 0.9, "gamma": 0.8, "name": "High Alpha (0.9)"},
        {"alpha": 0.7, "gamma": 0.6, "name": "Low Gamma (0.6)"},
        {"alpha": 0.7, "gamma": 0.95, "name": "High Gamma (0.95)"},
    ]
    
    results_summary = []
    
    for exp in experiments:
        alpha = exp["alpha"]
        gamma = exp["gamma"]
        name = exp["name"]
        
        metrics, agent = train_with_params(
            alpha=alpha,
            gamma=gamma,
            experiment_name=name,
            num_episodes=5000,
            max_steps=99
        )
        
        # Save individual experiment results
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        metrics.save_to_csv(f"results/bellman/{safe_name}_metrics.csv")
        agent.save_model(f"results/bellman/{safe_name}_model.pt")
        
        # Collect summary stats
        avg_reward = np.mean(metrics.episode_rewards[-100:])
        max_reward = np.max(metrics.episode_rewards)
        results_summary.append({
            'experiment': name,
            'alpha': alpha,
            'gamma': gamma,
            'final_avg_reward': avg_reward,
            'max_reward': max_reward,
            'num_episodes': len(metrics.episode_rewards),
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/bellman/summary.csv", index=False)
    
    print("\n" + "="*60)
    print("BELLMAN EXPERIMENTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")
    
    print("Results saved to: results/bellman/")


if __name__ == "__main__":
    run_bellman_experiments()
