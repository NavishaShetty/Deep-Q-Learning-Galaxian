"""
Exploration Parameters Experiments

Experiments with different epsilon decay rates and starting epsilons.

Test cases:
- Baseline: decay_rate=0.01, epsilon=1.0
- Variation 1: decay_rate=0.005 (slower decay)
- Variation 2: decay_rate=0.02 (faster decay)
- Variation 3: epsilon=0.5 (start less exploratory)
- Variation 4: epsilon=1.0 with faster decay
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


def train_with_decay_params(decay_rate, initial_epsilon, experiment_name, num_episodes=5000, max_steps=99):
    """
    Train agent with specific exploration parameters.
    
    Args:
        decay_rate: Epsilon decay rate
        initial_epsilon: Initial epsilon value
        experiment_name: Name for this experiment
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
    
    Returns:
        MetricsTracker with training results
    """
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Decay rate: {decay_rate}")
    print(f"Initial epsilon: {initial_epsilon}")
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
        learning_rate=0.7,
        gamma=0.8,
        epsilon=initial_epsilon,
        min_epsilon=0.01,
        decay_rate=decay_rate,
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
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # Memory cleanup
        if episode % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    env.close()
    
    print(f"Training complete!")
    print(f"Final epsilon at max steps: {agent.epsilon:.6f}")
    metrics.print_summary()
    
    return metrics, agent


def run_decay_experiments():
    """Run all exploration parameter experiments."""
    
    print("\n" + "="*60)
    print("EXPLORATION PARAMETER EXPERIMENTS")
    print("="*60)
    
    # Create output directory
    os.makedirs("results/decay", exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        {"decay_rate": 0.005, "epsilon": 1.0, "name": "Slow Decay (0.005)"},
        {"decay_rate": 0.02, "epsilon": 1.0, "name": "Fast Decay (0.02)"},
        {"decay_rate": 0.01, "epsilon": 0.5, "name": "Lower Starting Epsilon (0.5)"},
        {"decay_rate": 0.01, "epsilon": 1.0, "name": "Standard (0.01, 1.0)"},
    ]
    
    results_summary = []
    epsilon_decay_curves = {}
    
    for exp in experiments:
        decay_rate = exp["decay_rate"]
        epsilon = exp["epsilon"]
        name = exp["name"]
        
        metrics, agent = train_with_decay_params(
            decay_rate=decay_rate,
            initial_epsilon=epsilon,
            experiment_name=name,
            num_episodes=5000,
            max_steps=99
        )
        
        # Save individual experiment results
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        metrics.save_to_csv(f"results/decay/{safe_name}_metrics.csv")
        agent.save_model(f"results/decay/{safe_name}_model.pt")
        
        # Store epsilon decay curve
        epsilon_decay_curves[name] = metrics.epsilons
        
        # Collect summary stats
        avg_reward = np.mean(metrics.episode_rewards[-100:])
        max_reward = np.max(metrics.episode_rewards)
        final_epsilon = metrics.epsilons[-1]
        
        results_summary.append({
            'experiment': name,
            'decay_rate': decay_rate,
            'initial_epsilon': epsilon,
            'final_epsilon': final_epsilon,
            'final_avg_reward': avg_reward,
            'max_reward': max_reward,
            'num_episodes': len(metrics.episode_rewards),
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/decay/summary.csv", index=False)
    
    print("\n" + "="*60)
    print("EXPLORATION PARAMETER EXPERIMENTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Calculate epsilon at max_steps (99) for each experiment
    print("Epsilon values at max steps (step 99):")
    print("-" * 60)
    for name, epsilons in epsilon_decay_curves.items():
        if len(epsilons) >= 1:
            print(f"{name:30s}: ε = {epsilons[0]:.6f}")
    print("-" * 60 + "\n")
    
    print("Results saved to: results/decay/")


if __name__ == "__main__":
    run_decay_experiments()
