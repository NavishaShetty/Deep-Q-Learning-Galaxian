"""
Baseline DQN Training on Galaxian

Trains DQN agent with suggested hyperparameters:
- total_episodes = 5000
- total_test_episodes = 100
- max_steps = 99
- learning_rate = 0.7
- gamma = 0.8
- epsilon = 1.0
- decay_rate = 0.01

Saves metrics to results/baseline/metrics.csv
Saves model to results/baseline/model.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.agent import DQNAgent
from src.environment import AtariEnvironment
from src.utils import MetricsTracker, preprocess_state


def train_baseline():
    """Run baseline training."""
    
    print("="*60)
    print("BASELINE DQN TRAINING - SEAQUEST")
    print("="*60)
    
    # Hyperparameters
    TOTAL_EPISODES = 5000
    TOTAL_TEST_EPISODES = 100
    MAX_STEPS = 99
    LEARNING_RATE = 0.7  # alpha
    GAMMA = 0.8          # discount factor
    EPSILON = 1.0
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.01
    DECAY_RATE = 0.01
    BATCH_SIZE = 32
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initialize environment
    print("Initializing environment...")
    env = AtariEnvironment(game_name="ALE/Galaxian-v5")
    state_size = env.get_observation_space()
    action_size = env.get_action_space()
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}\n")
    
    # Initialize agent
    print("Initializing agent...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        min_epsilon=MIN_EPSILON,
        decay_rate=DECAY_RATE,
        device=device
    )
    print(f"Learning rate (alpha): {LEARNING_RATE}")
    print(f"Gamma (discount factor): {GAMMA}")
    print(f"Initial epsilon: {EPSILON}")
    print(f"Min epsilon: {MIN_EPSILON}")
    print(f"Decay rate: {DECAY_RATE}\n")
    
    # Initialize metrics tracker
    metrics = MetricsTracker(window_size=100)
    
    # Training phase
    print("Starting training phase...")
    print("-" * 60)
    
    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(MAX_STEPS):
            # Choose action using ε-greedy policy
            action = agent.act(state, epsilon=agent.epsilon, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on random batch
            loss = agent.replay(batch_size=BATCH_SIZE)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Decay exploration rate
        agent.decay_exploration(episode)
        
        # Track metrics
        metrics.add_episode(episode, episode_reward, episode_length, agent.epsilon)
        
        # Print progress
        if (episode + 1) % 500 == 0 or episode == 0:
            avg_reward = metrics.get_moving_average_reward()
            print(f"Episode {episode+1:4d}/{TOTAL_EPISODES} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg (100 ep): {avg_reward:6.1f} | "
                  f"Length: {episode_length:3d} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 60)
    print("Training complete!\n")
    
    # Testing phase
    print("Starting testing phase...")
    print("-" * 60)
    
    test_rewards = []
    test_lengths = []
    
    for test_episode in range(TOTAL_TEST_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(MAX_STEPS):
            # Choose action greedily (no exploration)
            action = agent.act(state, epsilon=0.0, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        
        if (test_episode + 1) % 20 == 0 or test_episode == 0:
            print(f"Test Episode {test_episode+1:3d}/{TOTAL_TEST_EPISODES} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Length: {episode_length:3d}")
    
    print("-" * 60)
    print("Testing complete!\n")
    
    # Print test summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    print(f"Max test reward: {np.max(test_rewards):.2f}")
    print(f"Min test reward: {np.min(test_rewards):.2f}")
    print(f"Std dev: {np.std(test_rewards):.2f}")
    print(f"Average test length: {np.mean(test_lengths):.2f}")
    print("="*60 + "\n")
    
    # Save results
    print("Saving results...")
    os.makedirs("results/baseline", exist_ok=True)
    
    metrics.save_to_csv("results/baseline/metrics.csv")
    agent.save_model("results/baseline/model.pt")
    
    # Save test results
    test_df_data = {
        'test_episode': list(range(TOTAL_TEST_EPISODES)),
        'test_reward': test_rewards,
        'test_length': test_lengths,
    }
    import pandas as pd
    test_df = pd.DataFrame(test_df_data)
    test_df.to_csv("results/baseline/test_metrics.csv", index=False)
    
    # Print training summary
    metrics.print_summary()
    
    # Cleanup
    env.close()
    
    print("Baseline training complete!")
    print(f"Results saved to: results/baseline/")
    print(f"  - metrics.csv: Training metrics")
    print(f"  - test_metrics.csv: Testing metrics")
    print(f"  - model.pt: Trained model weights")


if __name__ == "__main__":
    train_baseline()
