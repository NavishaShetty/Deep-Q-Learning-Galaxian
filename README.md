# Deep Q-Learning for Atari Galaxian

A Deep Q-Network (DQN) implementation for the Atari Galaxian environment, exploring various hyperparameter configurations and exploration strategies to optimize agent performance.

## Folder Structure

```
├── src/
│    ├── agent.py                          # DQN agent implementation
│    ├── environment.py                    # Atari environment wrapper
│    ├── neural_network.py                 # Q-network architecture
│    ├── replay_buffer.py                  # Experience replay buffer
│    └── utils.py                          # Utility functions
│
│── experiments/ 
│    ├── baseline.py                       # Baseline experiment
│    ├── bellman_exp.py                    # Bellman equation parameter experiments
│    ├── decay_exp.py                      # Epsilon decay experiments
│    ├── policy_exp.py                     # Policy exploration experiments
│    └── run_all.py                        # Run all experiments
│
├── notebooks/
│    └── colab_training.ipynb              # Colab training notebook
│
├── results/
│    ├── bellman_comparison.csv            # Bellman experiment results
│    ├── decay_comparison.csv              # Decay experiment results
│    ├── policy_comparison.csv             # Policy experiment results
│    ├── summary.csv                       # Overall results summary
│    ├── baseline/                         # Model and result files
│    ├── bellman/                          # Model and result files
│    ├── decay/                            # Model and result files
│    ├── policy/                           # Model and result files
│    └── Visualizations/
│           ├── bellman_experiments.png           # Bellman results visualization
│           ├── decay_experiments.png             # Decay results visualization
│           ├── policy_experiments.png            # Policy results visualization
│           └──overall_comparison.png            # Combined results comparison
│
├── docs/
│    ├── baseline.py                       # Detailed Documentation
│    ├── bellman_exp.py                    # Results Summary
│    └── run_all.py                        # Attribution and Licensing information 
└── README.md                              # This file
```

## Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Running Experiments

**Run all experiments:**
```bash
python run_all.py
```

**Run individual experiments:**
```bash
# Baseline
python baseline.py

# Bellman equation parameters (alpha, gamma)
python bellman_exp.py

# Epsilon decay rates
python decay_exp.py

# Policy exploration (Boltzmann vs Epsilon-greedy)
python policy_exp.py
```

### Training Parameters

Default baseline configuration:
- **Episodes**: 1000
- **Learning Rate (α)**: 0.001
- **Discount Factor (γ)**: 0.99
- **Starting Epsilon**: 1.0
- **Epsilon Decay**: 0.001
- **Minimum Epsilon**: 0.01
- **Replay Buffer**: 10,000
- **Batch Size**: 32

## Results Summary

### Key Findings

| Experiment | Configuration | Avg Reward | Improvement |
|------------|--------------|------------|-------------|
| **Baseline** | Standard ε-greedy | 380 | - |
| **Boltzmann** | Temperature 0.5 | 735 | **+93%** |
| High Alpha | α = 0.9 | 350 | -8% |
| High Gamma | γ = 0.95 | 420 | +11% |
| Slow Decay | 0.0005 | 390 | +3% |

### Main Insights

1. **Policy Exploration Dominates**: Boltzmann exploration policy achieved 93% better performance than epsilon-greedy, demonstrating that exploration strategy has greater impact than hyperparameter tuning.

2. **Gamma Sensitivity**: Higher discount factors (γ = 0.95) improved performance by considering longer-term rewards.

3. **Decay Rate Impact**: Slower epsilon decay (0.0005) showed marginal improvement by allowing extended exploration.

4. **Alpha Stability**: Extremely high learning rates (α = 0.9) degraded performance, confirming the importance of stable learning.

### Visualizations

- **Overall Comparison**: `overall_comparison.png`
- **Bellman Experiments**: `bellman_experiments.png`
- **Decay Experiments**: `decay_experiments.png`
- **Policy Experiments**: `policy_experiments.png`

## References

### Primary Resources
- [Deep Reinforcement Learning for Atari Games Tutorial](https://towardsdatascience.com/deep-reinforcement-learning-for-atari-games-in-pytorch-5e2d5d0a6e42)
- [Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)

### Implementation References
- PyTorch DQN Tutorial: [https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- OpenAI Gymnasium: [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

### Course Materials
- Northeastern University MSDE Program
- Assignment: LLM Agents & Deep Q-Learning with Atari Games

---

**License**: MIT License (see code files for details)

**Author**: Navisha Shetty | Northeastern University

**Date**: November 2025