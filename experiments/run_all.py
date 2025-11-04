"""
Master Script to Run All Experiments

Runs all training experiments in sequence:
1. Baseline training
2. Bellman parameter experiments
3. Policy exploration experiments
4. Exploration parameter experiments

Run with: python experiments/run_all.py
"""

import subprocess
import sys
import time


def run_command(cmd, description):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run
        description: Description of what's running
    """
    print("\n" + "="*70)
    print(f"STARTING: {description}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=None)
        print(f"\n✓ {description} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {description}")
        print(f"Error code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Run all experiments."""
    
    print("\n" + "="*70)
    print("DEEP Q-LEARNING GALAXIAN - FULL EXPERIMENT SUITE")
    print("="*70)
    print(f"\nStarting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run the following experiments in order:")
    print("  1. Baseline training")
    print("  2. Bellman parameter variations")
    print("  3. Policy exploration variations")
    print("  4. Exploration parameter variations")
    print("\nEstimated time: ~3-5 hours (depending on hardware)")
    print("="*70)
    
    input("\nPress Enter to continue...")
    
    start_time = time.time()
    
    # Experiment 1: Baseline
    run_command(
        [sys.executable, "experiments/baseline.py"],
        "BASELINE TRAINING"
    )
    
    # Experiment 2: Bellman
    run_command(
        [sys.executable, "experiments/bellman_exp.py"],
        "BELLMAN PARAMETER EXPERIMENTS"
    )
    
    # Experiment 3: Policy
    run_command(
        [sys.executable, "experiments/policy_exp.py"],
        "POLICY EXPLORATION EXPERIMENTS"
    )
    
    # Experiment 4: Decay
    run_command(
        [sys.executable, "experiments/decay_exp.py"],
        "EXPLORATION PARAMETER EXPERIMENTS"
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved to:")
    print("  - results/baseline/")
    print("  - results/bellman/")
    print("  - results/policy/")
    print("  - results/decay/")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
        sys.exit(0)
