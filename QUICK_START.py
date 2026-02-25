#!/usr/bin/env python
"""
Task 2: Quick Start Guide - Run Complete Implementation
========================================================

This script demonstrates how to use all components together
"""

# ==============================================================================
# QUICK START: Run Complete Task 2
# ==============================================================================

def quick_start_examples():
    """
    Examples of running the complete Task 2 implementation
    """
    
    # Example 1: Default run (train both agents, 100 episodes)
    print("Example 1: Default run")
    print("  Command: python run_task2.py")
    print("  Trains: DQN + PPO for 100 episodes each")
    print("  Evaluates: 20 episodes per agent")
    print()
    
    # Example 2: Quick test
    print("Example 2: Quick test (25 episodes)")
    print("  Command: python run_task2.py --episodes 25")
    print()
    
    # Example 3: DQN only
    print("Example 3: Train DQN only")
    print("  Command: python run_task2.py --train-dqn --episodes 75")
    print()
    
    # Example 4: PPO only
    print("Example 4: Train PPO only")
    print("  Command: python run_task2.py --train-ppo --episodes 75")
    print()
    
    # Example 5: Detailed analysis
    print("Example 5: Full analysis with detailed reports")
    print("  Command: python run_task2.py --detailed-analysis --eval-episodes 30")
    print()
    
    # Example 6: Custom training
    print("Example 6: Custom training (50 episodes, no evaluation)")
    print("  Command: python run_task2.py --episodes 50 --no-eval")
    print()


def code_examples():
    """
    Show how to use the agents programmatically
    """
    
    code = '''
# Example: Custom Training Code
# ================================

import sys
sys.path.insert(0, 'path/to/task2')

from task2_setup import MODELS_DIR, RESULTS_DIR
from task2_dqn_agent_v2 import DQNAgent
from task2_ppo_agent_v2 import PPOAgent
import gym

# 1. Create agents with custom hyperparameters
# ============================================

dqn_agent = DQNAgent(
    agent_name="DQN_Custom",
    state_size=50,
    action_size=99,
    learning_rate=5e-4,  # Custom learning rate
    epsilon_start=0.95,
    batch_size=32
)

ppo_agent = PPOAgent(
    agent_name="PPO_Custom",
    state_size=50,
    action_size=99,
    learning_rate=2e-4,
    gae_lambda=0.98,  # Slightly higher lambda
    clip_ratio=0.15   # Tighter clipping
)

# 2. Create or load environment
# ==============================

# Try to load Chef's Hat environment
try:
    env = gym.make("ChefHat-v0")
except:
    # Fallback to dummy environment
    from train_complete import create_dummy_environment
    env = create_dummy_environment()

# 3. Train agents
# ================

num_episodes = 100

for episode in range(num_episodes):
    # Train DQN
    dqn_reward, dqn_loss = dqn_agent.train_episode(env)
    
    # Train PPO
    ppo_reward, ppo_loss = ppo_agent.train_episode(env)
    
    # Log progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}")
        print(f"  DQN: Reward={dqn_reward:.2f}, Loss={dqn_loss:.4f}")
        print(f"  PPO: Reward={ppo_reward:.2f}, Loss={ppo_loss:.4f}")
    
    # Save checkpoints
    if (episode + 1) % 25 == 0:
        dqn_agent.save_checkpoint(episode)
        ppo_agent.save_checkpoint(episode)

# 4. Generate summaries
# =====================

dqn_summary = dqn_agent.save_summary()
ppo_summary = ppo_agent.save_summary()

print("DQN Summary:", dqn_summary)
print("PPO Summary:", ppo_summary)

# 5. Evaluate agents
# ===================

for episode in range(20):
    observation = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Get action (greedy, no training)
        action = dqn_agent.act(observation, training=False)
        
        # Take step
        observation, reward, done, info = env.step(action)
        episode_reward += reward
    
    print(f"Eval {episode}: Reward={episode_reward:.2f}")
    '''
    
    return code


# ==============================================================================
# CONFIGURATION REFERENCE
# ==============================================================================

def config_reference():
    """
    Show configuration options for agents
    """
    
    config = {
        'DQN Agent': {
            'learning_rate': '1e-3, [1e-4, 1e-2]',
            'epsilon_start': '1.0',
            'epsilon_end': '0.01, [0.001, 0.1]',
            'epsilon_decay': '0.995, [0.99, 0.999]',
            'batch_size': '64, [32, 128]',
            'replay_buffer_size': '100000, [50000, 500000]',
            'target_update_freq': '1000, [500, 5000]',
            'gamma': '0.99, [0.95, 0.999]'
        },
        'PPO Agent': {
            'learning_rate': '3e-4, [1e-4, 1e-3]',
            'gamma': '0.99, [0.95, 0.999]',
            'gae_lambda': '0.95, [0.90, 0.99]',
            'clip_ratio': '0.2, [0.1, 0.3]',
            'entropy_coeff': '0.01, [0.001, 0.1]',
            'value_loss_coeff': '0.5, [0.1, 1.0]',
            'epochs_per_update': '3, [1, 10]',
            'batch_size': '64, [32, 128]'
        }
    }
    
    print("\n" + "="*70)
    print("HYPERPARAMETER REFERENCE")
    print("="*70)
    
    for agent, params in config.items():
        print(f"\n{agent}:")
        for param, value in params.items():
            print(f"  {param:<25} {value}")


# ==============================================================================
# OUTPUT FILES REFERENCE
# ==============================================================================

def output_reference():
    """
    Document the output files generated
    """
    
    outputs = {
        'task2_outputs/models/': {
            'format': 'PyTorch .pt files',
            'content': ['DQN_Agent_episode_*.pt', 'PPO_Agent_episode_*.pt'],
            'description': 'Trained agent network weights at checkpoint episodes'
        },
        'task2_outputs/results/': {
            'format': 'Mixed (JSON, TXT)',
            'content': [
                'training_summaries.json - Summary statistics for both agents',
                'comparison_report.txt - Side-by-side performance comparison',
                'evaluation_results.json - Evaluation metrics',
                'DQN_Agent_summary.json - DQN-specific summary',
                'PPO_Agent_summary.json - PPO-specific summary'
            ],
            'description': 'Analysis and results documentation'
        }
    }
    
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70 + "\n")
    
    for location, info in outputs.items():
        print(f"📁 {location}")
        print(f"   Format: {info['format']}")
        print(f"   Description: {info['description']}")
        print("   Files:")
        for file in info['content']:
            print(f"     - {file}")
        print()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TASK 2: RL AGENTS FOR CHEF'S HAT GYM - QUICK START GUIDE")
    print("="*70)
    
    print("\n" + "-"*70)
    print("COMMAND LINE EXAMPLES")
    print("-"*70)
    quick_start_examples()
    
    print("\n" + "-"*70)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("-"*70)
    print(code_examples())
    
    print("\n" + "-"*70)
    print("CONFIGURATION")
    print("-"*70)
    config_reference()
    
    print("\n" + "-"*70)
    print("OUTPUT FILES")
    print("-"*70)
    output_reference()
    
    print("\n" + "="*70)
    print("GET STARTED:")
    print("  python run_task2.py")
    print("="*70 + "\n")
