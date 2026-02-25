"""
Task 2: Comprehensive Training Script
Trains DQN and PPO agents on Chef's Hat Gym with sparse/delayed rewards
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime
import json

# Setup paths
from task2_setup import MODELS_DIR, LOGS_DIR, RESULTS_DIR

# Import agents
from task2_dqn_agent_v2 import DQNAgent
from task2_ppo_agent_v2 import PPOAgent


def setup_environment():
    """
    Setup Chef's Hat Gym environment
    """
    try:
        import gym
        print("✓ Gym imported successfully")
        
        # Try to import and register Chef's Hat if available
        try:
            import chefshatgym
            print("✓ ChefsHatGym package detected")
            
            # List available environments
            from gym import envs
            all_envs = [env_spec.id for env_spec in envs.registry.values()]
            chefshat_envs = [e for e in all_envs if 'chef' in e.lower()]
            
            if chefshat_envs:
                print(f"✓ Available Chef's Hat environments: {chefshat_envs}")
                return chefshat_envs[0]
            else:
                print("! No Chef's Hat environments found in registry")
                return None
                
        except ImportError:
            print("! ChefsHatGym not installed as package")
            # Try to load from local source
            try:
                sys.path.insert(0, str(Path(__file__).parent / "ChefsHatGYM_repo" / "src"))
                import chefshatgym
                print("✓ ChefsHatGym loaded from local source")
                from gym import envs
                all_envs = [env_spec.id for env_spec in envs.registry.values()]
                chefshat_envs = [e for e in all_envs if 'chef' in e.lower()]
                if chefshat_envs:
                    return chefshat_envs[0]
            except:
                pass
        
        return None
        
    except ImportError as e:
        print(f"! Error importing gym: {e}")
        return None


def create_dummy_environment():
    """
    Create a dummy environment for testing when Chef's Hat isn't available.
    This simulates a simplified game environment.
    """
    import gym
    from gym import spaces
    
    class ChefHatDummyEnv(gym.Env):
        """Simple dummy Chef's Hat-like environment for testing"""
        
        def __init__(self):
            self.action_space = spaces.Discrete(99)  # 99 possible actions
            # State: hand cards (12), table cards (5), round (1) = 18 dims
            self.observation_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32)
            self.game_length = np.random.randint(10, 50)
            self.current_step = 0
        
        def reset(self):
            """Reset environment and return initial observation"""
            self.current_step = 0
            self.game_length = np.random.randint(10, 50)
            return self._get_obs()
        
        def _get_obs(self):
            """Generate random observation"""
            return np.random.rand(18).astype(np.float32)
        
        def step(self, action):
            """
            Step environment with sparse reward
            
            Returns: observation, reward, done, info
            """
            self.current_step += 1
            done = self.current_step >= self.game_length
            
            # Sparse reward: only at end of game
            if done:
                # Win with some probability
                reward = 1.0 if np.random.random() > 0.5 else -1.0
            else:
                # No intermediate reward (sparse)
                reward = 0.0
            
            obs = self._get_obs()
            info = {'round': self.current_step, 'progress': self.current_step / self.game_length}
            
            return obs, reward, done, info
    
    return ChefHatDummyEnv()


def train_agent(agent_class, env, agent_name: str, num_episodes: int = 100,
                use_dummy: bool = False, **kwargs) -> Dict:
    """
    Train a single agent on the environment.
    
    Args:
        agent_class: Agent class (DQNAgent or PPOAgent)
        env: Environment
        agent_name: Name for logging
        num_episodes: Number of training episodes
        use_dummy: Whether using dummy environment
        **kwargs: Agent hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Training {agent_name}")
    print(f"{'='*60}")
    
    # Initialize agent
    agent = agent_class(agent_name=agent_name, **kwargs)
    agent.training_started = True
    agent.training_start_time = datetime.now()
    
    # Training loop
    max_steps = 500 if not use_dummy else 50
    
    for episode in range(num_episodes):
        try:
            # Train one episode
            episode_reward, episode_loss = agent.train_episode(env, max_steps=max_steps)
            
            # Log progress
            agent.log_training_progress(episode, episode_reward, episode_loss)
            
            # Save checkpoint every 25 episodes
            if (episode + 1) % 25 == 0:
                agent.save_checkpoint(episode)
        
        except Exception as e:
            print(f"! Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final model and summary
    print(f"\n✓ Training complete for {agent_name}")
    summary = agent.get_summary()
    
    return summary, agent


def evaluate_agents(agents_dict: Dict, env, num_episodes: int = 20) -> Dict:
    """
    Evaluate trained agents.
    
    Args:
        agents_dict: Dictionary of {agent_name: agent}
        env: Environment
        num_episodes: Number of evaluation episodes
    """
    print(f"\n{'='*60}")
    print("Evaluating Agents")
    print(f"{'='*60}")
    
    results = {}
    
    for agent_name, agent in agents_dict.items():
        print(f"\nEvaluating {agent_name}...")
        
        eval_rewards = []
        for episode in range(num_episodes):
            try:
                observation = env.reset()
                episode_reward = 0.0
                done = False
                step = 0
                
                while not done and step < 500:
                    # Act without training (greedy)
                    if hasattr(agent.act, '__call__'):
                        action = agent.act(observation, training=False)
                        if isinstance(action, tuple):
                            action = action[0]
                    else:
                        action = env.action_space.sample()
                    
                    # Step environment
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        observation, reward, done, info = step_result
                    else:
                        observation, reward, done, truncated, info = step_result
                        done = done or truncated
                    
                    episode_reward += reward
                    step += 1
                
                eval_rewards.append(episode_reward)
                print(f"  Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}")
            
            except Exception as e:
                print(f"  Error in evaluation episode {episode}: {e}")
                eval_rewards.append(0.0)
        
        # Compute statistics
        eval_summary = {
            'agent': agent_name,
            'num_eval_episodes': num_episodes,
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'max_reward': float(np.max(eval_rewards)),
            'min_reward': float(np.min(eval_rewards)),
        }
        
        results[agent_name] = eval_summary
        
        print(f"✓ {agent_name}: Mean = {eval_summary['mean_reward']:.2f}, "
              f"Std = {eval_summary['std_reward']:.2f}")
    
    return results


def main(args):
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("Task 2: RL Agent Training for Chef's Hat Gym")
    print("Sparse/Delayed Reward Focus")
    print("="*60)
    
    # Setup environment
    env_id = setup_environment()
    
    use_dummy = False
    if env_id is None:
        print("\n! Chef's Hat environment not found, using dummy environment for testing")
        env = create_dummy_environment()
        use_dummy = True
    else:
        print(f"\n✓ Using environment: {env_id}")
        try:
            import gym
            env = gym.make(env_id)
        except Exception as e:
            print(f"! Error creating environment: {e}")
            print("! Falling back to dummy environment")
            env = create_dummy_environment()
            use_dummy = True
    
    # Common hyperparameters
    # Detect state size from environment
    try:
        state_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 50
    except:
        state_size = 50
    
    hp_common = {
        'state_size': state_size,
        'action_size': 99,
        'gamma': 0.99,
        'device': 'cpu'
    }
    
    print(f"✓ State size detected: {state_size}")
    
    # Agent hyperparameters
    hp_dqn = {
        **hp_common,
        'learning_rate': 1e-3,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'replay_buffer_size': 100000,
        'batch_size': 64,
        'target_update_freq': 1000
    }
    
    hp_ppo = {
        **hp_common,
        'learning_rate': 3e-4,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coeff': 0.01,
        'value_loss_coeff': 0.5,
        'batch_size': 64,
        'epochs_per_update': 3
    }
    
    # Train agents
    agents_dict = {}
    summaries = {}
    
    # DQN Agent
    if args.train_dqn or args.train_both:
        summary, agent = train_agent(
            DQNAgent, env, "DQN_Agent",
            num_episodes=args.num_episodes,
            use_dummy=use_dummy,
            **hp_dqn
        )
        agents_dict['DQN_Agent'] = agent
        summaries['DQN_Agent'] = summary
    
    # PPO Agent
    if args.train_ppo or args.train_both:
        summary, agent = train_agent(
            PPOAgent, env, "PPO_Agent",
            num_episodes=args.num_episodes,
            use_dummy=use_dummy,
            **hp_ppo
        )
        agents_dict['PPO_Agent'] = agent
        summaries['PPO_Agent'] = summary
    
    # Evaluate agents
    if len(agents_dict) > 0:
        eval_results = evaluate_agents(agents_dict, env, num_episodes=args.eval_episodes)
        
        # Save evaluation results
        eval_path = RESULTS_DIR / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n✓ Evaluation results saved: {eval_path}")
    
    # Save training summaries
    summary_path = RESULTS_DIR / "training_summaries.json"
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"✓ Training summaries saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agents on Chef's Hat Gym with sparse rewards"
    )
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--train-dqn', action='store_true',
                       help='Train DQN agent only')
    parser.add_argument('--train-ppo', action='store_true',
                       help='Train PPO agent only')
    parser.add_argument('--train-both', action='store_true',
                       help='Train both agents (default)')
    
    args = parser.parse_args()
    
    # Default to training both if neither specified
    if not (args.train_dqn or args.train_ppo):
        args.train_both = True
    
    main(args)
