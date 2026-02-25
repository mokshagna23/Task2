"""
Task 2: Enhanced Base Agent with Sparse/Delayed Reward Handling
Provides foundational classes for all RL agents
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path

# Import setup first
from task2_setup import LOGS_DIR, MODELS_DIR, RESULTS_DIR


@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    valid_actions: Optional[np.ndarray] = None


class StateRepresentation:
    """
    Converts raw environment observations to normalized feature vectors.
    
    Features extracted:
    - Current player's hand encoding
    - Table state encoding
    - Opponent hands (hidden)
    - Game progress indicators
    """
    
    def __init__(self, max_hand_size: int = 12, num_players: int = 4):
        self.max_hand_size = max_hand_size
        self.num_players = num_players
        
    def encode_observation(self, observation: Any) -> np.ndarray:
        """
        Convert raw observation to feature vector.
        
        Returns normalized feature vector [0, 1]
        """
        if isinstance(observation, dict):
            # Handle dict-based observations
            features = []
            
            # Player's hand
            if 'hand' in observation:
                hand = observation['hand']
                hand_encoding = self._encode_hand(hand)
                features.extend(hand_encoding)
            
            # Table state
            if 'table' in observation:
                table_encoding = self._encode_table(observation['table'])
                features.extend(table_encoding)
            
            # Game state
            if 'round' in observation:
                features.append(observation['round'] / 100.0)  # Normalize
                
            return np.array(features, dtype=np.float32)
        
        elif isinstance(observation, np.ndarray):
            # Already numeric, normalize
            return observation.astype(np.float32) / (max(np.max(observation), 1.0))
        
        else:
            raise ValueError(f"Unsupported observation type: {type(observation)}")
    
    def _encode_hand(self, hand: Any) -> List[float]:
        """Encode player hand as normalized features"""
        encoding = [0.0] * self.max_hand_size
        if hasattr(hand, '__len__'):
            for i, card in enumerate(hand[:self.max_hand_size]):
                encoding[i] = float(card) / 50.0  # Normalize card value
        return encoding
    
    def _encode_table(self, table: Any) -> List[float]:
        """Encode table/discard pile state"""
        encoding = [0.0]  # Number of cards on table, normalized
        if hasattr(table, '__len__'):
            encoding[0] = min(len(table), 100) / 100.0
        return encoding


class ActionHandler:
    """
    Manages action space and masking of invalid actions
    """
    
    def __init__(self, action_space_size: int, enforce_masking: bool = True):
        self.action_space_size = action_space_size
        self.enforce_masking = enforce_masking
        
    def get_valid_actions(self, observation: Any) -> np.ndarray:
        """
        Extract valid actions from observation.
        
        Returns binary mask [0, 1] of valid actions
        """
        if isinstance(observation, dict) and 'valid_actions' in observation:
            valid_mask = observation['valid_actions']
            if isinstance(valid_mask, (list, tuple)):
                return np.array(valid_mask, dtype=np.float32)
        
        # Default: all actions valid
        return np.ones(self.action_space_size, dtype=np.float32)
    
    def mask_invalid_actions(self, q_values: np.ndarray, 
                            valid_actions: np.ndarray) -> np.ndarray:
        """
        Mask Q-values for invalid actions (set to -inf)
        """
        if not self.enforce_masking:
            return q_values
            
        masked = q_values.copy()
        masked[valid_actions == 0] = -np.inf
        return masked
    
    def select_action(self, q_values: np.ndarray, 
                     valid_actions: np.ndarray,
                     epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy with action masking
        """
        if np.random.random() < epsilon:
            # Explore: sample from valid actions only
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) == 0:
                return 0
            return np.random.choice(valid_indices)
        else:
            # Exploit: choose best valid action
            masked_q = self.mask_invalid_actions(q_values, valid_actions)
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) == 0:
                return 0
            best_idx = np.argmax(masked_q[valid_indices])
            return valid_indices[best_idx]


class RewardHandler:
    """
    Handles sparse/delayed reward transformation.
    
    Strategy for sparse rewards in Chef's Hat:
    - Assign actual reward only at end of game
    - Small shaping signal: penalty for slow games, bonus for efficiency
    - Track cumulative reward separately
    """
    
    def __init__(self, shaping_enabled: bool = True):
        self.shaping_enabled = shaping_enabled
        
    def process_reward(self, raw_reward: float, done: bool, 
                      info: Dict[str, Any], step: int = 0) -> float:
        """
        Transform sparse game reward with optional shaping.
        
        For sparse rewards:
        - If done: return final game reward
        - If not done: return small shaping signal
        """
        if done:
            # Terminal reward (actual game outcome)
            return float(raw_reward)
        else:
            # Intermediate reward shaping
            if self.shaping_enabled:
                # Penalty for taking too long (encourages faster games)
                time_penalty = -0.001
                
                # Small bonus if info suggests progress (if available)
                progress_bonus = 0.0
                if isinstance(info, dict) and 'progress' in info:
                    progress_bonus = 0.001 * info.get('progress', 0)
                
                return time_penalty + progress_bonus
            else:
                return 0.0


class ReplayBuffer:
    """
    Prioritized experience replay buffer for off-policy learning
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, 
            valid_actions: Optional[np.ndarray] = None,
            td_error: float = 1.0):
        """Add experience with priority"""
        experience = Experience(state, action, reward, next_state, done, valid_actions)
        self.buffer.append(experience)
        # Priority proportional to TD error (1.0 for new experiences)
        self.priorities.append(min(abs(td_error) + 1e-6, 1.0))
    
    def sample(self, batch_size: int = 64, 
               use_priority: bool = True) -> Tuple[List, List]:
        """
        Sample batch of experiences.
        
        Returns:
            experiences: List of Experience objects
            indices: Indices of sampled experiences (for priority update)
        """
        if len(self.buffer) == 0:
            return [], []
        
        batch_size = min(batch_size, len(self.buffer))
        
        if use_priority and len(self.priorities) > 0:
            # Priority sampling
            priorities = np.array(self.priorities)
            priorities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=priorities)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = min(abs(float(td_error)) + 1e-6, 1.0)
    
    def __len__(self):
        return len(self.buffer)


class BaseRLAgent:
    """
    Base class for all RL agents with logging and checkpointing
    """
    
    def __init__(self, agent_name: str, num_players: int = 4,
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 device: str = "cpu"):
        self.agent_name = agent_name
        self.num_players = num_players
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)
        
        # State/action/reward handling
        self.state_encoder = StateRepresentation()
        self.action_handler = ActionHandler(action_space_size=99)  # Chef's Hat has ~99 actions
        self.reward_handler = RewardHandler()
        
        # Training tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.total_steps = 0
        self.training_started = False
        self.training_start_time = None
        
    def act(self, observation: Any, valid_actions: Optional[np.ndarray] = None,
            training: bool = True) -> int:
        """
        Select action from observation.
        To be implemented by subclasses.
        """
        raise NotImplementedError
    
    def learn(self, batch_size: int = 64) -> float:
        """
        Perform learning update step.
        To be implemented by subclasses.
        
        Returns loss value
        """
        raise NotImplementedError
    
    def log_training_progress(self, episode: int, reward: float, loss: float = 0.0):
        """Log episode training metrics"""
        self.episode_rewards.append(reward)
        if loss > 0:
            self.episode_losses.append(loss)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_loss = np.mean(self.episode_losses[-100:]) if self.episode_losses else 0.0
            print(f"[{self.agent_name}] Episode {episode+1} | Avg Reward: {avg_reward:.2f} | Loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint and training state"""
        checkpoint_path = MODELS_DIR / f"{self.agent_name}_episode_{episode}.pt"
        checkpoint = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def save_summary(self):
        """Save training summary to JSON"""
        summary = {
            'agent_name': self.agent_name,
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'avg_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
            'min_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0,
            'avg_loss': float(np.mean(self.episode_losses)) if self.episode_losses else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = RESULTS_DIR / f"{self.agent_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}")
        
        return summary
