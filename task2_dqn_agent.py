"""
Task 2: Deep Q-Network (DQN) Agent for Chef's Hat Gym
Implements DQN algorithm with experience replay and target networks.
Focus: Sparse/Delayed Reward Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

from task2_base_agent import (
    BaseRLAgent, StateRepresentation, ActionHandler, 
    RewardHandler, ReplayBuffer, Experience
)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for estimating action values.
    
    Architecture:
    - Input: State representation (concatenated features)
    - Hidden layers with ReLU activation
    - Output: Q-values for each action
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
        """
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
            
        layers = []
        prev_size = state_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        # Output layer (no activation, produces Q-values)
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Batch of states [batch_size, state_size]
            
        Returns:
            Q-values [batch_size, action_size]
        """
        return self.network(state)


class DQNAgent(BaseRLAgent):
    """
    Deep Q-Network Agent for Chef's Hat.
    
    Implements:
    - Experience replay for off-policy learning
    - Target network for stability
    - Epsilon-greedy exploration
    - Action masking for invalid actions
    """
    
    def __init__(self,
                 agent_name: str = "DQN_Agent",
                 num_players: int = 4,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 replay_buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: str = "cpu",
                 log_directory: Optional[str] = None):
        """
        Args:
            agent_name: Name for logging
            num_players: Number of players
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            replay_buffer_size: Capacity of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Update target network every N steps
            device: 'cpu' or 'cuda'
            log_directory: Directory for logs
        """
        super().__init__(agent_name, num_players, log_directory)
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Network parameters
        self.state_size = 47  # Size of encoded state vector
        self.action_size = 11  # Max 10 cards + 1 pass action
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        
        # Networks
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training tracking
        self.target_update_freq = target_update_freq
        self.steps = 0
        self.episodes = 0
        self.train_losses = deque(maxlen=1000)
        self.episode_returns = deque(maxlen=100)
        
    def pick_best_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """
        Select action with highest Q-value (exploitation only).
        
        Args:
            state: Current state vector
            valid_actions: Mask of valid actions
            
        Returns:
            Action index
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            
            # Mask invalid actions (set to very low value)
            q_values[~valid_actions] = -np.inf
            
            return np.argmax(q_values)
            
    def pick_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = None) -> int:
        """
        Select action with epsilon-greedy strategy.
        
        Args:
            state: Current state vector
            valid_actions: Mask of valid actions
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Action index
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions)[0]
            return np.random.choice(valid_indices)
        else:
            # Best action
            return self.pick_best_action(state, valid_actions)
            
    def store_experience(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        valid_actions: np.ndarray):
        """Store experience in replay buffer."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            valid_actions=valid_actions
        )
        self.replay_buffer.push(experience)
        
    def train_step(self):
        """
        Perform one training step using mini-batch from replay buffer.
        Implements DQN loss: MSE(Q(s,a), r + gamma * max_a' Q_target(s', a'))
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences yet
            
        # Sample mini-batch
        states, actions, rewards, next_states, dones, valid_actions_list = \
            self.replay_buffer.sample(self.batch_size)
            
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s, a) - Q-values of taken actions
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Compute max Q'(s', a') using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t)
            
            # Apply action masking for next states
            for i, valid_mask in enumerate(valid_actions_list):
                next_q_values[i, ~valid_mask] = -np.inf
                
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            
        # Compute target: r + gamma * max Q'(s', a') * (1 - done)
        target_q_values = rewards_t + self.gamma * max_next_q_values * (1 - dones_t)
        
        # MSE Loss
        loss = self.criterion(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track training metrics
        self.train_losses.append(loss.item())
        self.steps += 1
        
        # Periodically update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(list(self.train_losses)) if self.train_losses else 0,
            'avg_episode_return': np.mean(list(self.episode_returns)) if self.episode_returns else 0,
        }
        
    def save(self, filepath: str):
        """Save agent weights."""
        torch.save(self.q_network.state_dict(), filepath)
        
    def load(self, filepath: str):
        """Load agent weights."""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())


# Random Agent baseline for comparison
class RandomAgent(BaseRLAgent):
    """Simple random agent for baseline comparison."""
    
    def pick_best_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Always return random valid action."""
        valid_indices = np.where(valid_actions)[0]
        return np.random.choice(valid_indices) if len(valid_indices) > 0 else self.action_size - 1
        
    def pick_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = None) -> int:
        """Return random valid action."""
        return self.pick_best_action(state, valid_actions)
        
    def train_step(self):
        """No training needed."""
        pass


# Greedy Agent (baseline)
class GreedyAgent(BaseRLAgent):
    """
    Greedy agent that plays highest card when possible.
    Good baseline for comparison.
    """
    
    def pick_best_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Play highest valid card."""
        valid_indices = np.where(valid_actions)[0]
        
        if len(valid_indices) == 0:
            return self.action_size - 1  # Pass
            
        # Greedy heuristic: prefer playing high cards to reduce hand
        # In this simplified version, just pick first valid action that's not pass
        non_pass_actions = [a for a in valid_indices if a < self.action_size - 1]
        
        if non_pass_actions:
            return max(non_pass_actions)  # Play highest card index
        else:
            return self.action_size - 1  # Pass
            
    def pick_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = None) -> int:
        """Return greedy action."""
        return self.pick_best_action(state, valid_actions)
        
    def train_step(self):
        """No training needed."""
        pass
