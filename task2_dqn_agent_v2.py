"""
Task 2: Deep Q-Network (DQN) Agent with Sparse/Delayed Rewards
Complete implementation with prioritized experience replay and target networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random

from task2_base_agent_v2 import (
    BaseRLAgent, StateRepresentation, ActionHandler, 
    RewardHandler, ReplayBuffer, Experience
)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for estimating state-action values.
    
    Architecture:
    - Fully connected layers with ReLU activation
    - Dropout for regularization
    - Output: Q-values for each action
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 hidden_sizes: List[int] = None, dropout: float = 0.2):
        super(DQNNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        
        layers = []
        prev_size = state_size
        
        # Build hidden layers with dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output Q-values (no activation)
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> Q-values
        
        Args:
            state: [batch_size, state_size]
            
        Returns:
            q_values: [batch_size, action_size]
        """
        return self.network(state)


class DQNAgent(BaseRLAgent):
    """
    Deep Q-Network Agent for Chef's Hat with Sparse/Delayed Rewards.
    
    Key features:
    - Experience replay with prioritized sampling
    - Target network for training stability
    - Double DQN for overestimation reduction
    - Epsilon-greedy exploration
    - Action masking for invalid actions
    - Sparse reward handling
    """
    
    def __init__(self,
                 agent_name: str = "DQN_Agent",
                 num_players: int = 4,
                 state_size: int = 50,
                 action_size: int = 99,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 replay_buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: str = "cpu"):
        """
        Initialize DQN Agent
        
        Args:
            state_size: State representation dimension
            action_size: Number of possible actions
            epsilon_decay: Decay rate for exploration
            target_update_freq: Steps between target network updates
        """
        super().__init__(agent_name, num_players, learning_rate, gamma, device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # No gradients needed for target
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Update counter
        self.update_counter = 0
        
        print(f"✓ {agent_name} initialized")
        print(f"  State size: {state_size}, Action size: {action_size}")
        print(f"  Gamma: {gamma}, Learning rate: {learning_rate}")
        print(f"  Epsilon: {self.epsilon:.3f} -> {epsilon_end:.3f}")
    
    def act(self, observation: Any, valid_actions: Optional[np.ndarray] = None,
            training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Sparse/delayed reward handling: Same exploration regardless of reward type
        since we're learning from delayed/sparse signals.
        """
        # Encode observation
        state = self.state_encoder.encode_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        if valid_actions is None:
            valid_actions = self.action_handler.get_valid_actions(observation)
        
        # Epsilon-greedy with action masking
        if training and np.random.random() < self.epsilon:
            # Explore: random valid action
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) == 0:
                action = 0
            else:
                action = np.random.choice(valid_indices)
        else:
            # Exploit: best valid action
            action = self.action_handler.select_action(q_values, valid_actions, epsilon=0.0)
        
        self.total_steps += 1
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool, 
                 valid_actions: Optional[np.ndarray] = None):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done, valid_actions)
    
    def learn(self, batch_size: int = None) -> float:
        """
        Update Q-network from experience replay.
        
        Sparse reward handling:
        - Learn from both sparse terminal rewards and shaped intermediate rewards
        - Double DQN to reduce overestimation of rare high rewards
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch using prioritized sampling
        experiences, indices = self.replay_buffer.sample(batch_size, use_priority=True)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values (Double DQN for sparse rewards)
        with torch.no_grad():
            # Use main network to select action
            next_q_main = self.q_network(next_states)
            next_actions = torch.argmax(next_q_main, dim=1)
            
            # Use target network to evaluate action
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Bellman target with sparse reward
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss and optimization
        loss = nn.MSELoss()(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities based on TD error
        td_errors = (q_values - targets).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return float(loss.item())
    
    def train_episode(self, env, max_steps: int = 500) -> Tuple[float, float]:
        """
        Train for one episode.
        
        Handles sparse/delayed rewards from environment.
        """
        observation = env.reset()
        episode_reward = 0.0
        losses = []
        
        for step in range(max_steps):
            # Get valid actions from environment
            valid_actions = None  # Will be determined by action_handler
            
            # Select action
            action = self.act(observation, valid_actions, training=True)
            
            # Step environment
            try:
                step_result = env.step(action)
                if len(step_result) == 4:
                    next_observation, reward, done, info = step_result
                else:
                    next_observation, reward, done, truncated, info = step_result
                    done = done or truncated
            except Exception as e:
                print(f"Error during step: {e}")
                break
            
            # Process reward with shaping for sparse rewards
            processed_reward = self.reward_handler.process_reward(reward, done, info, step)
            episode_reward += processed_reward
            
            # Encode states for storage
            state = self.state_encoder.encode_observation(observation)
            next_state = self.state_encoder.encode_observation(next_observation)
            
            # Store in replay buffer
            self.remember(state, action, processed_reward, next_state, done, valid_actions)
            
            # Learn from replay buffer
            if len(self.replay_buffer) > self.batch_size:
                loss = self.learn(self.batch_size)
                losses.append(loss)
            
            observation = next_observation
            
            if done:
                break
        
        avg_loss = np.mean(losses) if losses else 0.0
        return episode_reward, avg_loss
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        summary = super().save_summary()
        summary.update({
            'algorithm': 'DQN',
            'epsilon': float(self.epsilon),
            'buffer_size': len(self.replay_buffer),
            'updates': self.update_counter
        })
        return summary
