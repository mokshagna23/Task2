"""
Task 2: Proximal Policy Optimization (PPO) Agent for Chef's Hat Gym
Implements on-policy PPO algorithm with action masking.
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
    RewardHandler, Experience
)


class PPOActor(nn.Module):
    """Actor network for PPO - outputs action probabilities."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
        super(PPOActor, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
            
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.network(state)


class PPOCritic(nn.Module):
    """Critic network for PPO - estimates state value."""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = None):
        super(PPOCritic, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
            
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value."""
        return self.network(state).squeeze(-1)


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization Agent for Chef's Hat.
    
    Implements:
    - Policy gradient with advantage estimation
    - Clipped surrogate objective
    - Value function for baseline
    - Action masking for valid actions only
    """
    
    def __init__(self,
                 agent_name: str = "PPO_Agent",
                 num_players: int = 4,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 epsilon_clip: float = 0.2,
                 c1_value_loss: float = 0.5,
                 c2_entropy: float = 0.01,
                 batch_size: int = 64,
                 num_epochs: int = 3,
                 device: str = "cpu",
                 log_directory: Optional[str] = None):
        """
        Args:
            agent_name: Name for logging
            num_players: Number of players
            learning_rate: Learning rate for both actor and critic
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            epsilon_clip: PPO clipping parameter
            c1_value_loss: Weight for value loss
            c2_entropy: Weight for entropy bonus
            batch_size: Batch size for training
            num_epochs: Number of update epochs per batch
            device: 'cpu' or 'cuda'
            log_directory: Directory for logs
        """
        super().__init__(agent_name, num_players, log_directory)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Network parameters
        self.state_size = 47  # Same as DQN
        self.action_size = 11
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.c1_value_loss = c1_value_loss
        self.c2_entropy = c2_entropy
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Networks
        self.actor = PPOActor(self.state_size, self.action_size).to(self.device)
        self.critic = PPOCritic(self.state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Storage for experience collection
        self.trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'valid_actions': []
        }
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        self.train_losses = deque(maxlen=1000)
        self.episode_returns = deque(maxlen=100)
        self.episode_length = 0
        self.episode_reward = 0
        
    def pick_best_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """
        Select action with highest policy probability (exploitation).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor).squeeze(0)
            
            # Mask invalid actions
            logits[~valid_actions] = -np.inf
            
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()
            return np.argmax(probabilities)
            
    def pick_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = None) -> int:
        """
        Sample action from policy distribution.
        
        Always uses policy sampling for PPO (no epsilon-greedy needed).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor).squeeze(0)
            
            # Apply masking: set invalid actions to very negative logits
            logits_masked = logits.clone()
            logits_masked[~valid_actions] = -1e8
            
            # Sample from policy
            probabilities = torch.softmax(logits_masked, dim=0)
            action = torch.multinomial(probabilities, 1).item()
            
            return action
            
    def store_transition(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        value: float,
                        log_prob: float,
                        done: bool,
                        valid_actions: np.ndarray):
        """Store transition in trajectory buffer."""
        self.trajectories['states'].append(state)
        self.trajectories['actions'].append(action)
        self.trajectories['rewards'].append(reward)
        self.trajectories['values'].append(value)
        self.trajectories['log_probs'].append(log_prob)
        self.trajectories['dones'].append(done)
        self.trajectories['valid_actions'].append(valid_actions)
        
        self.episode_length += 1
        self.episode_reward += reward
        
    def compute_advantages(self, next_value: float = 0.0) -> np.ndarray:
        """
        Compute generalized advantage estimation (GAE).
        
        Args:
            next_value: Value estimate of next state (0 if terminal)
            
        Returns:
            Array of advantage estimates
        """
        rewards = np.array(self.trajectories['rewards'])
        values = np.array(self.trajectories['values'] + [next_value])
        dones = np.array(self.trajectories['dones'])
        
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        return advantages
        
    def train_step(self):
        """
        Perform training on collected trajectories.
        Implements PPO update with clipped surrogate objective.
        """
        if len(self.trajectories['states']) == 0:
            return
            
        # Compute advantages
        advantages = self.compute_advantages()
        returns = advantages + np.array(self.trajectories['values'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.trajectories['states'])).to(self.device)
        actions_t = torch.LongTensor(np.array(self.trajectories['actions'])).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(self.trajectories['log_probs'])).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # PPO training epochs
        num_samples = len(self.trajectories['states'])
        indices = np.arange(num_samples)
        
        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                states_batch = states_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                old_log_probs_batch = old_log_probs_t[batch_indices]
                advantages_batch = advantages_t[batch_indices]
                returns_batch = returns_t[batch_indices]
                
                # Actor update
                logits = self.actor(states_batch)
                probabilities = torch.softmax(logits, dim=1)
                log_probs = torch.log_softmax(logits, dim=1)
                
                # Gather log probs of taken actions
                log_probs_taken = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
                
                # PPO clipped surrogate objective
                ratio = torch.exp(log_probs_taken - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus for exploration
                entropy = -(probabilities * log_probs).sum(dim=1).mean()
                
                # Critic update
                values = self.critic(states_batch).squeeze(1)
                value_loss = ((returns_batch - values) ** 2).mean()
                
                # Total loss
                total_loss = actor_loss + self.c1_value_loss * value_loss - self.c2_entropy * entropy
                
                # Update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                self.train_losses.append(total_loss.item())
                
        # Clear trajectories for next episode
        self.trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'valid_actions': []
        }
        
        self.episodes += 1
        self.episode_returns.append(self.episode_reward)
        self.episode_length = 0
        self.episode_reward = 0
        self.steps += len(states_t)
        
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor).squeeze().item()
            return value
            
    def get_log_prob(self, state: np.ndarray, action: int, valid_actions: np.ndarray) -> float:
        """Get log probability of action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor).squeeze(0)
            
            # Mask invalid actions
            logits_masked = logits.clone()
            logits_masked[~valid_actions] = -1e8
            
            log_probs = torch.log_softmax(logits_masked, dim=0)
            log_prob = log_probs[action].item()
            return log_prob
            
    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'avg_loss': np.mean(list(self.train_losses)) if self.train_losses else 0,
            'avg_episode_return': np.mean(list(self.episode_returns)) if self.episode_returns else 0,
        }
        
    def save(self, actor_path: str, critic_path: str):
        """Save agent networks."""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load(self, actor_path: str, critic_path: str):
        """Load agent networks."""
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
