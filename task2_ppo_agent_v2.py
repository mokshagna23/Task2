"""
Task 2: Proximal Policy Optimization (PPO) Agent with Sparse/Delayed Rewards
On-policy learning with advantage estimation for delayed rewards
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import random

from task2_base_agent_v2 import BaseRLAgent, StateRepresentation, ActionHandler, RewardHandler


class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Outputs:
    - Policy: probability distribution over actions
    - Value: estimated state value for advantage computation
    """
    
    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: List[int] = None, dropout: float = 0.2):
        super(PPONetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
        # Shared feature extraction
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: features -> (policy, value)
        
        Args:
            state: [batch_size, state_size]
            
        Returns:
            policy: [batch_size, action_size] - action probabilities
            value: [batch_size, 1] - state value estimate
        """
        features = self.feature_layers(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization (PPO) Agent for Chef's Hat.
    
    On-policy learning with:
    - Clipped objective for stable updates
    - Critic for advantage estimation
    - Entropy regularization for exploration
    - Generalized Advantage Estimation (GAE) for value estimates
    
    Well-suited for sparse rewards due to on-policy nature and
    advantage normalization.
    """
    
    def __init__(self,
                 agent_name: str = "PPO_Agent",
                 num_players: int = 4,
                 state_size: int = 50,
                 action_size: int = 99,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 batch_size: int = 64,
                 epochs_per_update: int = 3,
                 device: str = "cpu"):
        """
        Initialize PPO Agent
        
        Args:
            gae_lambda: GAE parameter for advantage computation
            clip_ratio: PPO clipping parameter
            entropy_coeff: Weight for entropy regularization
            value_loss_coeff: Weight for critic loss
            epochs_per_update: Training epochs per batch collection
        """
        super().__init__(agent_name, num_players, learning_rate, gamma, device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        
        # Actor-Critic network
        self.network = PPONetwork(state_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Trajectory buffer for one episode
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        print(f"✓ {agent_name} initialized")
        print(f"  State size: {state_size}, Action size: {action_size}")
        print(f"  Gamma: {gamma}, Lambda: {gae_lambda}, Clip: {clip_ratio:.3f}")
    
    def act(self, observation: Any, valid_actions: Optional[np.ndarray] = None,
            training: bool = True) -> int:
        """
        Select action from policy distribution.
        
        For sparse rewards: On-policy nature helps with delayed signals.
        """
        # Encode state
        state = self.state_encoder.encode_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get policy and value
        with torch.no_grad():
            policy, value = self.network(state_tensor)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        # Mask invalid actions and renormalize
        if valid_actions is None:
            valid_actions = self.action_handler.get_valid_actions(observation)
        
        policy = policy * valid_actions
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform over valid actions
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) == 0:
                return 0
            policy = np.zeros(self.action_size)
            policy[valid_indices] = 1.0 / len(valid_indices)
        
        # Sample action from policy (training) or greedy (evaluation)
        if training:
            action = np.random.choice(self.action_size, p=policy)
        else:
            action = np.argmax(policy)
        
        # Store for trajectory
        log_prob = np.log(policy[action] + 1e-8)
        
        self.total_steps += 1
        
        return action, log_prob, value
    
    def remember_transition(self, state: np.ndarray, action: int, 
                           reward: float, value: float, log_prob: float, done: bool):
        """Store transition in trajectory buffer"""
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['values'].append(value)
        self.trajectory['log_probs'].append(log_prob)
        self.trajectory['dones'].append(done)
    
    def compute_gae_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Important for sparse rewards:
        - Stabilizes advantage estimates
        - Reduces variance in credit assignment
        """
        rewards = np.array(self.trajectory['rewards'])
        values = np.array(self.trajectory['values'])
        dones = np.array(self.trajectory['dones'])
        
        nsteps = len(rewards)
        advantages = np.zeros(nsteps)
        returns = np.zeros(nsteps)
        
        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            # TD residual
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages (important for sparse rewards)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self) -> float:
        """
        Update actor and critic networks.
        
        Uses PPO objective with clipped surrogate loss.
        """
        if len(self.trajectory['states']) == 0:
            return 0.0
        
        # Convert trajectory to tensors
        states = torch.FloatTensor(np.array(self.trajectory['states'])).to(self.device)
        actions = torch.LongTensor(np.array(self.trajectory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.trajectory['log_probs'])).to(self.device)
        
        advantages, returns = self.compute_gae_advantages()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training epochs
        total_loss = 0.0
        for epoch in range(self.epochs_per_update):
            # Forward pass
            policies, values = self.network(states)
            values = values.squeeze()
            
            # New log probabilities
            log_probs = torch.log(
                policies.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8
            )
            
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (MSE)
            critic_loss = self.value_loss_coeff * nn.MSELoss()(values, returns)
            
            # Entropy regularization for exploration
            entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()
            entropy_loss = -self.entropy_coeff * entropy
            
            # Total loss
            loss = actor_loss + critic_loss + entropy_loss
            total_loss += loss.item()
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear trajectory buffer
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        return total_loss / self.epochs_per_update
    
    def train_episode(self, env, max_steps: int = 500) -> Tuple[float, float]:
        """
        Train for one episode collecting full trajectory.
        
        On-policy: Update from complete episode.
        """
        observation = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = self.act(observation, training=True)
            
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
            
            # Encode state for storage
            state = self.state_encoder.encode_observation(observation)
            
            # Remember transition
            self.remember_transition(state, action, processed_reward, value, log_prob, done)
            
            observation = next_observation
            
            if done:
                break
        
        # Update policy from collected trajectory
        loss = self.update_policy()
        
        return episode_reward, loss
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        summary = super().save_summary()
        summary.update({
            'algorithm': 'PPO',
            'clip_ratio': float(self.clip_ratio),
            'gae_lambda': float(self.gae_lambda),
            'entropy_coeff': float(self.entropy_coeff)
        })
        return summary
