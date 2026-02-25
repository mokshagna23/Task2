"""
Task 2: Reinforcement Learning Agent for Chef's Hat Gym
Focus: Sparse/Delayed Reward Variant

This module contains the base RL agent class and state/action representation handlers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from collections import namedtuple, deque


class StateRepresentation:
    """
    Handles state representation for Chef's Hat environment.
    
    State includes:
    - Player's hand (cards)
    - Players' scores
    - Current player index
    - Table cards (discarded)
    - Game status
    """
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.max_hand_size = 10  # Chef's Hat max hand size
        self.num_card_types = 13  # Standard deck: A, 2-10, J, Q, K
        
    def encode_hand(self, hand: List[int]) -> np.ndarray:
        """
        Encode player's hand as a histogram of card types.
        
        Args:
            hand: List of card values
            
        Returns:
            numpy array of shape (13,) representing card distribution
        """
        encoded = np.zeros(self.num_card_types)
        for card in hand:
            if 0 <= card < self.num_card_types:
                encoded[card] += 1
        return encoded / np.linalg.norm(encoded + 1e-8)  # Normalize
        
    def encode_game_state(self, 
                         hand: List[int],
                         scores: List[int],
                         current_player: int,
                         table_cards: List[int],
                         round_number: int,
                         game_status: Dict) -> np.ndarray:
        """
        Encode complete game state into a feature vector.
        
        Args:
            hand: Player's hand
            scores: All players' scores
            current_player: Index of current player
            table_cards: Cards on table
            round_number: Current round (game phase)
            game_status: Dict with additional game info
            
        Returns:
            Flattened feature vector
        """
        features = []
        
        # Hand representation (13 features)
        hand_encoded = self.encode_hand(hand)
        features.append(hand_encoded)
        
        # Scores (normalized to [0, 1])
        scores_normalized = np.array(scores) / (np.max(scores) + 1)
        features.append(scores_normalized)
        
        # One-hot current player (4 features for 4 players)
        current_player_onehot = np.zeros(self.num_players)
        current_player_onehot[current_player] = 1
        features.append(current_player_onehot)
        
        # Table cards distribution (13 features)
        table_encoded = self.encode_hand(table_cards)
        features.append(table_encoded)
        
        # Hand size normalized (1 feature)
        hand_size = np.array([len(hand) / self.max_hand_size])
        features.append(hand_size)
        
        # Round number normalized (1 feature)
        round_normalized = np.array([round_number / 4.0])  # 4 rounds per game
        features.append(round_normalized)
        
        return np.concatenate(features).astype(np.float32)


class ActionHandler:
    """
    Handles action space and action selection strategies.
    
    Chef's Hat has variable action space:
    - Play single card or pass
    - Invalid actions must be masked
    """
    
    def __init__(self, max_hand_size: int = 10):
        self.max_hand_size = max_hand_size
        
    def get_valid_actions(self, 
                         hand: List[int],
                         table_cards: List[int],
                         current_round: int) -> np.ndarray:
        """
        Determine valid actions in current state.
        
        Valid actions:
        - Pass (always valid)
        - Play card with value > highest table card
        
        Args:
            hand: Player's current hand
            table_cards: Cards on table
            current_round: Current round number
            
        Returns:
            Boolean mask of valid actions (size MAX_HAND_SIZE + 1)
            Last action is 'pass'
        """
        valid_mask = np.zeros(self.max_hand_size + 1, dtype=bool)
        
        # Pass is always an option
        valid_mask[-1] = True
        
        if len(table_cards) == 0:
            # If table is empty, any card can be played
            for i in range(min(len(hand), self.max_hand_size)):
                valid_mask[i] = True
        else:
            # Only cards higher than max table card are valid
            max_table = max(table_cards)
            for i, card in enumerate(hand[:self.max_hand_size]):
                if card > max_table:
                    valid_mask[i] = True
                    
        return valid_mask
        
    def action_to_card(self, action: int, hand: List[int]) -> Optional[int]:
        """
        Convert action index to card to play.
        
        Args:
            action: Action index
            hand: Player's hand
            
        Returns:
            Card value to play, or None for pass
        """
        if action == len(hand) or action == self.max_hand_size:
            return None  # Pass
        elif action < len(hand):
            return hand[action]
        else:
            return None


class RewardHandler:
    """
    Implements reward strategy for sparse/delayed reward variant.
    
    Sparse rewards: Only terminal (match end) rewards
    """
    
    def __init__(self, use_shaped_rewards: bool = False):
        """
        Args:
            use_shaped_rewards: If True, add small shaping rewards
                               If False, strictly sparse rewards
        """
        self.use_shaped_rewards = use_shaped_rewards
        
    def compute_reward(self,
                      match_info: Dict,
                      player_idx: int,
                      is_terminal: bool) -> float:
        """
        Compute reward based on game outcome.
        
        Sparse Reward Logic:
        - Winner (1st place): +1.0
        - 2nd place: +0.0
        - 3rd place: -0.5
        - 4th place: -1.0
        
        Args:
            match_info: Dict containing match results
            player_idx: Index of player
            is_terminal: Whether match has ended
            
        Returns:
            Reward value
        """
        if not is_terminal:
            return 0.0  # No reward until match end
            
        # Get player's final position
        player_scores = match_info.get('scores', [])
        if not player_scores or player_idx >= len(player_scores):
            return 0.0
            
        # Calculate placement (lower score is better in Chef's Hat)
        player_score = player_scores[player_idx]
        positions = sorted(enumerate(player_scores), key=lambda x: x[1])
        placement = next(i for i, (idx, _) in enumerate(positions) if idx == player_idx) + 1
        
        # Assign rewards based on placement
        placement_rewards = {
            1: 1.0,    # First place
            2: 0.0,    # Second place (no reward)
            3: -0.5,   # Third place
            4: -1.0    # Fourth place
        }
        
        return placement_rewards.get(placement, 0.0)
        
    def get_shaped_reward(self,
                        prev_hand_size: int,
                        curr_hand_size: int,
                        player_position: int) -> float:
        """
        Get reward shaping for better learning (optional).
        
        Small rewards for:
        - Playing cards (reducing hand size)
        - Winning rounds
        
        Args:
            prev_hand_size: Hand size before action
            curr_hand_size: Hand size after action
            player_position: Player's current position
            
        Returns:
            Small shaping reward
        """
        if not self.use_shaped_rewards:
            return 0.0
            
        reward = 0.0
        
        # Small reward for reducing hand size (progress)
        if curr_hand_size < prev_hand_size:
            reward += 0.01 * (prev_hand_size - curr_hand_size)
            
        return reward


# Experience replay structure
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'valid_actions'])


class ReplayBuffer:
    """Experience replay buffer for offline learning from sparse rewards."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.memory.append(experience)
        
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        """Sample random batch from buffer."""
        import random
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        valid_actions_list = [e.valid_actions for e in batch]
        
        return states, actions, rewards, next_states, dones, valid_actions_list
        
    def __len__(self):
        return len(self.memory)


class BaseRLAgent(ABC):
    """Base class for RL agents."""
    
    def __init__(self, 
                 agent_name: str,
                 num_players: int = 4,
                 log_directory: Optional[str] = None):
        """
        Args:
            agent_name: Name of agent
            num_players: Number of players in game
            log_directory: Directory for logging
        """
        self.agent_name = agent_name
        self.num_players = num_players
        self.log_directory = log_directory
        
        # Initialize components
        self.state_rep = StateRepresentation(num_players)
        self.action_handler = ActionHandler()
        self.reward_handler = RewardHandler(use_shaped_rewards=False)  # Sparse rewards
        
    @abstractmethod
    def pick_best_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Select best action given state (exploitation)."""
        pass
        
    @abstractmethod
    def pick_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action with exploration/exploitation trade-off."""
        pass
        
    @abstractmethod
    def train_step(self):
        """Perform training step."""
        pass
