"""
Base Reinforcement Learning Agent

This module provides the foundation class for all RL agents,
including common functionality for training, evaluation, and integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gymnasium as gym

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class BaseRLAgent(ABC):
    """
    Base class for all reinforcement learning agents.
    
    Provides common functionality for training, evaluation, experience replay,
    and integration with the HFT trading system.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 device: Optional[str] = None):
        """
        Initialize base RL agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            device: Device to run on (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training state
        self.is_trained = False
        self.training_episode = 0
        self.total_steps = 0
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'exploration_rates': []
        }
        
        # Agent metadata
        self.agent_config = {
            'agent_type': self.__class__.__name__,
            'state_size': state_size,
            'action_size': action_size,
            'lr': lr,
            'gamma': gamma,
            'created_at': datetime.now().isoformat(),
        }
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Select action given current state."""
        pass
    
    @abstractmethod
    def learn(self, experiences: List[Experience]) -> Dict[str, float]:
        """Learn from batch of experiences."""
        pass
    
    @abstractmethod
    def get_networks(self) -> Dict[str, nn.Module]:
        """Return dictionary of neural networks used by agent."""
        pass
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess state for neural network input.
        
        Args:
            state: Raw state array
            
        Returns:
            Preprocessed state tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        return state
    
    def update_training_history(self, 
                               episode_reward: float,
                               episode_length: int,
                               loss: Optional[float] = None,
                               exploration_rate: Optional[float] = None):
        """Update training history with episode results."""
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        
        if loss is not None:
            self.training_history['losses'].append(loss)
        
        if exploration_rate is not None:
            self.training_history['exploration_rates'].append(exploration_rate)
    
    def train_agent(self,
                   environment,
                   num_episodes: int = 1000,
                   max_steps: int = 1000,
                   evaluation_freq: int = 100,
                   save_freq: int = 500,
                   target_reward: Optional[float] = None) -> Dict[str, List[float]]:
        """
        Train the RL agent in the given environment.
        
        Args:
            environment: Trading environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            evaluation_freq: Episodes between evaluations
            save_freq: Episodes between model saves
            target_reward: Target reward to stop training
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            self.training_episode = episode
            
            # Reset environment
            state, _ = environment.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action
                action = self.act(state, training=True)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
                
                # Store experience (if agent uses experience replay)
                if hasattr(self, 'store_experience'):
                    self.store_experience(state, action, reward, next_state, done)
                
                # Learn from experience
                if hasattr(self, 'should_learn') and self.should_learn():
                    loss_info = self.learn_step()
                    if loss_info and 'loss' in loss_info:
                        self.training_history['losses'].append(loss_info['loss'])
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                if done:
                    break
            
            # Update training history
            exploration_rate = getattr(self, 'epsilon', None) or getattr(self, 'exploration_rate', None)
            self.update_training_history(episode_reward, episode_length, 
                                       exploration_rate=exploration_rate)
            
            # Track best performance
            if episode_reward > best_reward:
                best_reward = episode_reward
                if hasattr(self, 'save_checkpoint'):
                    self.save_checkpoint('best_model.pth')
            
            # Evaluation
            if episode % evaluation_freq == 0:
                eval_reward = self.evaluate_agent(environment, num_episodes=5)
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Eval Reward={eval_reward:.2f}, Best={best_reward:.2f}")
                
                # Check for early stopping
                if target_reward and eval_reward >= target_reward:
                    logger.info(f"Target reward {target_reward} reached!")
                    break
            
            # Save model
            if episode % save_freq == 0:
                self.save_agent(f'checkpoint_episode_{episode}.pth')
        
        self.is_trained = True
        logger.info("Training completed successfully")
        
        return self.training_history
    
    def evaluate_agent(self, 
                      environment,
                      num_episodes: int = 10,
                      render: bool = False) -> float:
        """
        Evaluate agent performance.
        
        Args:
            environment: Trading environment
            num_episodes: Number of evaluation episodes
            render: Whether to render environment
            
        Returns:
            Average episode reward
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = environment.reset()
            episode_reward = 0
            
            done = False
            while not done:
                # Select action (no exploration)
                action = self.act(state, training=False)
                
                # Take action
                state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if render:
                    environment.render()
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        logger.info(f"Evaluation: Avg Reward={avg_reward:.2f} ± {std_reward:.2f}")
        
        return avg_reward
    
    def save_agent(self, filepath: str):
        """Save agent state to file."""
        save_dict = {
            'agent_config': self.agent_config,
            'training_history': self.training_history,
            'training_episode': self.training_episode,
            'total_steps': self.total_steps,
            'is_trained': self.is_trained,
        }
        
        # Save network states
        networks = self.get_networks()
        network_states = {}
        for name, network in networks.items():
            network_states[f'{name}_state_dict'] = network.state_dict()
        
        save_dict.update(network_states)
        
        # Save additional agent-specific state
        if hasattr(self, 'get_agent_state'):
            save_dict.update(self.get_agent_state())
        
        torch.save(save_dict, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load agent state from file."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Load basic config
        self.agent_config = save_dict.get('agent_config', {})
        self.training_history = save_dict.get('training_history', {})
        self.training_episode = save_dict.get('training_episode', 0)
        self.total_steps = save_dict.get('total_steps', 0)
        self.is_trained = save_dict.get('is_trained', False)
        
        # Load network states
        networks = self.get_networks()
        for name, network in networks.items():
            state_key = f'{name}_state_dict'
            if state_key in save_dict:
                network.load_state_dict(save_dict[state_key])
        
        # Load agent-specific state
        if hasattr(self, 'load_agent_state'):
            self.load_agent_state(save_dict)
        
        logger.info(f"Agent loaded from {filepath}")
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress."""
        if not self.training_history['episode_rewards']:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        # Losses (if available)
        if self.training_history['losses']:
            axes[1, 0].plot(self.training_history['losses'])
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].grid(True)
        
        # Exploration rates (if available)
        if self.training_history['exploration_rates']:
            axes[1, 1].plot(self.training_history['exploration_rates'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Exploration Rate')
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training progress plot saved to {save_path}")
        else:
            plt.show()
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent summary."""
        networks = self.get_networks()
        total_params = sum(sum(p.numel() for p in net.parameters()) for net in networks.values())
        trainable_params = sum(sum(p.numel() for p in net.parameters() if p.requires_grad) 
                             for net in networks.values())
        
        summary = {
            'agent_type': self.__class__.__name__,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_episode': self.training_episode,
            'total_steps': self.total_steps,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'config': self.agent_config,
            'networks': list(networks.keys()),
        }
        
        if self.training_history['episode_rewards']:
            summary['performance'] = {
                'avg_reward': np.mean(self.training_history['episode_rewards'][-100:]),
                'best_reward': max(self.training_history['episode_rewards']),
                'total_episodes': len(self.training_history['episode_rewards']),
            }
        
        return summary


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: Union[int, np.ndarray], 
            reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
