"""
Reinforcement Learning Agents for Adaptive Trading

This module implements state-of-the-art RL algorithms optimized for financial
trading environments, including policy-based and actor-critic methods.

Agents:
- PPO: Proximal Policy Optimization for stable policy updates
- A3C: Asynchronous Actor-Critic for distributed learning
- SAC: Soft Actor-Critic for continuous action spaces
- DQN: Deep Q-Network for discrete trading actions
- TD3: Twin Delayed DDPG for robust continuous control
"""

from .ppo_agent import PPOTradingAgent
from .a3c_agent import A3CTradingAgent
from .sac_agent import SACTradingAgent
from .dqn_agent import DQNTradingAgent
from .td3_agent import TD3TradingAgent
from .trading_environment import TradingEnvironment
from .base_agent import BaseRLAgent

__all__ = [
    'PPOTradingAgent',
    'A3CTradingAgent', 
    'SACTradingAgent',
    'DQNTradingAgent',
    'TD3TradingAgent',
    'TradingEnvironment',
    'BaseRLAgent',
]
