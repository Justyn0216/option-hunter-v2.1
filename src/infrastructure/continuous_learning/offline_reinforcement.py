"""
Offline Reinforcement Learning Module

This module provides offline reinforcement learning capabilities for the Option Hunter system.
It allows training RL agents on historical trade data without direct market interaction,
enabling safer exploration and optimization of trading strategies.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue
from collections import deque, defaultdict
import random
import pickle
import copy
import time
import multiprocessing

# Try importing RL libraries with graceful fallback
try:
    import gym
    from gym import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from stable_baselines3 import DQN, PPO, A2C, SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for option trading based on historical data.
    """
    
    def __init__(self, historical_data, features, config=None):
        """
        Initialize the trading environment.
        
        Args:
            historical_data (pd.DataFrame): Historical market and option data
            features (list): Feature columns to use as state representation
            config (dict, optional): Environment configuration
        """
        super(TradingEnvironment, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Store historical data
        self.historical_data = historical_data
        self.features = features
        
        # Trading parameters
        self.starting_balance = self.config.get("starting_balance", 100000.0)
        self.transaction_cost = self.config.get("transaction_cost", 0.01)  # 1% of trade value
        self.max_position_size = self.config.get("max_position_size", 10)  # Max number of contracts
        self.hold_penalty = self.config.get("hold_penalty", 0.0001)  # Small penalty for holding positions
        self.done_on_zero_balance = self.config.get("done_on_zero_balance", True)
        self.episode_length = self.config.get("episode_length", 252)  # Trading days in a year
        self.reward_scaling = self.config.get("reward_scaling", 0.01)  # Scale rewards to avoid large values
        
        # State variables
        self.current_step = 0
        self.balance = self.starting_balance
        self.current_position = 0  # Number of contracts: positive for long, negative for short
        self.entry_price = 0
        
        # Environment dimensions
        self.state_dim = len(features)
        
        # Define action space: 0=hold, 1=buy, 2=sell to close, 3=short, 4=buy to cover
        self.action_space = spaces.Discrete(5)
        
        # Define observation space: features + [balance, position, entry_price]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim + 3,), dtype=np.float32
        )
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            numpy.ndarray: Initial state
        """
        # Reset state variables
        self.current_step = 0
        self.balance = self.starting_balance
        self.current_position = 0
        self.entry_price = 0
        
        # Choose random starting point with enough data remaining
        if len(self.historical_data) > self.episode_length:
            max_start = len(self.historical_data) - self.episode_length
            self.current_step = random.randint(0, max_start)
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Action to take (0=hold, 1=buy, 2=sell, 3=short, 4=cover)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get current data
        current_data = self.historical_data.iloc[self.current_step]
        current_price = current_data['option_price']
        
        # Track pre-action state
        prev_balance = self.balance
        prev_position = self.current_position
        
        # Process action
        if action == 0:  # Hold
            reward = -self.hold_penalty  # Small penalty for holding
            
        elif action == 1:  # Buy (open long position)
            if self.current_position <= 0:  # Only if not already long
                # Calculate position size based on available balance
                max_contracts = min(self.max_position_size, int(self.balance / (current_price * 100)))
                position_size = max(1, max_contracts)  # At least 1 contract
                
                # Calculate cost
                cost = position_size * current_price * 100
                transaction_fee = cost * self.transaction_cost
                
                # Update state
                self.balance -= (cost + transaction_fee)
                self.current_position = position_size
                self.entry_price = current_price
                
                reward = 0  # Neutral reward for opening position
            else:
                reward = -0.1  # Penalty for invalid action
        
        elif action == 2:  # Sell (close long position)
            if self.current_position > 0:  # Only if currently long
                # Calculate proceeds
                proceeds = self.current_position * current_price * 100
                transaction_fee = proceeds * self.transaction_cost
                
                # Calculate P&L
                cost_basis = self.current_position * self.entry_price * 100
                pnl = proceeds - cost_basis - transaction_fee
                
                # Update state
                self.balance += proceeds - transaction_fee
                self.current_position = 0
                self.entry_price = 0
                
                # Reward based on P&L
                reward = pnl * self.reward_scaling
            else:
                reward = -0.1  # Penalty for invalid action
        
        elif action == 3:  # Short (open short position)
            if self.current_position >= 0:  # Only if not already short
                # Calculate position size based on available balance
                max_contracts = min(self.max_position_size, int(self.balance / (current_price * 100)))
                position_size = max(1, max_contracts)  # At least 1 contract
                
                # Calculate proceeds
                proceeds = position_size * current_price * 100
                transaction_fee = proceeds * self.transaction_cost
                
                # Update state
                self.balance += (proceeds - transaction_fee)  # Short sale proceeds
                self.current_position = -position_size  # Negative for short
                self.entry_price = current_price
                
                reward = 0  # Neutral reward for opening position
            else:
                reward = -0.1  # Penalty for invalid action
        
        elif action == 4:  # Cover (close short position)
            if self.current_position < 0:  # Only if currently short
                # Calculate cost
                cost = -self.current_position * current_price * 100
                transaction_fee = cost * self.transaction_cost
                
                # Calculate P&L
                proceeds_basis = -self.current_position * self.entry_price * 100
                pnl = proceeds_basis - cost - transaction_fee
                
                # Update state
                self.balance -= (cost + transaction_fee)
                self.current_position = 0
                self.entry_price = 0
                
                # Reward based on P&L
                reward = pnl * self.reward_scaling
            else:
                reward = -0.1  # Penalty for invalid action
        
        # Move to next step
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = False
        
        if self.current_step >= len(self.historical_data) - 1:
            done = True  # Reached end of data
        elif self.done_on_zero_balance and self.balance <= 0:
            done = True  # Out of money
        elif self.current_step >= self.episode_length + self.current_step:
            done = True  # Episode length reached
        
        # If episode is done and still holding position, close it
        if done and self.current_position != 0:
            next_price = current_price  # Use current price since we're at the end
            
            if self.current_position > 0:  # Long position
                proceeds = self.current_position * next_price * 100
                transaction_fee = proceeds * self.transaction_cost
                pnl = proceeds - (self.current_position * self.entry_price * 100) - transaction_fee
                self.balance += proceeds - transaction_fee
            else:  # Short position
                cost = -self.current_position * next_price * 100
                transaction_fee = cost * self.transaction_cost
                pnl = (-self.current_position * self.entry_price * 100) - cost - transaction_fee
                self.balance -= cost + transaction_fee
            
            # Add final P&L to reward
            reward += pnl * self.reward_scaling
            
            # Clear position
            self.current_position = 0
            self.entry_price = 0
        
        # Calculate portfolio change
        portfolio_change = ((self.balance - prev_balance) / prev_balance) if prev_balance > 0 else 0
        
        # Additional info for monitoring
        info = {
            "balance": self.balance,
            "position": self.current_position,
            "entry_price": self.entry_price,
            "step_return": portfolio_change,
            "current_price": current_price,
            "action": action
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Construct the current state representation.
        
        Returns:
            numpy.ndarray: Current state
        """
        if self.current_step >= len(self.historical_data):
            # Use last available data if we've reached the end
            step_data = self.historical_data.iloc[-1]
        else:
            step_data = self.historical_data.iloc[self.current_step]
        
        # Extract features
        feature_values = step_data[self.features].values
        
        # Add account state
        account_state = np.array([
            self.balance / self.starting_balance,  # Normalize balance
            self.current_position / self.max_position_size,  # Normalize position
            self.entry_price / step_data['option_price'] if self.entry_price > 0 else 0  # Relative entry price
        ])
        
        # Combine feature values and account state
        state = np.concatenate([feature_values, account_state])
        
        return state.astype(np.float32)
    
    def render(self, mode='human'):
        """
        Render the environment (for visualization).
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, " +
                  f"Position: {self.current_position}, Entry Price: ${self.entry_price:.2f}")


class OfflineRLTrainer:
    """
    Trainer for offline reinforcement learning using historical trade data.
    
    Features:
    - Training RL agents from historical trade data
    - Behavior cloning from expert demonstrations
    - Offline policy evaluation
    - Batch-constrained Q-learning
    - Conservative Q-learning for risk-aware trading
    """
    
    def __init__(self, config=None):
        """
        Initialize the OfflineRLTrainer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.rl_params = self.config.get("offline_reinforcement", {})
        
        # Default parameters
        self.framework = self.rl_params.get("framework", "stable_baselines3" if HAS_SB3 else "tensorflow" if HAS_TF else "pytorch")
        self.algorithm = self.rl_params.get("algorithm", "ppo")
        self.batch_size = self.rl_params.get("batch_size", 64)
        self.buffer_size = self.rl_params.get("buffer_size", 100000)
        self.learning_rate = self.rl_params.get("learning_rate", 0.0003)
        self.gamma = self.rl_params.get("gamma", 0.99)
        self.train_freq = self.rl_params.get("train_freq", 1)
        self.gradient_steps = self.rl_params.get("gradient_steps", 1)
        self.target_update_interval = self.rl_params.get("target_update_interval", 1000)
        self.exploration_fraction = self.rl_params.get("exploration_fraction", 0.1)
        self.exploration_final_eps = self.rl_params.get("exploration_final_eps", 0.05)
        self.env_kwargs = self.rl_params.get("env_kwargs", {})
        
        # Data and agent state
        self.data = {}
        self.replay_buffer = None
        self.model = None
        self.env = None
        
        # Make sure required RL libraries are available
        self._check_framework_availability()
        
        # Create logs and models directories
        self.logs_dir = "logs/offline_rl"
        self.models_dir = "models/offline_rl"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.logger.info(f"OfflineRLTrainer initialized with {self.framework} framework")
    
    def _check_framework_availability(self):
        """Check if the requested RL framework is available."""
        if self.framework == "stable_baselines3" and not HAS_SB3:
            self.logger.warning("Stable-Baselines3 not available. Falling back to TensorFlow or PyTorch.")
            self.framework = "tensorflow" if HAS_TF else "pytorch" if HAS_TORCH else None
        
        if self.framework == "tensorflow" and not HAS_TF:
            self.logger.warning("TensorFlow not available. Falling back to PyTorch.")
            self.framework = "pytorch" if HAS_TORCH else None
        
        if self.framework == "pytorch" and not HAS_TORCH:
            self.logger.warning("PyTorch not available. Falling back to TensorFlow.")
            self.framework = "tensorflow" if HAS_TF else None
        
        if self.framework is None:
            self.logger.error("No supported RL framework available. Install TensorFlow, PyTorch, or Stable-Baselines3.")
    
    def load_historical_data(self, data, data_id="default", preprocess=True):
        """
        Load historical trade data for training.
        
        Args:
            data (pd.DataFrame): Historical market and option data
            data_id (str): Identifier for this dataset
            preprocess (bool): Whether to preprocess the data
            
        Returns:
            pd.DataFrame: Processed data
        """
        if data is None or len(data) == 0:
            self.logger.error("No data provided")
            return None
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Basic data validation
            required_columns = ['option_price', 'option_type', 'strike', 'underlying_price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Preprocess data if requested
            if preprocess:
                df = self._preprocess_data(df)
            
            # Store data
            self.data[data_id] = df
            
            self.logger.info(f"Loaded {len(df)} historical data points with ID: {data_id}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return None
    
    def _preprocess_data(self, df):
        """
        Preprocess historical data for RL training.
        
        Args:
            df (pd.DataFrame): Raw historical data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            # Ensure data is sorted by date
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Handle missing values
            df = df.fillna(method='ffill')
            
            # Calculate returns
            if 'option_price' in df.columns:
                df['option_return'] = df['option_price'].pct_change()
            
            if 'underlying_price' in df.columns:
                df['underlying_return'] = df['underlying_price'].pct_change()
            
            # Calculate technical indicators
            if 'option_price' in df.columns:
                # Simple moving averages
                for window in [5, 10, 20]:
                    df[f'option_sma_{window}'] = df['option_price'].rolling(window).mean()
                    
                    # Calculate SMA ratio
                    df[f'option_sma_ratio_{window}'] = df['option_price'] / df[f'option_sma_{window}']
                
                # Calculate volatility
                for window in [10, 20]:
                    df[f'option_volatility_{window}'] = df['option_price'].rolling(window).std()
            
            # Calculate RSI
            def calculate_rsi(prices, window=14):
                # Calculate price changes
                delta = prices.diff()
                
                # Separate gains and losses
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                
                # Calculate average gains and losses
                avg_gain = gains.rolling(window).mean()
                avg_loss = losses.rolling(window).mean()
                
                # Calculate relative strength
                rs = avg_gain / avg_loss
                
                # Calculate RSI
                rsi = 100 - (100 / (1 + rs))
                
                return rsi
            
            # Add RSI for option price
            if 'option_price' in df.columns:
                df['option_rsi'] = calculate_rsi(df['option_price'])
            
            # Calculate days to expiration if available
            if all(col in df.columns for col in ['date', 'expiration']):
                df['days_to_expiration'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(df['date'])).dt.days
            
            # Calculate option moneyness
            if all(col in df.columns for col in ['strike', 'underlying_price', 'option_type']):
                # For calls: underlying/strike - 1, for puts: strike/underlying - 1
                df['moneyness'] = np.where(
                    df['option_type'] == 'call',
                    df['underlying_price'] / df['strike'] - 1,
                    df['strike'] / df['underlying_price'] - 1
                )
            
            # Remove initial NaN rows from indicators
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            return df
    
    def create_environment(self, data_id="default", features=None, config=None):
        """
        Create a trading environment using historical data.
        
        Args:
            data_id (str): Identifier for the dataset to use
            features (list, optional): List of feature columns to use
            config (dict, optional): Environment configuration overrides
            
        Returns:
            TradingEnvironment: Trading environment instance
        """
        try:
            # Check if data is available
            if data_id not in self.data or self.data[data_id] is None:
                self.logger.error(f"No data found with ID: {data_id}")
                return None
            
            data = self.data[data_id]
            
            # Determine features to use if not specified
            if features is None:
                # Default feature set
                default_features = [
                    'option_price', 'underlying_price', 'strike',
                    'option_sma_ratio_5', 'option_sma_ratio_10', 'option_volatility_10',
                    'moneyness', 'option_rsi', 'days_to_expiration'
                ]
                
                # Filter to available columns
                features = [f for f in default_features if f in data.columns]
                
                if not features:
                    self.logger.warning("No default features found. Using all numeric columns.")
                    features = data.select_dtypes(include=['number']).columns.tolist()
            
            # Merge config with defaults
            env_config = self.env_kwargs.copy()
            if config:
                env_config.update(config)
            
            # Create environment
            env = TradingEnvironment(data, features, env_config)
            
            # Store environment
            self.env = env
            
            self.logger.info(f"Created trading environment with {len(features)} features")
            
            return env
            
        except Exception as e:
            self.logger.error(f"Error creating environment: {str(e)}")
            return None
    
    def create_replay_buffer(self, buffer_size=None, data_id="default", demonstrations=None):
        """
        Create a replay buffer from historical data or expert demonstrations.
        
        Args:
            buffer_size (int, optional): Size of the replay buffer
            data_id (str): Identifier for the dataset to use
            demonstrations (list, optional): List of expert demonstrations
            
        Returns:
            ReplayBuffer: Replay buffer instance
        """
        if not HAS_SB3:
            self.logger.error("Stable-Baselines3 required for replay buffer creation")
            return None
        
        try:
            if buffer_size is None:
                buffer_size = self.buffer_size
            
            # Check if we have an environment
            if self.env is None:
                self.logger.error("Environment must be created before replay buffer")
                return None
            
            # Create replay buffer
            replay_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device="cpu",
                n_envs=1,
                optimize_memory_usage=True
            )
            
            # Fill buffer from demonstrations if provided
            if demonstrations:
                self._add_demonstrations_to_buffer(replay_buffer, demonstrations)
            
            # Store buffer
            self.replay_buffer = replay_buffer
            
            self.logger.info(f"Created replay buffer with capacity {buffer_size}")
            
            return replay_buffer
            
        except Exception as e:
            self.logger.error(f"Error creating replay buffer: {str(e)}")
            return None
    
    def _add_demonstrations_to_buffer(self, replay_buffer, demonstrations):
        """
        Add expert demonstrations to replay buffer.
        
        Args:
            replay_buffer: Replay buffer instance
            demonstrations (list): List of (obs, action, reward, next_obs, done) tuples
        """
        for obs, action, reward, next_obs, done in demonstrations:
            replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=[{}]
            )
        
        self.logger.info(f"Added {len(demonstrations)} demonstrations to replay buffer")
    
    def create_model(self, model_type=None, policy="MlpPolicy", env=None, custom_model=None):
        """
        Create a reinforcement learning model.
        
        Args:
            model_type (str, optional): Type of model to create
            policy (str): Policy network type
            env: Trading environment
            custom_model: Custom model instance
            
        Returns:
            object: RL model instance
        """
        if model_type is None:
            model_type = self.algorithm
        
        if env is None:
            env = self.env
        
        if env is None:
            self.logger.error("Environment must be created before model")
            return None
        
        try:
            # Create wrapped environment for Stable-Baselines3
            if self.framework == "stable_baselines3" and HAS_SB3:
                env = DummyVecEnv([lambda: env])
                
                # Create model
                if model_type.lower() == "dqn":
                    model = DQN(
                        policy=policy,
                        env=env,
                        learning_rate=self.learning_rate,
                        buffer_size=self.buffer_size,
                        batch_size=self.batch_size,
                        gamma=self.gamma,
                        train_freq=self.train_freq,
                        gradient_steps=self.gradient_steps,
                        target_update_interval=self.target_update_interval,
                        exploration_fraction=self.exploration_fraction,
                        exploration_final_eps=self.exploration_final_eps,
                        verbose=1
                    )
                elif model_type.lower() == "ppo":
                    model = PPO(
                        policy=policy,
                        env=env,
                        learning_rate=self.learning_rate,
                        batch_size=self.batch_size,
                        gamma=self.gamma,
                        verbose=1
                    )
                elif model_type.lower() == "a2c":
                    model = A2C(
                        policy=policy,
                        env=env,
                        learning_rate=self.learning_rate,
                        gamma=self.gamma,
                        verbose=1
                    )
                elif model_type.lower() == "sac":
                    model = SAC(
                        policy=policy,
                        env=env,
                        learning_rate=self.learning_rate,
                        buffer_size=self.buffer_size,
                        batch_size=self.batch_size,
                        gamma=self.gamma,
                        train_freq=self.train_freq,
                        gradient_steps=self.gradient_steps,
                        verbose=1
                    )
                else:
                    self.logger.error(f"Unsupported model type: {model_type}")
                    return None
                
            elif self.framework == "tensorflow" and HAS_TF:
                # TensorFlow implementation
                if custom_model is not None:
                    model = custom_model
                else:
                    # Create a simple DQN model
                    model = self._create_tf_dqn(env)
                
            elif self.framework == "pytorch" and HAS_TORCH:
                # PyTorch implementation
                if custom_model is not None:
                    model = custom_model
                else:
                    # Create a simple DQN model
                    model = self._create_torch_dqn(env)
                
            else:
                self.logger.error("No supported RL framework available")
                return None
            
            # Store model
            self.model = model
            
            self.logger.info(f"Created {model_type} model with {policy} policy")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            return None
    
    def _create_tf_dqn(self, env):
        """
        Create a simple DQN model using TensorFlow.
        
        Args:
            env: Trading environment
            
        Returns:
            object: TensorFlow DQN model
        """
        # Create a simple Q-network
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(state_dim,)),
            Dense(64, activation='relu'),
            Dense(action_dim, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        # Create a simple DQN wrapper
        class SimpleDQN:
            def __init__(self, model, action_dim, gamma, batch_size):
                self.model = model
                self.action_dim = action_dim
                self.gamma = gamma
                self.batch_size = batch_size
                self.epsilon = 1.0
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.995
                self.memory = deque(maxlen=100000)
            
            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))
            
            def act(self, state, training=True):
                if training and np.random.rand() <= self.epsilon:
                    return np.random.randint(self.action_dim)
                
                q_values = self.model.predict(state.reshape(1, -1), verbose=0)
                return np.argmax(q_values[0])
            
            def replay(self):
                if len(self.memory) < self.batch_size:
                    return
                
                # Sample from memory
                minibatch = random.sample(self.memory, self.batch_size)
                
                states = np.array([s[0] for s in minibatch])
                actions = np.array([s[1] for s in minibatch])
                rewards = np.array([s[2] for s in minibatch])
                next_states = np.array([s[3] for s in minibatch])
                dones = np.array([s[4] for s in minibatch])
                
                # Get current Q values
                targets = self.model.predict(states, verbose=0)
                
                # Get next Q values
                next_q_values = self.model.predict(next_states, verbose=0)
                
                # Update targets for actions taken
                for i in range(self.batch_size):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
                
                # Train model
                self.model.fit(states, targets, epochs=1, verbose=0)
                
                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            
            def save(self, filepath):
                self.model.save(filepath)
            
            def load(self, filepath):
                self.model = tf.keras.models.load_model(filepath)
        
        return SimpleDQN(model, action_dim, self.gamma, self.batch_size)
    
    def _create_torch_dqn(self, env):
        """
        Create a simple DQN model using PyTorch.
        
        Args:
            env: Trading environment
            
        Returns:
            object: PyTorch DQN model
        """
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Define Q-network
        class QNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_dim, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, action_dim)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        # Create network
        q_network = QNetwork(state_dim, action_dim)
        target_network = QNetwork(state_dim, action_dim)
        target_network.load_state_dict(q_network.state_dict())
        
        optimizer = optim.Adam(q_network.parameters(), lr=self.learning_rate)
        
        # Create DQN wrapper
        class SimpleDQN:
            def __init__(self, q_network, target_network, optimizer, action_dim, gamma, batch_size):
                self.q_network = q_network
                self.target_network = target_network
                self.optimizer = optimizer
                self.action_dim = action_dim
                self.gamma = gamma
                self.batch_size = batch_size
                self.epsilon = 1.0
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.995
                self.memory = deque(maxlen=100000)
                self.update_count = 0
                self.update_frequency = 10
            
            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))
            
            def act(self, state, training=True):
                if training and random.random() <= self.epsilon:
                    return random.randrange(self.action_dim)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
            
            def replay(self):
                if len(self.memory) < self.batch_size:
                    return
                
                # Sample from memory
                minibatch = random.sample(self.memory, self.batch_size)
                
                states = torch.FloatTensor([s[0] for s in minibatch])
                actions = torch.LongTensor([s[1] for s in minibatch])
                rewards = torch.FloatTensor([s[2] for s in minibatch])
                next_states = torch.FloatTensor([s[3] for s in minibatch])
                dones = torch.FloatTensor([s[4] for s in minibatch])
                
                # Get current Q values
                q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # Get next Q values
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0]
                
                # Compute target Q values
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Compute loss
                loss = F.mse_loss(q_values.squeeze(), target_q_values)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update target network
                self.update_count += 1
                if self.update_count % self.update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                return loss.item()
            
            def save(self, filepath):
                torch.save({
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon
                }, filepath)
            
            def load(self, filepath):
                checkpoint = torch.load(filepath)
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
        
        return SimpleDQN(q_network, target_network, optimizer, action_dim, self.gamma, self.batch_size)
    
    def train_from_demonstrations(self, num_iterations=1000, batch_size=None, model=None, replay_buffer=None):
        """
        Train agent from demonstrations (behavior cloning).
        
        Args:
            num_iterations (int): Number of training iterations
            batch_size (int, optional): Batch size for training
            model: RL model to train
            replay_buffer: Replay buffer with demonstrations
            
        Returns:
            object: Trained model
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if model is None:
            model = self.model
        
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
        
        if model is None or replay_buffer is None:
            self.logger.error("Model and replay buffer must be created before training")
            return None
        
        try:
            self.logger.info(f"Training from demonstrations for {num_iterations} iterations")
            
            # Perform training
            if self.framework == "stable_baselines3":
                # SB3 training from buffer
                model.train(
                    gradient_steps=num_iterations,
                    batch_size=batch_size,
                    replay_buffer=replay_buffer
                )
                
            elif self.framework == "tensorflow" or self.framework == "pytorch":
                # Manual training loop
                for i in range(num_iterations):
                    # Train model
                    if hasattr(model, 'replay'):
                        loss = model.replay()
                        
                        # Log progress
                        if i % 100 == 0:
                            self.logger.info(f"Iteration {i}/{num_iterations}, Loss: {loss}")
                    else:
                        self.logger.error("Model does not have a replay method")
                        break
            
            self.logger.info("Training from demonstrations completed")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training from demonstrations: {str(e)}")
            return model
    
    def train_offline(self, total_timesteps=10000, model=None, env=None):
        """
        Train model using offline methods (without direct environment interaction).
        
        Args:
            total_timesteps (int): Total number of training timesteps
            model: RL model to train
            env: Training environment
            
        Returns:
            object: Trained model
        """
        if model is None:
            model = self.model
        
        if env is None:
            env = self.env
        
        if model is None or env is None:
            self.logger.error("Model and environment must be created before training")
            return None
        
        try:
            self.logger.info(f"Starting offline training for {total_timesteps} timesteps")
            
            # Perform training
            if self.framework == "stable_baselines3":
                # SB3 training
                model.learn(
                    total_timesteps=total_timesteps,
                    log_interval=100
                )
                
            elif self.framework == "tensorflow" or self.framework == "pytorch":
                # Manual training loop
                for episode in range(1, int(total_timesteps / 200) + 1):
                    # Reset environment
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    
                    # Episode loop
                    step = 0
                    while not done and step < 200:  # Limit episode length
                        # Select action
                        action = model.act(state)
                        
                        # Take action
                        next_state, reward, done, info = env.step(action)
                        
                        # Remember experience
                        model.remember(state, action, reward, next_state, done)
                        
                        # Train model
                        if hasattr(model, 'replay'):
                            model.replay()
                        
                        # Update state and statistics
                        state = next_state
                        episode_reward += reward
                        step += 1
                    
                    # Log progress
                    if episode % 10 == 0:
                        self.logger.info(f"Episode {episode}/{int(total_timesteps/200)}, Reward: {episode_reward:.2f}, Epsilon: {model.epsilon:.2f}")
            
            self.logger.info("Offline training completed")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error in offline training: {str(e)}")
            return model
    
    def train_model(self, method="offline", total_timesteps=10000, num_iterations=1000, batch_size=None, 
                   model=None, env=None, replay_buffer=None):
        """
        Train model using the specified method.
        
        Args:
            method (str): Training method ('offline', 'demonstrations', or 'combined')
            total_timesteps (int): Total timesteps for offline training
            num_iterations (int): Number of iterations for demonstration training
            batch_size (int, optional): Batch size for training
            model: RL model to train
            env: Training environment
            replay_buffer: Replay buffer with demonstrations
            
        Returns:
            object: Trained model
        """
        if model is None:
            model = self.model
        
        if env is None:
            env = self.env
        
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
        
        if model is None:
            self.logger.error("Model must be created before training")
            return None
        
        try:
            if method == "offline":
                # Offline RL training
                return self.train_offline(total_timesteps, model, env)
                
            elif method == "demonstrations":
                # Training from demonstrations
                if replay_buffer is None:
                    self.logger.error("Replay buffer required for demonstration training")
                    return model
                
                return self.train_from_demonstrations(num_iterations, batch_size, model, replay_buffer)
                
            elif method == "combined":
                # Combined approach: first train from demonstrations, then offline
                if replay_buffer is None:
                    self.logger.error("Replay buffer required for combined training")
                    return model
                
                # First train from demonstrations
                model = self.train_from_demonstrations(num_iterations, batch_size, model, replay_buffer)
                
                # Then train offline
                return self.train_offline(total_timesteps, model, env)
                
            else:
                self.logger.error(f"Unknown training method: {method}")
                return model
                
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            return model
    
    def evaluate_model(self, num_episodes=10, render=False, model=None, env=None):
        """
        Evaluate a trained model on the environment.
        
        Args:
            num_episodes (int): Number of episodes for evaluation
            render (bool): Whether to render the environment
            model: Model to evaluate
            env: Evaluation environment
            
        Returns:
            dict: Evaluation results
        """
        if model is None:
            model = self.model
        
        if env is None:
            env = self.env
        
        if model is None or env is None:
            self.logger.error("Model and environment must be created before evaluation")
            return None
        
        try:
            self.logger.info(f"Evaluating model for {num_episodes} episodes")
            
            # Track evaluation results
            episode_rewards = []
            episode_returns = []
            episode_lengths = []
            final_balances = []
            
            # Run evaluation episodes
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                step = 0
                
                while not done:
                    # Select action
                    if self.framework == "stable_baselines3":
                        action, _ = model.predict(state, deterministic=True)
                    else:
                        action = model.act(state, training=False)
                    
                    # Take action
                    state, reward, done, info = env.step(action)
                    
                    if render:
                        env.render()
                    
                    # Update statistics
                    episode_reward += reward
                    step += 1
                
                # Record episode statistics
                episode_rewards.append(episode_reward)
                episode_returns.append((env.balance / env.starting_balance) - 1)  # Return as percentage
                episode_lengths.append(step)
                final_balances.append(env.balance)
                
                self.logger.info(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Return: {episode_returns[-1]:.2%}, Final Balance: ${final_balances[-1]:.2f}")
            
            # Calculate evaluation metrics
            mean_reward = np.mean(episode_rewards)
            mean_return = np.mean(episode_returns)
            mean_length = np.mean(episode_lengths)
            mean_balance = np.mean(final_balances)
            
            results = {
                "mean_reward": mean_reward,
                "mean_return": mean_return,
                "mean_episode_length": mean_length,
                "mean_final_balance": mean_balance,
                "episode_rewards": episode_rewards,
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
                "final_balances": final_balances
            }
            
            self.logger.info(f"Evaluation results: Mean Reward: {mean_reward:.2f}, Mean Return: {mean_return:.2%}, Mean Final Balance: ${mean_balance:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            return None
    
    def save_model(self, filepath=None, model=None):
        """
        Save trained model to file.
        
        Args:
            filepath (str, optional): Path to save the model
            model: Model to save
            
        Returns:
            str: Path to saved model
        """
        if model is None:
            model = self.model
        
        if model is None:
            self.logger.error("No model to save")
            return None
        
        # Create default filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.models_dir, f"rl_model_{timestamp}")
        
        try:
            # Save model based on framework
            if self.framework == "stable_baselines3":
                model.save(filepath)
            elif hasattr(model, 'save'):
                model.save(filepath)
            else:
                self.logger.error("Model does not have a save method")
                return None
            
            self.logger.info(f"Model saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, filepath, model_type=None, env=None):
        """
        Load model from file.
        
        Args:
            filepath (str): Path to model file
            model_type (str, optional): Type of model to load
            env: Environment for the model
            
        Returns:
            object: Loaded model
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            return None
        
        if env is None:
            env = self.env
        
        if env is None:
            self.logger.error("Environment must be created before loading model")
            return None
        
        try:
            # Load model based on framework
            if self.framework == "stable_baselines3" and HAS_SB3:
                if model_type is None:
                    # Try to infer model type from filename
                    if "dqn" in filepath.lower():
                        model_type = "dqn"
                    elif "ppo" in filepath.lower():
                        model_type = "ppo"
                    elif "a2c" in filepath.lower():
                        model_type = "a2c"
                    elif "sac" in filepath.lower():
                        model_type = "sac"
                    else:
                        model_type = self.algorithm
                
                # Create wrapped environment for SB3
                vec_env = DummyVecEnv([lambda: env])
                
                # Load model
                if model_type.lower() == "dqn":
                    model = DQN.load(filepath, env=vec_env)
                elif model_type.lower() == "ppo":
                    model = PPO.load(filepath, env=vec_env)
                elif model_type.lower() == "a2c":
                    model = A2C.load(filepath, env=vec_env)
                elif model_type.lower() == "sac":
                    model = SAC.load(filepath, env=vec_env)
                else:
                    self.logger.error(f"Unsupported model type: {model_type}")
                    return None
                
            elif self.framework == "tensorflow" and HAS_TF:
                # Create a new model
                model = self._create_tf_dqn(env)
                
                # Load weights
                if hasattr(model, 'load'):
                    model.load(filepath)
                else:
                    self.logger.error("Model does not have a load method")
                    return None
                
            elif self.framework == "pytorch" and HAS_TORCH:
                # Create a new model
                model = self._create_torch_dqn(env)
                
                # Load weights
                if hasattr(model, 'load'):
                    model.load(filepath)
                else:
                    self.logger.error("Model does not have a load method")
                    return None
                
            else:
                self.logger.error("No supported RL framework available")
                return None
            
            # Store model
            self.model = model
            
            self.logger.info(f"Model loaded from {filepath}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def visualize_training(self, training_history=None, save_path=None):
        """
        Visualize training progress.
        
        Args:
            training_history (dict): Training history data
            save_path (str, optional): Path to save the visualization
            
        Returns:
            tuple: Figure and axes
        """
        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot rewards
            if training_history and 'rewards' in training_history:
                rewards = training_history['rewards']
                axes[0, 0].plot(rewards, label='Episode Reward')
                if len(rewards) > 10:
                    window_size = min(10, len(rewards) // 5)
                    rolling_mean = pd.Series(rewards).rolling(window=window_size).mean()
                    axes[0, 0].plot(rolling_mean, label=f'{window_size}-Episode Moving Avg', color='red')
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            else:
                axes[0, 0].text(0.5, 0.5, "No reward data available", 
                              ha='center', va='center', transform=axes[0, 0].transAxes)
            
            # Plot returns
            if training_history and 'returns' in training_history:
                returns = training_history['returns']
                axes[0, 1].plot(returns, label='Episode Return')
                if len(returns) > 10:
                    window_size = min(10, len(returns) // 5)
                    rolling_mean = pd.Series(returns).rolling(window=window_size).mean()
                    axes[0, 1].plot(rolling_mean, label=f'{window_size}-Episode Moving Avg', color='red')
                axes[0, 1].set_title('Episode Returns (%)')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Return')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            else:
                axes[0, 1].text(0.5, 0.5, "No return data available", 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # Plot balances
            if training_history and 'balances' in training_history:
                balances = training_history['balances']
                axes[1, 0].plot(balances, label='Final Balance')
                if len(balances) > 10:
                    window_size = min(10, len(balances) // 5)
                    rolling_mean = pd.Series(balances).rolling(window=window_size).mean()
                    axes[1, 0].plot(rolling_mean, label=f'{window_size}-Episode Moving Avg', color='red')
                axes[1, 0].set_title('Final Account Balance')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Balance ($)')
                axes[1, 0].axhline(y=self.env_kwargs.get('starting_balance', 100000), color='green', linestyle='--', 
                                 label='Starting Balance')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, "No balance data available", 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Plot action distribution
            if training_history and 'actions' in training_history:
                actions = training_history['actions']
                action_counts = pd.Series(actions).value_counts().sort_index()
                
                action_names = ['Hold', 'Buy', 'Sell', 'Short', 'Cover']
                action_labels = [f"{action_names[i]} ({count})" for i, count in enumerate(action_counts)]
                
                axes[1, 1].bar(action_names[:len(action_counts)], action_counts, color='skyblue')
                axes[1, 1].set_title('Action Distribution')
                axes[1, 1].set_xlabel('Action')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].grid(True, axis='y')
            else:
                axes[1, 1].text(0.5, 0.5, "No action data available", 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Visualization saved to {save_path}")
            
            return fig, axes
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            return None, None
