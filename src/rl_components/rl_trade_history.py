"""
RL Trade History Model Module

This module implements reinforcement learning models for analyzing historical trade data
to optimize trade parameters, entry/exit timing, and risk management strategies.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import for RL algorithms
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class TradingEnv(gym.Env):
    """
    Custom gym environment for historical trade data.
    
    This environment simulates trading decisions based on historical trade data,
    allowing the agent to learn optimal trade parameters and strategies.
    """
    
    def __init__(self, trade_data, lookback=10, episode_length=None):
        """
        Initialize the environment.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            lookback (int): Number of past trades to include in state
            episode_length (int, optional): Length of each episode
        """
        super(TradingEnv, self).__init__()
        
        # Store data
        self.trade_data = trade_data
        self.lookback = lookback
        
        # Set episode length
        if episode_length is None:
            self.episode_length = len(trade_data) - lookback
        else:
            self.episode_length = min(episode_length, len(trade_data) - lookback)
        
        # Define action space
        # Action space represents:
        # 1. Entry threshold (0-1): How strong a signal needs to be to enter
        # 2. Exit threshold (0-1): How strong a signal needs to be to exit
        # 3. Position size (0-1): How much capital to allocate (% of max)
        # 4. Stop loss (0-1): Where to set stop loss (% of entry price)
        # 5. Take profit (0-1): Where to set take profit (% of entry price)
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(5,),
            dtype=np.float32
        )
        
        # Define observation space
        # Features from trade history
        self.n_features = self._get_feature_count()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_features),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _get_feature_count(self):
        """
        Determine the number of features in the processed data.
        
        Returns:
            int: Number of features
        """
        # Process a single row to determine feature count
        sample_features = self._process_data_row(self.trade_data.iloc[0:self.lookback])
        return sample_features.shape[1]
    
    def _process_data_row(self, data_window):
        """
        Process trade data to extract features.
        
        Args:
            data_window (pd.DataFrame): Window of trade data
            
        Returns:
            np.ndarray: Extracted features
        """
        # Get required columns
        required_columns = ['entry_price', 'exit_price', 'pnl', 'pnl_percent']
        
        # Check for missing columns and add them if possible
        if 'entry_price' not in data_window.columns or 'exit_price' not in data_window.columns:
            self.logger.error("Cannot process trade data without entry_price and exit_price")
            return np.zeros((1, 10))  # Default empty features
        
        if 'pnl' not in data_window.columns and 'entry_price' in data_window.columns and 'exit_price' in data_window.columns:
            data_window['pnl'] = data_window['exit_price'] - data_window['entry_price']
        
        if 'pnl_percent' not in data_window.columns and 'entry_price' in data_window.columns and 'exit_price' in data_window.columns:
            data_window['pnl_percent'] = (data_window['exit_price'] - data_window['entry_price']) / data_window['entry_price']
        
        # Process data and extract features
        df = data_window.copy()
        
        # Create basic features
        features = []
        
        # Profit/loss features
        features.append(df['pnl'].mean())
        features.append(df['pnl'].std())
        features.append(df['pnl_percent'].mean())
        features.append(df['pnl_percent'].std())
        
        # Win/loss metrics
        win_rate = (df['pnl'] > 0).mean()
        features.append(win_rate)
        
        # Average gain/loss ratio
        avg_gain = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if len(df[df['pnl'] < 0]) > 0 else 1  # Avoid division by zero
        gain_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 1
        features.append(gain_loss_ratio)
        
        # Consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        
        for i, row in df.iterrows():
            if row['pnl'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_lose_streak = max(max_lose_streak, abs(current_streak))
        
        features.append(max_win_streak)
        features.append(max_lose_streak)
        
        # Duration features if available
        if 'duration' in df.columns:
            features.append(df['duration'].mean())
            features.append(df['duration'].std())
        else:
            features.append(0)
            features.append(0)
        
        # Return features as numpy array
        return np.array(features).reshape(1, -1)
    
    def _process_data(self):
        """
        Process all trade data to create features.
        
        Returns:
            np.ndarray: Processed features
        """
        # Process each window of data
        feature_list = []
        
        for i in range(len(self.trade_data) - self.lookback + 1):
            window = self.trade_data.iloc[i:i+self.lookback]
            features = self._process_data_row(window)
            feature_list.append(features)
        
        # Concatenate all features
        return np.vstack(feature_list)
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            np.ndarray: Initial state
        """
        # Process data
        self.features = self._process_data()
        
        # Initialize state
        self.current_step = 0
        self.total_pnl = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.entry_prices = []
        self.exit_prices = []
        self.trade_pnls = []
        self.rewards = []
        
        # Create initial state
        self.state = self.features[0:self.lookback]
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (numpy.ndarray): Action vector [entry_threshold, exit_threshold, 
                                                position_size, stop_loss, take_profit]
            
        Returns:
            tuple: Next state, reward, done, info
        """
        # Extract actions
        entry_threshold = action[0]
        exit_threshold = action[1]
        position_size = action[2]
        stop_loss = action[3]
        take_profit = action[4]
        
        # Get current trade data
        current_idx = self.current_step + self.lookback
        current_trade = self.trade_data.iloc[current_idx]
        
        # Simulate trade based on parameters
        trade_pnl = 0
        trade_successful = False
        
        # Entry decision based on signal strength
        # For simulation, we use a simplified approach
        trade_signal = current_trade.get('entry_score', 0.5)  # Default to neutral if not available
        
        if trade_signal >= entry_threshold:
            # Enter trade
            entry_price = current_trade['entry_price']
            
            # Calculate exit price based on parameters
            exit_price = current_trade['exit_price']
            
            # Apply stop loss and take profit
            stop_price = entry_price * (1 - stop_loss)
            profit_price = entry_price * (1 + take_profit)
            
            # Check if stop loss or take profit was hit
            if stop_loss > 0 and exit_price <= stop_price:
                # Stop loss hit
                exit_price = stop_price
            elif take_profit > 0 and exit_price >= profit_price:
                # Take profit hit
                exit_price = profit_price
            
            # Calculate trade PnL
            trade_pnl = (exit_price - entry_price) / entry_price
            trade_pnl = trade_pnl * position_size  # Scale by position size
            
            # Record trade
            self.total_trades += 1
            self.entry_prices.append(entry_price)
            self.exit_prices.append(exit_price)
            self.trade_pnls.append(trade_pnl)
            
            if trade_pnl > 0:
                self.successful_trades += 1
                trade_successful = True
            
            # Update total PnL
            self.total_pnl += trade_pnl
        
        # Calculate reward
        reward = self._calculate_reward(trade_pnl, trade_successful, entry_threshold, exit_threshold, position_size)
        self.rewards.append(reward)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Get new state
        if not done:
            next_idx = min(self.current_step, len(self.features) - self.lookback)
            self.state = self.features[next_idx:next_idx+self.lookback]
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate': self.successful_trades / max(1, self.total_trades),
            'trade_pnl': trade_pnl,
            'cumulative_reward': sum(self.rewards)
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, trade_pnl, trade_successful, entry_threshold, exit_threshold, position_size):
        """
        Calculate reward based on trade outcome and parameters.
        
        Args:
            trade_pnl (float): Trade profit/loss percentage
            trade_successful (bool): Whether the trade was successful
            entry_threshold (float): Entry threshold parameter
            exit_threshold (float): Exit threshold parameter
            position_size (float): Position size parameter
            
        Returns:
            float: Reward
        """
        # Base reward on trade PnL
        reward = trade_pnl * 100  # Scale for more meaningful rewards
        
        # Additional rewards for good parameter choices
        if trade_successful:
            # Reward for successful trade
            reward += 1
            
            # Reward for efficient capital use
            if position_size > 0.2:  # Meaningful position size
                reward += position_size
            
            # Reward for good timing (higher threshold for better signals)
            reward += entry_threshold
        else:
            # Penalty for unsuccessful trade
            if trade_pnl != 0:  # Only if a trade was made
                reward -= 0.5
                
                # Larger penalty for aggressive position sizing on losing trades
                reward -= position_size
        
        # Reward for capital preservation
        if self.total_pnl >= 0:
            reward += 0.1
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented).
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            None
        """
        pass


class RLTradeHistoryModel:
    """
    Reinforcement learning model for analyzing historical trade data to optimize trading parameters.
    
    Features:
    - Learning optimal trade parameters from historical data
    - Optimization of entry/exit timing
    - Risk management parameter tuning
    - Reward function that balances returns and risk
    """
    
    def __init__(self, config=None):
        """
        Initialize the RLTradeHistoryModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = os.path.join("models", "rl_trade_history")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.lookback = self.config.get('lookback', 10)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.rl_model = None
        
        # Flag to check if stable-baselines3 is available
        self.sb3_available = HAS_SB3
        if not self.sb3_available:
            self.logger.warning("stable-baselines3 not available. Using custom implementation.")
        
        self.logger.info("RLTradeHistoryModel initialized")
    
    def create_env(self, trade_data, lookback=None, episode_length=None):
        """
        Create a gym environment for trade data.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            lookback (int, optional): Number of past trades for state
            episode_length (int, optional): Length of each episode
            
        Returns:
            TradingEnv: Gym environment
        """
        # Use specified lookback or default
        if lookback is None:
            lookback = self.lookback
        
        # Create environment
        env = TradingEnv(trade_data, lookback, episode_length)
        
        # Wrap in DummyVecEnv for stable-baselines3 compatibility
        if self.sb3_available:
            env = DummyVecEnv([lambda: env])
        
        return env
    
    def _build_model(self, input_shape, output_dim):
        """
        Build a policy model for RL.
        
        Args:
            input_shape (tuple): Shape of input data
            output_dim (int): Dimension of output (action space)
            
        Returns:
            tf.keras.Model: Policy model
        """
        # Flatten lookback window into single vector
        input_layer = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(input_layer)
        x = BatchNormalization()(x)
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer for continuous actions
        output_layer = Dense(output_dim, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model
    
    def train(self, trade_data, n_steps=100000, model_name=None):
        """
        Train the RL model on historical trade data.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            n_steps (int): Number of training steps
            model_name (str, optional): Name for the saved model
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training RL model on trade history data, steps: {n_steps}")
        
        # Create environment
        env = self.create_env(trade_data)
        
        # Set default model_name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"rl_trade_history_{timestamp}"
        
        # Create model save path
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
            if self.sb3_available:
                # Create SAC model using stable-baselines3
                # SAC is good for continuous action spaces
                self.rl_model = SAC(
                    "MlpLstmPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    verbose=1
                )
                
                # Create evaluation callback
                eval_callback = EvalCallback(
                    env,
                    best_model_save_path=model_path,
                    log_path=os.path.join(model_path, "logs"),
                    eval_freq=1000,
                    deterministic=True,
                    render=False
                )
                
                # Train model
                self.rl_model.learn(
                    total_timesteps=n_steps,
                    callback=eval_callback
                )
                
                # Save final model
                self.rl_model.save(os.path.join(model_path, "final_model"))
                
                # Evaluate policy
                mean_reward, std_reward = evaluate_policy(
                    self.rl_model,
                    env,
                    n_eval_episodes=10
                )
                
                metrics = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'n_steps': n_steps,
                    'model_type': 'stable_baselines3_sac'
                }
                
            else:
                # Use custom implementation for continuous action space
                self.logger.warning("Custom implementation for continuous action space is simplified.")
                
                # Get input and output shapes
                input_shape = (self.lookback, env.n_features)
                output_dim = 5  # 5 continuous parameters
                
                # Build model
                self.policy_model = self._build_model(input_shape, output_dim)
                
                # Training loop
                n_episodes = 100  # Simplified
                rewards_history = []
                
                for episode in range(n_episodes):
                    state = env.reset()
                    done = False
                    episode_rewards = []
                    
                    while not done:
                        # Predict action
                        action = self.policy_model.predict(np.expand_dims(state, axis=0))[0]
                        
                        # Add some exploration noise
                        noise = np.random.normal(0, 0.1, size=action.shape)
                        action = np.clip(action + noise, 0, 1)
                        
                        # Take action
                        next_state, reward, done, _ = env.step(action)
                        
                        # Store reward
                        episode_rewards.append(reward)
                        
                        # Create training sample
                        target = action.copy()
                        
                        # Update target based on reward (simple update rule)
                        if reward > 0:
                            # Reinforce successful actions
                            pass  # Keep the action as is
                        else:
                            # Modify unsuccessful actions
                            target += np.random.normal(0, 0.2, size=target.shape)
                            target = np.clip(target, 0, 1)
                        
                        # Train model
                        self.policy_model.fit(
                            np.expand_dims(state, axis=0),
                            np.expand_dims(target, axis=0),
                            verbose=0
                        )
                        
                        # Move to next state
                        state = next_state
                    
                    # Store episode rewards
                    total_reward = sum(episode_rewards)
                    rewards_history.append(total_reward)
                    
                    # Log progress
                    if (episode + 1) % 10 == 0:
                        self.logger.info(f"Episode {episode + 1}/{n_episodes}, Mean Reward: {np.mean(rewards_history[-10:]):.4f}")
                
                # Save model
                os.makedirs(model_path, exist_ok=True)
                self.policy_model.save(os.path.join(model_path, "policy_model.h5"))
                
                # Save config
                config = {
                    'lookback': self.lookback,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma
                }
                
                with open(os.path.join(model_path, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                
                # Calculate metrics
                metrics = {
                    'mean_reward': np.mean(rewards_history[-10:]),
                    'std_reward': np.std(rewards_history[-10:]),
                    'n_steps': n_steps,
                    'model_type': 'custom_policy'
                }
            
            # Save training metrics
            with open(os.path.join(model_path, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"Training completed. Mean reward: {metrics['mean_reward']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training RL model: {str(e)}")
            return None
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if self.sb3_available and os.path.exists(os.path.join(model_path, "final_model.zip")):
                # Load stable-baselines3 model
                self.rl_model = SAC.load(os.path.join(model_path, "final_model"))
                self.logger.info(f"Loaded stable-baselines3 model from {model_path}")
                return True
                
            elif os.path.exists(os.path.join(model_path, "policy_model.h5")):
                # Load custom model
                self.policy_model = load_model(os.path.join(model_path, "policy_model.h5"))
                
                # Load config
                if os.path.exists(os.path.join(model_path, "config.json")):
                    with open(os.path.join(model_path, "config.json"), 'r') as f:
                        config = json.load(f)
                        self.lookback = config.get('lookback', self.lookback)
                        self.learning_rate = config.get('learning_rate', self.learning_rate)
                        self.gamma = config.get('gamma', self.gamma)
                
                self.logger.info(f"Loaded custom policy model from {model_path}")
                return True
                
            else:
                self.logger.error(f"No valid model found at {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def optimize_parameters(self, trade_data):
        """
        Optimize trading parameters based on historical data.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            
        Returns:
            dict: Optimized parameters
        """
        if self.rl_model is None and not hasattr(self, 'policy_model'):
            self.logger.error("No model loaded. Please train or load a model first.")
            return None
        
        self.logger.info("Optimizing trade parameters")
        
        try:
            # Create environment
            env = self.create_env(trade_data)
            
            # Reset environment
            state = env.reset()
            
            # Get optimal parameters from model
            if self.sb3_available and self.rl_model is not None:
                action, _ = self.rl_model.predict(state, deterministic=True)
            else:
                action = self.policy_model.predict(np.expand_dims(state, axis=0))[0]
            
            # Extract parameters
            entry_threshold = float(action[0])
            exit_threshold = float(action[1])
            position_size = float(action[2])
            stop_loss = float(action[3])
            take_profit = float(action[4])
            
            # Create parameters dictionary
            parameters = {
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Optimized parameters: {parameters}")
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            return None
    
    def simulate_trading(self, trade_data, parameters=None, n_steps=None):
        """
        Simulate trading with given parameters or model predictions.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            parameters (dict, optional): Trading parameters
            n_steps (int, optional): Number of steps to simulate
            
        Returns:
            dict: Simulation results
        """
        self.logger.info("Simulating trading with parameters")
        
        try:
            # Create environment
            env = self.create_env(trade_data)
            
            # Set number of steps to simulate
            if n_steps is None:
                n_steps = len(trade_data) - self.lookback - 1
            
            # Reset environment
            state = env.reset()
            
            # Storage for simulation results
            rewards = []
            actions = []
            trades = []
            
            # Simulation loop
            for step in range(n_steps):
                # Get action
                if parameters is not None:
                    # Use provided parameters
                    action = np.array([
                        parameters.get('entry_threshold', 0.5),
                        parameters.get('exit_threshold', 0.5),
                        parameters.get('position_size', 0.5),
                        parameters.get('stop_loss', 0.1),
                        parameters.get('take_profit', 0.2)
                    ])
                elif self.rl_model is not None:
                    # Use stable-baselines3 model
                    action, _ = self.rl_model.predict(state, deterministic=True)
                elif hasattr(self, 'policy_model'):
                    # Use custom model
                    action = self.policy_model.predict(np.expand_dims(state, axis=0))[0]
                else:
                    self.logger.error("No model loaded or parameters provided")
                    return None
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store results
                rewards.append(float(reward))
                actions.append(action.tolist())
                
                # Store trade if made
                if info['trade_pnl'] != 0:
                    trades.append({
                        'step': step,
                        'pnl': float(info['trade_pnl']),
                        'entry_threshold': float(action[0]),
                        'exit_threshold': float(action[1]),
                        'position_size': float(action[2]),
                        'stop_loss': float(action[3]),
                        'take_profit': float(action[4])
                    })
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate performance metrics
            total_reward = sum(rewards)
            mean_reward = np.mean(rewards)
            win_rate = info['win_rate']
            total_trades = info['total_trades']
            total_pnl = info['total_pnl']
            
            # Calculate average parameters
            avg_entry_threshold = np.mean([a[0] for a in actions])
            avg_exit_threshold = np.mean([a[1] for a in actions])
            avg_position_size = np.mean([a[2] for a in actions])
            avg_stop_loss = np.mean([a[3] for a in actions])
            avg_take_profit = np.mean([a[4] for a in actions])
            
            # Compile results
            results = {
                'trades': trades,
                'total_trades': total_trades,
                'successful_trades': info['successful_trades'],
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'mean_reward': mean_reward,
                'total_reward': total_reward,
                'average_parameters': {
                    'entry_threshold': avg_entry_threshold,
                    'exit_threshold': avg_exit_threshold,
                    'position_size': avg_position_size,
                    'stop_loss': avg_stop_loss,
                    'take_profit': avg_take_profit
                },
                'simulation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Simulation completed. Win rate: {win_rate:.2f}, Total PnL: {total_pnl:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error simulating trading: {str(e)}")
            return None
    
    def analyze_results(self, simulation_results):
        """
        Analyze simulation results to extract insights.
        
        Args:
            simulation_results (dict): Results from simulation
            
        Returns:
            dict: Analysis of results
        """
        self.logger.info("Analyzing simulation results")
        
        try:
            # Extract key metrics
            win_rate = simulation_results['win_rate']
            total_pnl = simulation_results['total_pnl']
            total_trades = simulation_results['total_trades']
            trades = simulation_results['trades']
            avg_params = simulation_results['average_parameters']
            
            # Calculate trade metrics
            if trades:
                trade_pnls = [t['pnl'] for t in trades]
                avg_pnl = np.mean(trade_pnls)
                max_pnl = max(trade_pnls)
                min_pnl = min(trade_pnls)
                std_pnl = np.std(trade_pnls)
                
                # Calculate profit factor
                gains = sum(pnl for pnl in trade_pnls if pnl > 0)
                losses = sum(abs(pnl) for pnl in trade_pnls if pnl < 0)
                profit_factor = gains / losses if losses > 0 else float('inf')
                
                # Calculate drawdown
                cumulative_pnl = np.cumsum(trade_pnls)
                peak = np.maximum.accumulate(cumulative_pnl)
                drawdown = peak - cumulative_pnl
                max_drawdown = drawdown.max()
                
                # Parameter efficacy
                param_pnl_correlation = {}
                
                for param in ['entry_threshold', 'exit_threshold', 'position_size', 'stop_loss', 'take_profit']:
                    param_values = [t[param] for t in trades]
                    if len(param_values) > 5:  # Need enough data for correlation
                        correlation = np.corrcoef(param_values, trade_pnls)[0, 1]
                        param_pnl_correlation[param] = correlation
            else:
                avg_pnl = 0
                max_pnl = 0
                min_pnl = 0
                std_pnl = 0
                profit_factor = 0
                max_drawdown = 0
                param_pnl_correlation = {}
            
            # Compile analysis
            analysis = {
                'trade_metrics': {
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'avg_pnl': avg_pnl,
                    'max_pnl': max_pnl,
                    'min_pnl': min_pnl,
                    'std_pnl': std_pnl,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_drawdown
                },
                'parameter_efficacy': param_pnl_correlation,
                'recommended_parameters': {
                    # Adjust based on correlation with PnL
                    'entry_threshold': self._optimize_param(avg_params['entry_threshold'], param_pnl_correlation.get('entry_threshold', 0)),
                    'exit_threshold': self._optimize_param(avg_params['exit_threshold'], param_pnl_correlation.get('exit_threshold', 0)),
                    'position_size': self._optimize_param(avg_params['position_size'], param_pnl_correlation.get('position_size', 0)),
                    'stop_loss': self._optimize_param(avg_params['stop_loss'], param_pnl_correlation.get('stop_loss', 0)),
                    'take_profit': self._optimize_param(avg_params['take_profit'], param_pnl_correlation.get('take_profit', 0))
                },
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Analysis completed. Profit factor: {profit_factor:.2f}, Max drawdown: {max_drawdown:.4f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {str(e)}")
            return None
    
    def _optimize_param(self, param_value, correlation):
        """
        Optimize parameter based on correlation with PnL.
        
        Args:
            param_value (float): Current parameter value
            correlation (float): Correlation with PnL
            
        Returns:
            float: Optimized parameter value
        """
        # If correlation is positive, increase parameter value
        if correlation > 0.2:
            return min(1.0, param_value * 1.2)
        # If correlation is negative, decrease parameter value
        elif correlation < -0.2:
            return max(0.1, param_value * 0.8)
        # If correlation is weak, keep parameter value
        else:
            return param_value
    
    def visualize_results(self, simulation_results, output_file=None):
        """
        Visualize simulation results.
        
        Args:
            simulation_results (dict): Results from simulation
            output_file (str, optional): Path to save the visualization
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Visualizing simulation results")
        
        try:
            # Extract data
            trades = simulation_results['trades']
            
            if not trades:
                self.logger.warning("No trades available for visualization")
                return None, None
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 12))
            
            # 1. PnL over time
            trade_steps = [t['step'] for t in trades]
            trade_pnls = [t['pnl'] for t in trades]
            cumulative_pnl = np.cumsum(trade_pnls)
            
            axs[0].plot(trade_steps, cumulative_pnl, 'b-', linewidth=2)
            axs[0].set_title('Cumulative PnL over Time')
            axs[0].set_xlabel('Trade Step')
            axs[0].set_ylabel('Cumulative PnL')
            axs[0].grid(True)
            
            # Add drawdown visualization
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            
            # Fill between peak and cumulative PnL to show drawdown
            axs[0].fill_between(
                trade_steps, 
                cumulative_pnl, 
                peak,
                where=drawdown > 0,
                color='red',
                alpha=0.3,
                label='Drawdown'
            )
            
            axs[0].legend()
            
            # 2. Parameter values over time
            params = ['entry_threshold', 'exit_threshold', 'position_size', 'stop_loss', 'take_profit']
            param_values = {param: [t[param] for t in trades] for param in params}
            
            for param in params:
                axs[1].plot(trade_steps, param_values[param], label=param)
            
            axs[1].set_title('Parameter Values over Time')
            axs[1].set_xlabel('Trade Step')
            axs[1].set_ylabel('Parameter Value')
            axs[1].grid(True)
            axs[1].legend()
            
            # 3. Trade PnL distribution
            axs[2].hist(trade_pnls, bins=20, alpha=0.7, color='blue')
            axs[2].set_title('Trade PnL Distribution')
            axs[2].set_xlabel('PnL')
            axs[2].set_ylabel('Frequency')
            axs[2].grid(True)
            
            # Add mean and zero lines
            mean_pnl = np.mean(trade_pnls)
            axs[2].axvline(x=mean_pnl, color='g', linestyle='--', label=f'Mean PnL: {mean_pnl:.4f}')
            axs[2].axvline(x=0, color='r', linestyle='-', label='Break Even')
            axs[2].legend()
            
            # Add overall title with key metrics
            win_rate = simulation_results['win_rate']
            total_pnl = simulation_results['total_pnl']
            total_trades = simulation_results['total_trades']
            
            fig.suptitle(
                f'Trading Simulation Results\n'
                f'Win Rate: {win_rate:.2f}, Total PnL: {total_pnl:.4f}, Trades: {total_trades}',
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save figure if output file specified
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved visualization to {output_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing results: {str(e)}")
            return None, None
