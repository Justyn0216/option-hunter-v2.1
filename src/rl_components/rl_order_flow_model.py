"""
RL Order Flow Model Module

This module implements reinforcement learning models for analyzing option order flow data
to predict market movements and identify profitable trading strategies based on order patterns.
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

# Import for RL algorithms (PPO - Proximal Policy Optimization)
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class OrderFlowEnv(gym.Env):
    """
    Custom gym environment for option order flow data.
    
    This environment takes order flow data and creates a RL environment
    where the agent can learn to make trading decisions based on order patterns.
    """
    
    def __init__(self, order_flow_data, window_size=10, max_steps=None):
        """
        Initialize the environment.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            window_size (int): Number of time steps to include in the state
            max_steps (int, optional): Maximum number of steps in an episode
        """
        super(OrderFlowEnv, self).__init__()
        
        # Store data
        self.order_flow_data = order_flow_data
        self.window_size = window_size
        
        # Set max_steps
        if max_steps is None:
            self.max_steps = len(order_flow_data) - window_size - 1
        else:
            self.max_steps = min(max_steps, len(order_flow_data) - window_size - 1)
        
        # Define action space: 0 = no position, 1 = long call, 2 = long put
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (state)
        # Features include order flow metrics
        self.n_features = self._get_feature_count()
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, self.n_features),
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
        sample_features = self._process_data_row(self.order_flow_data.iloc[0:self.window_size])
        return sample_features.shape[1]
    
    def _process_data_row(self, data_window):
        """
        Process order flow data to extract features.
        
        Args:
            data_window (pd.DataFrame): Window of order flow data
            
        Returns:
            np.ndarray: Extracted features
        """
        # Check which columns are available
        required_columns = ['price', 'size', 'side', 'option_type']
        missing_columns = [col for col in required_columns if col not in data_window.columns]
        
        if missing_columns:
            # Use default values for missing columns
            for col in missing_columns:
                if col == 'side':
                    # Default to balanced buy/sell
                    data_window['side'] = 'buy'
                    data_window.loc[data_window.index % 2 == 0, 'side'] = 'sell'
                elif col == 'option_type':
                    # Default to balanced call/put
                    data_window['option_type'] = 'call'
                    data_window.loc[data_window.index % 2 == 0, 'option_type'] = 'put'
        
        # Create order flow metrics
        df = data_window.copy()
        
        # Convert categorical variables to numeric
        df['is_buy'] = (df['side'] == 'buy').astype(int)
        df['is_call'] = (df['option_type'] == 'call').astype(int)
        
        # Calculate trade statistics
        # Buy/sell imbalance
        buy_volume = df[df['is_buy'] == 1]['size'].sum()
        sell_volume = df[df['is_buy'] == 0]['size'].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
        else:
            buy_ratio = 0.5  # Default to balanced
        
        # Call/put imbalance
        call_volume = df[df['is_call'] == 1]['size'].sum()
        put_volume = df[df['is_call'] == 0]['size'].sum()
        total_option_volume = call_volume + put_volume
        
        if total_option_volume > 0:
            call_ratio = call_volume / total_option_volume
        else:
            call_ratio = 0.5  # Default to balanced
        
        # Calculate buy call, buy put, sell call, sell put volumes
        buy_call = df[(df['is_buy'] == 1) & (df['is_call'] == 1)]['size'].sum()
        buy_put = df[(df['is_buy'] == 1) & (df['is_call'] == 0)]['size'].sum()
        sell_call = df[(df['is_buy'] == 0) & (df['is_call'] == 1)]['size'].sum()
        sell_put = df[(df['is_buy'] == 0) & (df['is_call'] == 0)]['size'].sum()
        
        # Average price and size
        avg_price = df['price'].mean()
        avg_size = df['size'].mean()
        
        # Price trend (last price - first price) / first price
        if len(df) > 1:
            price_trend = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
        else:
            price_trend = 0
        
        # Get last row features for individual trades
        last_price = df['price'].iloc[-1]
        last_size = df['size'].iloc[-1]
        last_is_buy = df['is_buy'].iloc[-1]
        last_is_call = df['is_call'].iloc[-1]
        
        # Compile features
        features = np.array([
            buy_ratio,
            call_ratio,
            buy_call / (total_volume + 1e-10),
            buy_put / (total_volume + 1e-10),
            sell_call / (total_volume + 1e-10),
            sell_put / (total_volume + 1e-10),
            price_trend,
            avg_price,
            avg_size,
            last_price,
            last_size,
            last_is_buy,
            last_is_call,
            total_volume
        ])
        
        # Reshape to 2D array
        return features.reshape(1, -1)
    
    def _process_data(self):
        """
        Process all order flow data to create features.
        
        Returns:
            np.ndarray: Processed features
        """
        # Process each window of data
        feature_list = []
        
        for i in range(len(self.order_flow_data) - self.window_size + 1):
            window = self.order_flow_data.iloc[i:i+self.window_size]
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
        
        # Initialize position
        self.current_step = 0
        self.current_position = 0  # 0 = no position, 1 = long call, 2 = long put
        self.entry_price = None
        self.rewards = []
        
        # Create initial state (window of features)
        self.state = self.features[0:self.window_size]
        
        return self.state
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action (int): Action to take (0 = no position, 1 = long call, 2 = long put)
            
        Returns:
            tuple: Next state, reward, done, info
        """
        # Get current price (use the last feature row's price value)
        current_price = self.features[self.current_step, 7]  # avg_price feature
        
        # Calculate reward based on action and price change
        next_step = min(self.current_step + 1, len(self.features) - 1)
        next_price = self.features[next_step, 7]
        price_change = (next_price - current_price) / current_price
        
        # Calculate reward
        reward = self._calculate_reward(action, price_change)
        self.rewards.append(reward)
        
        # Update position
        self.current_position = action
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new state
        if not done:
            state_idx = min(self.current_step, len(self.features) - self.window_size)
            self.state = self.features[state_idx:state_idx+self.window_size]
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'current_position': self.current_position,
            'cumulative_reward': sum(self.rewards)
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, action, price_change):
        """
        Calculate reward based on action and price change.
        
        Args:
            action (int): Action taken (0 = no position, 1 = long call, 2 = long put)
            price_change (float): Price change percentage
            
        Returns:
            float: Reward
        """
        if action == 0:  # No position
            reward = 0
        elif action == 1:  # Long call (bullish)
            reward = price_change * 100  # Amplify for more meaningful rewards
        elif action == 2:  # Long put (bearish)
            reward = -price_change * 100  # Negative price change is good for puts
        
        # Add transaction cost for changing positions
        if self.current_position != action:
            reward -= 0.1  # Small transaction cost
        
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


class RLOrderFlowModel:
    """
    Reinforcement learning model for analyzing option order flow data to predict market movements.
    
    Features:
    - PPO (Proximal Policy Optimization) for learning optimal trading actions
    - Custom gym environment for order flow data
    - State representation with order flow metrics
    - Reward function based on trading performance
    """
    
    def __init__(self, config=None):
        """
        Initialize the RLOrderFlowModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = os.path.join("models", "rl_order_flow")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.window_size = self.config.get('window_size', 10)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.n_steps = self.config.get('n_steps', 2048)  # Steps per update
        self.batch_size = self.config.get('batch_size', 64)
        self.n_epochs = self.config.get('n_epochs', 10)  # PPO epochs
        self.ppo_model = None
        
        # Flag to check if stable-baselines3 is available
        self.sb3_available = HAS_SB3
        if not self.sb3_available:
            self.logger.warning("stable-baselines3 not available. Using custom PPO implementation.")
        
        self.logger.info("RLOrderFlowModel initialized")
    
    def create_env(self, order_flow_data, window_size=None, max_steps=None):
        """
        Create a gym environment for order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            window_size (int, optional): Window size for state
            max_steps (int, optional): Maximum number of steps in an episode
            
        Returns:
            OrderFlowEnv: Gym environment
        """
        # Use specified window size or default
        if window_size is None:
            window_size = self.window_size
        
        # Create environment
        env = OrderFlowEnv(order_flow_data, window_size, max_steps)
        
        # Wrap in DummyVecEnv for stable-baselines3 compatibility
        if self.sb3_available:
            env = DummyVecEnv([lambda: env])
        
        return env
    
    def _build_policy_network(self, input_shape):
        """
        Build a policy network for PPO.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tuple: Actor and critic models
        """
        # Actor network (policy)
        actor_input = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(actor_input)
        x = BatchNormalization()(x)
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        action_probs = Dense(3, activation='softmax')(x)
        actor = Model(inputs=actor_input, outputs=action_probs)
        
        # Critic network (value function)
        critic_input = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(critic_input)
        x = BatchNormalization()(x)
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        value = Dense(1)(x)
        critic = Model(inputs=critic_input, outputs=value)
        
        return actor, critic
    
    def train(self, order_flow_data, n_steps=100000, model_name=None):
        """
        Train the PPO model on order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            n_steps (int): Number of training steps
            model_name (str, optional): Name for the saved model
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training RL model on order flow data, steps: {n_steps}")
        
        # Create environment
        env = self.create_env(order_flow_data)
        
        # Set default model_name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"ppo_order_flow_{timestamp}"
        
        # Create model save path
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
            if self.sb3_available:
                # Create PPO model using stable-baselines3
                self.ppo_model = PPO(
                    "MlpLstmPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    n_steps=self.n_steps,
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
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
                self.ppo_model.learn(
                    total_timesteps=n_steps,
                    callback=eval_callback
                )
                
                # Save final model
                self.ppo_model.save(os.path.join(model_path, "final_model"))
                
                # Evaluate policy
                mean_reward, std_reward = evaluate_policy(
                    self.ppo_model,
                    env,
                    n_eval_episodes=10
                )
                
                metrics = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'n_steps': n_steps,
                    'model_type': 'stable_baselines3_ppo'
                }
                
            else:
                # Use custom PPO implementation (simplified)
                self.logger.warning("Custom PPO implementation is a simplified version and may not be as effective.")
                
                # Get input shape from environment
                input_shape = (self.window_size, env.n_features)
                
                # Build policy networks
                self.actor, self.critic = self._build_policy_network(input_shape)
                
                # Compile models
                self.actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
                self.critic.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
                
                # Simple PPO loop
                n_episodes = 100  # Simplified
                rewards_history = []
                
                for episode in range(n_episodes):
                    state = env.reset()
                    done = False
                    episode_rewards = []
                    states, actions, rewards, values = [], [], [], []
                    
                    while not done:
                        # Predict action probabilities and value
                        action_probs = self.actor.predict(np.expand_dims(state, axis=0))[0]
                        value = self.critic.predict(np.expand_dims(state, axis=0))[0, 0]
                        
                        # Choose action
                        action = np.random.choice(3, p=action_probs)
                        
                        # Take action
                        next_state, reward, done, _ = env.step(action)
                        
                        # Store data
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        values.append(value)
                        episode_rewards.append(reward)
                        
                        # Move to next state
                        state = next_state
                    
                    # Calculate discounted rewards
                    discounted_rewards = []
                    cumulative_reward = 0
                    for reward in reversed(rewards):
                        cumulative_reward = reward + self.gamma * cumulative_reward
                        discounted_rewards.insert(0, cumulative_reward)
                    
                    # Convert to numpy arrays
                    states = np.array(states)
                    actions = np.array(actions)
                    discounted_rewards = np.array(discounted_rewards)
                    values = np.array(values)
                    
                    # Calculate advantages
                    advantages = discounted_rewards - values
                    
                    # Actor update
                    for _ in range(self.n_epochs):
                        self.actor.fit(
                            states, 
                            np.eye(3)[actions], 
                            sample_weight=advantages, 
                            verbose=0, 
                            batch_size=self.batch_size
                        )
                    
                    # Critic update
                    self.critic.fit(
                        states, 
                        discounted_rewards, 
                        verbose=0, 
                        batch_size=self.batch_size
                    )
                    
                    # Store episode rewards
                    rewards_history.append(sum(episode_rewards))
                    
                    # Log progress
                    if (episode + 1) % 10 == 0:
                        self.logger.info(f"Episode {episode + 1}/{n_episodes}, Mean Reward: {np.mean(rewards_history[-10:]):.4f}")
                
                # Save models
                os.makedirs(model_path, exist_ok=True)
                self.actor.save(os.path.join(model_path, "actor_model.h5"))
                self.critic.save(os.path.join(model_path, "critic_model.h5"))
                
                # Save config
                config = {
                    'window_size': self.window_size,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'n_steps': self.n_steps,
                    'batch_size': self.batch_size,
                    'n_epochs': self.n_epochs
                }
                
                with open(os.path.join(model_path, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                
                # Calculate metrics
                metrics = {
                    'mean_reward': np.mean(rewards_history[-10:]),
                    'std_reward': np.std(rewards_history[-10:]),
                    'n_steps': n_steps,
                    'model_type': 'custom_ppo'
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
                self.ppo_model = PPO.load(os.path.join(model_path, "final_model"))
                self.logger.info(f"Loaded stable-baselines3 PPO model from {model_path}")
                return True
                
            elif os.path.exists(os.path.join(model_path, "actor_model.h5")):
                # Load custom PPO model
                self.actor = load_model(os.path.join(model_path, "actor_model.h5"))
                self.critic = load_model(os.path.join(model_path, "critic_model.h5"))
                
                # Load config
                if os.path.exists(os.path.join(model_path, "config.json")):
                    with open(os.path.join(model_path, "config.json"), 'r') as f:
                        config = json.load(f)
                        self.window_size = config.get('window_size', self.window_size)
                        self.learning_rate = config.get('learning_rate', self.learning_rate)
                        self.gamma = config.get('gamma', self.gamma)
                        self.n_steps = config.get('n_steps', self.n_steps)
                        self.batch_size = config.get('batch_size', self.batch_size)
                        self.n_epochs = config.get('n_epochs', self.n_epochs)
                
                self.logger.info(f"Loaded custom PPO model from {model_path}")
                return True
                
            else:
                self.logger.error(f"No valid model found at {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, order_flow_data, n_steps=None, strategy='trading'):
        """
        Make predictions on order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            n_steps (int, optional): Number of steps to predict
            strategy (str): Prediction strategy ('trading' or 'sentiment')
            
        Returns:
            dict: Prediction results
        """
        if self.ppo_model is None and not hasattr(self, 'actor'):
            self.logger.error("No model loaded. Please train or load a model first.")
            return None
        
        self.logger.info("Making predictions on order flow data")
        
        try:
            # Create environment
            env = self.create_env(order_flow_data)
            
            # Set number of steps to predict
            if n_steps is None:
                n_steps = len(order_flow_data) - self.window_size - 1
            
            # Initialize state
            state = env.reset()
            
            # Storage for predictions
            actions = []
            rewards = []
            positions = []
            probabilities = []
            
            # Make predictions
            for step in range(n_steps):
                # Get action from model
                if self.sb3_available and self.ppo_model is not None:
                    action, _ = self.ppo_model.predict(state, deterministic=(strategy == 'trading'))
                    
                    # Get action probabilities (not directly available from SB3 predict)
                    probs = [0.33, 0.33, 0.33]  # Default placeholder
                    
                else:
                    # Use custom model
                    probs = self.actor.predict(np.expand_dims(state, axis=0))[0]
                    
                    if strategy == 'trading':
                        # Deterministic - choose highest probability action
                        action = np.argmax(probs)
                    else:
                        # Stochastic - sample from probability distribution
                        action = np.random.choice(3, p=probs)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store action, reward, and probability
                actions.append(int(action))
                rewards.append(float(reward))
                positions.append(info['current_position'])
                probabilities.append(probs)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate sentiment from action distribution
            action_counts = {0: 0, 1: 0, 2: 0}  # 0: neutral, 1: bullish, 2: bearish
            for action in actions:
                action_counts[action] += 1
            
            total_actions = len(actions)
            sentiment_score = 0
            
            if total_actions > 0:
                # Calculate sentiment score (-1 to 1)
                bullish_ratio = action_counts[1] / total_actions
                bearish_ratio = action_counts[2] / total_actions
                sentiment_score = bullish_ratio - bearish_ratio
            
            # Convert sentiment score to category
            if sentiment_score > 0.3:
                sentiment = "bullish"
            elif sentiment_score < -0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Create result dictionary
            timestamps = order_flow_data['date'].iloc[self.window_size:self.window_size + len(actions)] if 'date' in order_flow_data.columns else None
            
            results = {
                'actions': actions,
                'positions': positions,
                'rewards': rewards,
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'action_distribution': {
                    'neutral': action_counts[0] / total_actions if total_actions > 0 else 0,
                    'bullish': action_counts[1] / total_actions if total_actions > 0 else 0,
                    'bearish': action_counts[2] / total_actions if total_actions > 0 else 0
                },
                'mean_reward': np.mean(rewards) if rewards else 0,
                'total_reward': sum(rewards),
                'n_steps': len(actions),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if timestamps is not None:
                results['timestamps'] = timestamps.tolist()
            
            # Add trading signals based on actions
            action_to_signal = {0: 'hold', 1: 'buy_call', 2: 'buy_put'}
            results['signals'] = [action_to_signal[a] for a in actions]
            
            self.logger.info(f"Generated predictions for {len(actions)} steps with sentiment: {sentiment}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def visualize_predictions(self, order_flow_data, prediction_results, output_file=None):
        """
        Visualize order flow data with predicted actions.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            prediction_results (dict): Prediction results from predict method
            output_file (str, optional): Path to save the visualization
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Visualizing order flow predictions")
        
        try:
            # Extract predictions
            actions = prediction_results['actions']
            rewards = prediction_results['rewards']
            sentiment_score = prediction_results['sentiment_score']
            
            # Create figure
            fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # 1. Order flow with actions
            # Create dataframe for plotting
            df = order_flow_data.copy()
            
            # Ensure we have a date/time column
            if 'date' not in df.columns and 'time' not in df.columns:
                df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
            
            time_col = 'date' if 'date' in df.columns else 'time'
            
            # Select relevant data points for plotting
            start_idx = self.window_size
            end_idx = start_idx + len(actions)
            
            if end_idx > len(df):
                end_idx = len(df)
                actions = actions[:end_idx - start_idx]
                rewards = rewards[:end_idx - start_idx]
            
            # Plot price if available
            if 'price' in df.columns:
                axs[0].plot(df[time_col].iloc[start_idx:end_idx], df['price'].iloc[start_idx:end_idx], label='Price')
                
                # Add markers for buy call/put actions
                buy_call_indices = [i + start_idx for i, a in enumerate(actions) if a == 1]
                buy_put_indices = [i + start_idx for i, a in enumerate(actions) if a == 2]
                
                if buy_call_indices:
                    axs[0].scatter(df[time_col].iloc[buy_call_indices], df['price'].iloc[buy_call_indices], 
                                 color='green', marker='^', s=100, label='Buy Call')
                
                if buy_put_indices:
                    axs[0].scatter(df[time_col].iloc[buy_put_indices], df['price'].iloc[buy_put_indices], 
                                 color='red', marker='v', s=100, label='Buy Put')
                
                axs[0].set_title('Option Price with Predicted Actions')
                axs[0].set_ylabel('Price')
                axs[0].grid(True)
                axs[0].legend()
            else:
                # If no price data, plot action types
                action_values = np.array(actions)
                axs[0].plot(df[time_col].iloc[start_idx:end_idx], action_values, 'o-')
                axs[0].set_yticks([0, 1, 2])
                axs[0].set_yticklabels(['Hold', 'Buy Call', 'Buy Put'])
                axs[0].set_title('Predicted Actions')
                axs[0].grid(True)
            
            # 2. Plot volume/size if available
            if 'size' in df.columns:
                axs[1].bar(df[time_col].iloc[start_idx:end_idx], df['size'].iloc[start_idx:end_idx], color='blue', alpha=0.6)
                axs[1].set_title('Order Size')
                axs[1].set_ylabel('Size')
                axs[1].grid(True)
            else:
                # Plot cumulative rewards
                cumulative_rewards = np.cumsum(rewards)
                axs[1].plot(df[time_col].iloc[start_idx:end_idx], cumulative_rewards, label='Cumulative Reward', color='purple')
                axs[1].set_title('Cumulative Reward')
                axs[1].set_ylabel('Reward')
                axs[1].grid(True)
            
            # 3. Plot sentiment
            # Create a sentiment timeline
            window = min(10, len(actions))
            sentiment_timeline = []
            
            for i in range(len(actions)):
                start = max(0, i - window + 1)
                window_actions = actions[start:i+1]
                window_bullish = sum(1 for a in window_actions if a == 1)
                window_bearish = sum(1 for a in window_actions if a == 2)
                total = len(window_actions)
                
                if total > 0:
                    window_sentiment = (window_bullish - window_bearish) / total
                else:
                    window_sentiment = 0
                
                sentiment_timeline.append(window_sentiment)
            
            axs[2].plot(df[time_col].iloc[start_idx:end_idx], sentiment_timeline, 'g-', label='Sentiment')
            axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[2].set_title(f'Order Flow Sentiment (Score: {sentiment_score:.2f})')
            axs[2].set_ylabel('Sentiment Score')
            axs[2].set_xlabel('Time')
            axs[2].grid(True)
            
            # Add colored regions for sentiment
            axs[2].axhspan(0.3, 1, alpha=0.2, color='green', label='Bullish')
            axs[2].axhspan(-0.3, 0.3, alpha=0.2, color='gray', label='Neutral')
            axs[2].axhspan(-1, -0.3, alpha=0.2, color='red', label='Bearish')
            axs[2].legend()
            
            # Format time axis
            plt.gcf().autofmt_xdate()
            
            # Add overall title
            fig.suptitle('Order Flow Analysis', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save figure if output file specified
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved visualization to {output_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing predictions: {str(e)}")
            return None, None
    
    def analyze_order_flow_patterns(self, order_flow_data, lookback_period=None):
        """
        Analyze order flow patterns to detect unusual activity or significant patterns.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            lookback_period (int, optional): Period to analyze for patterns
            
        Returns:
            dict: Analysis results
        """
        self.logger.info("Analyzing order flow patterns")
        
        try:
            df = order_flow_data.copy()
            
            # Use the last N days/periods if lookback_period is specified
            if lookback_period is not None:
                if 'date' in df.columns:
                    last_date = df['date'].max()
                    start_date = last_date - pd.Timedelta(days=lookback_period)
                    df = df[df['date'] >= start_date]
                else:
                    # Use the last N rows
                    df = df.iloc[-lookback_period:]
            
            # 1. Calculate buy/sell imbalance
            if 'side' in df.columns:
                buy_volume = df[df['side'] == 'buy']['size'].sum()
                sell_volume = df[df['side'] == 'sell']['size'].sum()
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    buy_ratio = buy_volume / total_volume
                    sell_ratio = sell_volume / total_volume
                    buy_sell_imbalance = buy_ratio - sell_ratio
                else:
                    buy_ratio = 0.5
                    sell_ratio = 0.5
                    buy_sell_imbalance = 0
            else:
                buy_ratio = None
                sell_ratio = None
                buy_sell_imbalance = None
            
            # 2. Calculate call/put imbalance
            if 'option_type' in df.columns:
                call_volume = df[df['option_type'] == 'call']['size'].sum()
                put_volume = df[df['option_type'] == 'put']['size'].sum()
                total_option_volume = call_volume + put_volume
                
                if total_option_volume > 0:
                    call_ratio = call_volume / total_option_volume
                    put_ratio = put_volume / total_option_volume
                    call_put_imbalance = call_ratio - put_ratio
                else:
                    call_ratio = 0.5
                    put_ratio = 0.5
                    call_put_imbalance = 0
            else:
                call_ratio = None
                put_ratio = None
                call_put_imbalance = None
            
            # 3. Detect large trades
            if 'size' in df.columns:
                # Calculate mean and standard deviation of trade size
                mean_size = df['size'].mean()
                std_size = df['size'].std()
                
                # Identify large trades (> 2 standard deviations)
                large_trade_threshold = mean_size + 2 * std_size
                large_trades = df[df['size'] > large_trade_threshold]
                
                # Calculate percentage of large trades
                large_trade_percentage = len(large_trades) / len(df) if len(df) > 0 else 0
                
                # Analyze large trade direction
                if len(large_trades) > 0 and 'side' in large_trades.columns:
                    large_buy_volume = large_trades[large_trades['side'] == 'buy']['size'].sum()
                    large_sell_volume = large_trades[large_trades['side'] == 'sell']['size'].sum()
                    large_total_volume = large_buy_volume + large_sell_volume
                    
                    if large_total_volume > 0:
                        large_buy_ratio = large_buy_volume / large_total_volume
                        large_sell_ratio = large_sell_volume / large_total_volume
                        large_trade_imbalance = large_buy_ratio - large_sell_ratio
                    else:
                        large_buy_ratio = 0.5
                        large_sell_ratio = 0.5
                        large_trade_imbalance = 0
                else:
                    large_buy_ratio = None
                    large_sell_ratio = None
                    large_trade_imbalance = None
            else:
                large_trade_percentage = None
                large_buy_ratio = None
                large_sell_ratio = None
                large_trade_imbalance = None
            
            # 4. Analyze overall order flow sentiment
            # Use model to predict sentiment if available, otherwise use simple heuristics
            if self.ppo_model is not None or hasattr(self, 'actor'):
                # Use model predictions
                prediction_results = self.predict(order_flow_data, strategy='sentiment')
                sentiment = prediction_results['sentiment']
                sentiment_score = prediction_results['sentiment_score']
            else:
                # Use heuristic based on imbalances
                if buy_sell_imbalance is not None and call_put_imbalance is not None:
                    # Combine buy/sell and call/put signals
                    combined_score = (buy_sell_imbalance + call_put_imbalance) / 2
                    
                    if combined_score > 0.2:
                        sentiment = "bullish"
                        sentiment_score = combined_score
                    elif combined_score < -0.2:
                        sentiment = "bearish"
                        sentiment_score = combined_score
                    else:
                        sentiment = "neutral"
                        sentiment_score = combined_score
                elif buy_sell_imbalance is not None:
                    # Use only buy/sell imbalance
                    if buy_sell_imbalance > 0.2:
                        sentiment = "bullish"
                        sentiment_score = buy_sell_imbalance
                    elif buy_sell_imbalance < -0.2:
                        sentiment = "bearish"
                        sentiment_score = buy_sell_imbalance
                    else:
                        sentiment = "neutral"
                        sentiment_score = buy_sell_imbalance
                elif call_put_imbalance is not None:
                    # Use only call/put imbalance
                    if call_put_imbalance > 0.2:
                        sentiment = "bullish"
                        sentiment_score = call_put_imbalance
                    elif call_put_imbalance < -0.2:
                        sentiment = "bearish"
                        sentiment_score = call_put_imbalance
                    else:
                        sentiment = "neutral"
                        sentiment_score = call_put_imbalance
                else:
                    sentiment = "neutral"
                    sentiment_score = 0
            
            # 5. Compile analysis results
            results = {
                'buy_sell_imbalance': {
                    'buy_ratio': buy_ratio,
                    'sell_ratio': sell_ratio,
                    'imbalance': buy_sell_imbalance
                },
                'call_put_imbalance': {
                    'call_ratio': call_ratio,
                    'put_ratio': put_ratio,
                    'imbalance': call_put_imbalance
                },
                'large_trades': {
                    'percentage': large_trade_percentage,
                    'buy_ratio': large_buy_ratio,
                    'sell_ratio': large_sell_ratio,
                    'imbalance': large_trade_imbalance
                },
                'overall_sentiment': {
                    'sentiment': sentiment,
                    'score': sentiment_score
                },
                'total_volume': total_volume if 'total_volume' in locals() else None,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow patterns: {str(e)}")
            return None
