"""
RL Time Sales Model Module

This module implements reinforcement learning models for analyzing time & sales data
to learn optimal trading actions based on high-frequency trade patterns.
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

# Import for DQN (Deep Q-Network)
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    

class TimeSalesEnv(gym.Env):
    """
    Custom gym environment for time & sales data.
    
    This environment takes time & sales data and creates a RL environment
    where the agent can learn to make trading decisions based on trade patterns.
    """
    
    def __init__(self, time_sales_data, lookback_window=20, max_steps=None):
        """
        Initialize the environment.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            lookback_window (int): Number of past time steps to include in the state
            max_steps (int, optional): Maximum number of steps in an episode
        """
        super(TimeSalesEnv, self).__init__()
        
        # Store data
        self.time_sales_data = time_sales_data
        self.lookback_window = lookback_window
        
        # Set max_steps
        if max_steps is None:
            self.max_steps = len(time_sales_data) - lookback_window - 1
        else:
            self.max_steps = min(max_steps, len(time_sales_data) - lookback_window - 1)
        
        # Define action space: 0 = no action, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (state)
        # Features: price, volume, direction, time gap
        self.n_features = 4
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback_window, self.n_features),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _preprocess_data(self):
        """
        Preprocess time & sales data to create features.
        
        Returns:
            np.ndarray: Processed features
        """
        df = self.time_sales_data.copy()
        
        # Ensure time is datetime type
        if not pd.api.types.is_datetime64_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Sort by time
        df = df.sort_values('time')
        
        # Calculate price changes
        df['price_change'] = df['price'].pct_change()
        
        # Calculate trade direction (-1 for down, 0 for same, 1 for up)
        df['direction'] = np.sign(df['price_change'])
        
        # Calculate time gap between trades (in seconds)
        df['time_gap'] = df['time'].diff().dt.total_seconds()
        
        # Normalize features
        # Price: normalize by dividing by the first price
        df['price_norm'] = df['price'] / df['price'].iloc[0]
        
        # Volume: log transform and normalize
        df['volume_norm'] = np.log1p(df['size']) / np.log1p(df['size'].max())
        
        # Direction: already normalized (-1 to 1)
        # Time gap: normalize by dividing by the maximum gap
        df['time_gap_norm'] = df['time_gap'] / df['time_gap'].max()
        
        # Create feature matrix
        features = np.column_stack([
            df['price_norm'].values,
            df['volume_norm'].values,
            df['direction'].values,
            df['time_gap_norm'].values
        ])
        
        return features
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            np.ndarray: Initial state
        """
        # Preprocess data
        self.features = self._preprocess_data()
        
        # Initialize position
        self.current_step = 0
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.position_entry_step = None
        self.rewards = []
        
        # Create initial state
        self.state = self.features[0:self.lookback_window]
        
        return self.state
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action (int): Action to take (0 = hold, 1 = buy, 2 = sell)
            
        Returns:
            tuple: Next state, reward, done, info
        """
        # Convert action to position
        if action == 0:
            new_position = 0  # No position
        elif action == 1:
            new_position = 1  # Long
        else:  # action == 2
            new_position = -1  # Short
        
        # Calculate reward
        reward = self._calculate_reward(new_position)
        self.rewards.append(reward)
        
        # Update current position
        self.current_position = new_position
        if new_position != 0:
            self.position_entry_step = self.current_step
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new state
        if not done:
            next_idx = self.current_step + self.lookback_window
            self.state = self.features[self.current_step:next_idx]
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'current_position': self.current_position,
            'cumulative_reward': sum(self.rewards)
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, new_position):
        """
        Calculate reward based on position and price changes.
        
        Args:
            new_position (int): New position (-1, 0, 1)
            
        Returns:
            float: Reward
        """
        # Get price at current step and next step
        current_idx = self.current_step + self.lookback_window - 1
        next_idx = self.current_step + self.lookback_window
        
        if next_idx >= len(self.features):
            next_idx = current_idx
        
        current_price = self.features[current_idx, 0]  # Normalized price
        next_price = self.features[next_idx, 0]
        
        # Calculate price change
        price_change = next_price - current_price
        
        # Reward is based on position and price movement
        position_reward = 0
        
        # If we're holding a position, reward/penalize based on price change
        if self.current_position == 1:  # Long position
            position_reward = price_change * 100  # Scale up for more meaningful rewards
        elif self.current_position == -1:  # Short position
            position_reward = -price_change * 100
        
        # Add transaction cost for changing positions
        transaction_cost = 0
        if self.current_position != new_position:
            transaction_cost = 0.01  # Small fixed cost for changing positions
        
        # Final reward is position reward minus transaction cost
        reward = position_reward - transaction_cost
        
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


class RLTimeSalesModel:
    """
    Reinforcement learning model for analyzing time & sales data to predict short-term price movements.
    
    Features:
    - DQN (Deep Q-Network) for learning optimal trading actions
    - Custom gym environment for time & sales data
    - State representation with lookback window for temporal patterns
    - Reward function based on trading performance
    """
    
    def __init__(self, config=None):
        """
        Initialize the RLTimeSalesModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = os.path.join("models", "rl_time_sales")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.lookback_window = self.config.get('lookback_window', 20)
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 64)
        self.replay_buffer_size = self.config.get('replay_buffer_size', 10000)
        self.target_update_freq = self.config.get('target_update_freq', 1000)
        self.training_interval = self.config.get('training_interval', 4)
        self.dqn_model = None
        
        # Flag to check if stable-baselines3 is available
        self.sb3_available = HAS_SB3
        if not self.sb3_available:
            self.logger.warning("stable-baselines3 not available. Using custom DQN implementation.")
        
        self.logger.info("RLTimeSalesModel initialized")
    
    def _build_q_network(self, input_shape):
        """
        Build a Q-network for DQN.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tf.keras.Model: Q-network model
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            LSTM(32),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)  # 3 actions: no action, buy, sell
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model
    
    def create_env(self, time_sales_data, lookback_window=None, max_steps=None):
        """
        Create a gym environment for time & sales data.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            lookback_window (int, optional): Number of past time steps to include in the state
            max_steps (int, optional): Maximum number of steps in an episode
            
        Returns:
            TimeSalesEnv: Gym environment
        """
        # Use specified lookback window or default
        if lookback_window is None:
            lookback_window = self.lookback_window
        
        # Create environment
        env = TimeSalesEnv(time_sales_data, lookback_window, max_steps)
        
        # Wrap in DummyVecEnv for stable-baselines3 compatibility
        if self.sb3_available:
            env = DummyVecEnv([lambda: env])
        
        return env
    
    def train(self, time_sales_data, n_steps=100000, model_name=None):
        """
        Train the DQN model on time & sales data.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            n_steps (int): Number of training steps
            model_name (str, optional): Name for the saved model
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training RL model on time & sales data, steps: {n_steps}")
        
        # Create environment
        env = self.create_env(time_sales_data)
        
        # Set default model_name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"dqn_time_sales_{timestamp}"
        
        # Create model save path
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
            if self.sb3_available:
                # Create DQN model using stable-baselines3
                self.dqn_model = DQN(
                    "MlpLstmPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    batch_size=self.batch_size,
                    buffer_size=self.replay_buffer_size,
                    target_update_interval=self.target_update_freq,
                    train_freq=self.training_interval,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.01,
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
                self.dqn_model.learn(
                    total_timesteps=n_steps,
                    callback=eval_callback
                )
                
                # Save final model
                self.dqn_model.save(os.path.join(model_path, "final_model"))
                
                # Evaluate policy
                mean_reward, std_reward = evaluate_policy(
                    self.dqn_model,
                    env,
                    n_eval_episodes=10
                )
                
                metrics = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'n_steps': n_steps,
                    'model_type': 'stable_baselines3_dqn'
                }
                
            else:
                # Use custom DQN implementation (simplified)
                # Implement a basic DQN algorithm
                # This is a simplified version and may not be as efficient as stable-baselines3
                
                # Get input shape from environment
                input_shape = (self.lookback_window, env.n_features)
                
                # Build Q-networks
                self.q_network = self._build_q_network(input_shape)
                self.target_network = self._build_q_network(input_shape)
                
                # Initialize target network with same weights
                self.target_network.set_weights(self.q_network.get_weights())
                
                # Initialize replay buffer (simple list-based)
                replay_buffer = []
                
                # Training loop
                state = env.reset()
                total_rewards = []
                episode_rewards = []
                
                for step in range(n_steps):
                    # Epsilon-greedy action selection
                    epsilon = max(0.01, 0.5 - 0.45 * (step / n_steps))
                    
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
                        action = np.argmax(q_values)
                    
                    # Take action
                    next_state, reward, done, _ = env.step(action)
                    episode_rewards.append(reward)
                    
                    # Store transition in replay buffer
                    replay_buffer.append((state, action, reward, next_state, done))
                    if len(replay_buffer) > self.replay_buffer_size:
                        replay_buffer.pop(0)
                    
                    # Update state
                    state = next_state
                    
                    # Train q_network
                    if step % self.training_interval == 0 and len(replay_buffer) >= self.batch_size:
                        # Sample batch
                        batch_indices = np.random.choice(len(replay_buffer), self.batch_size, replace=False)
                        batch = [replay_buffer[i] for i in batch_indices]
                        
                        # Unpack batch
                        states, actions, rewards, next_states, dones = zip(*batch)
                        states = np.array(states)
                        actions = np.array(actions)
                        rewards = np.array(rewards)
                        next_states = np.array(next_states)
                        dones = np.array(dones)
                        
                        # Compute targets
                        next_q_values = self.target_network.predict(next_states)
                        max_next_q = np.max(next_q_values, axis=1)
                        targets = rewards + self.gamma * max_next_q * (1 - dones)
                        
                        # Get current q values
                        current_q = self.q_network.predict(states)
                        
                        # Update targets for actions taken
                        for i in range(self.batch_size):
                            current_q[i, actions[i]] = targets[i]
                        
                        # Train network
                        self.q_network.fit(states, current_q, verbose=0)
                    
                    # Update target network
                    if step % self.target_update_freq == 0:
                        self.target_network.set_weights(self.q_network.get_weights())
                    
                    # Handle episode end
                    if done:
                        total_rewards.append(sum(episode_rewards))
                        state = env.reset()
                        episode_rewards = []
                        
                        # Log progress every 10 episodes
                        if len(total_rewards) % 10 == 0:
                            mean_reward = np.mean(total_rewards[-10:])
                            self.logger.info(f"Step {step}, Mean reward (last 10 episodes): {mean_reward:.4f}")
                
                # Save final model
                os.makedirs(model_path, exist_ok=True)
                self.q_network.save(os.path.join(model_path, "q_network.h5"))
                
                # Save model config
                config = {
                    'lookback_window': self.lookback_window,
                    'gamma': self.gamma,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'replay_buffer_size': self.replay_buffer_size,
                    'target_update_freq': self.target_update_freq,
                    'training_interval': self.training_interval
                }
                
                with open(os.path.join(model_path, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                
                # Calculate final metrics
                metrics = {
                    'mean_reward': np.mean(total_rewards[-10:]),
                    'std_reward': np.std(total_rewards[-10:]),
                    'n_steps': n_steps,
                    'model_type': 'custom_dqn'
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
                self.dqn_model = DQN.load(os.path.join(model_path, "final_model"))
                self.logger.info(f"Loaded stable-baselines3 DQN model from {model_path}")
                return True
                
            elif os.path.exists(os.path.join(model_path, "q_network.h5")):
                # Load custom DQN model
                self.q_network = load_model(os.path.join(model_path, "q_network.h5"))
                
                # Load config
                if os.path.exists(os.path.join(model_path, "config.json")):
                    with open(os.path.join(model_path, "config.json"), 'r') as f:
                        config = json.load(f)
                        self.lookback_window = config.get('lookback_window', self.lookback_window)
                        self.gamma = config.get('gamma', self.gamma)
                        self.learning_rate = config.get('learning_rate', self.learning_rate)
                        self.batch_size = config.get('batch_size', self.batch_size)
                        self.replay_buffer_size = config.get('replay_buffer_size', self.replay_buffer_size)
                        self.target_update_freq = config.get('target_update_freq', self.target_update_freq)
                        self.training_interval = config.get('training_interval', self.training_interval)
                
                self.logger.info(f"Loaded custom DQN model from {model_path}")
                return True
                
            else:
                self.logger.error(f"No valid model found at {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, time_sales_data, n_steps=None):
        """
        Make trading action predictions based on time & sales data.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            n_steps (int, optional): Number of steps to predict
            
        Returns:
            dict: Prediction results
        """
        if self.dqn_model is None and not hasattr(self, 'q_network'):
            self.logger.error("No model loaded. Please train or load a model first.")
            return None
        
        self.logger.info("Making predictions on time & sales data")
        
        try:
            # Create environment
            env = self.create_env(time_sales_data)
            
            # Set number of steps to predict
            if n_steps is None:
                n_steps = len(time_sales_data) - self.lookback_window - 1
            
            # Initialize state
            state = env.reset()
            
            # Storage for predictions
            actions = []
            rewards = []
            positions = []
            
            # Make predictions
            for step in range(n_steps):
                # Get action from model
                if self.sb3_available and self.dqn_model is not None:
                    action, _ = self.dqn_model.predict(state, deterministic=True)
                else:
                    q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
                    action = np.argmax(q_values)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store action and reward
                actions.append(int(action))
                rewards.append(float(reward))
                positions.append(info['current_position'])
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate performance metrics
            total_reward = sum(rewards)
            mean_reward = np.mean(rewards) if rewards else 0
            
            # Create result dictionary
            prediction_times = time_sales_data['time'].iloc[self.lookback_window:self.lookback_window + len(actions)]
            
            results = {
                'actions': actions,
                'positions': positions,
                'rewards': rewards,
                'total_reward': total_reward,
                'mean_reward': mean_reward,
                'n_steps': len(actions),
                'timestamps': prediction_times.tolist(),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add trading signals based on actions
            action_to_signal = {0: 'hold', 1: 'buy', 2: 'sell'}
            results['signals'] = [action_to_signal[a] for a in actions]
            
            self.logger.info(f"Generated predictions for {len(actions)} steps with total reward: {total_reward:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def visualize_predictions(self, time_sales_data, prediction_results, window=None, output_file=None):
        """
        Visualize time & sales data with predicted actions.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            prediction_results (dict): Prediction results from predict method
            window (int, optional): Number of recent data points to visualize
            output_file (str, optional): Path to save the visualization
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Visualizing time & sales predictions")
        
        try:
            # Extract predictions
            actions = prediction_results['actions']
            positions = prediction_results['positions']
            rewards = prediction_results['rewards']
            
            # Get price data from time & sales
            df = time_sales_data.copy()
            
            # Ensure time is datetime type
            if not pd.api.types.is_datetime64_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Sort by time
            df = df.sort_values('time')
            
            # Align predictions with data
            start_idx = self.lookback_window
            end_idx = start_idx + len(actions)
            
            if end_idx > len(df):
                end_idx = len(df)
                actions = actions[:end_idx - start_idx]
                positions = positions[:end_idx - start_idx]
                rewards = rewards[:end_idx - start_idx]
            
            # Select window of data to visualize
            if window and window < len(actions):
                start_idx = start_idx + len(actions) - window
                actions = actions[-window:]
                positions = positions[-window:]
                rewards = rewards[-window:]
            
            # Select relevant data points
            plot_df = df.iloc[start_idx:end_idx].copy()
            plot_df['action'] = actions
            plot_df['position'] = positions
            plot_df['reward'] = rewards
            
            # Calculate cumulative rewards
            plot_df['cumulative_reward'] = np.cumsum(rewards)
            
            # Create visualization
            fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            # Plot price
            axs[0].plot(plot_df['time'], plot_df['price'], label='Price')
            
            # Add colored background for positions
            for i in range(len(plot_df) - 1):
                if plot_df['position'].iloc[i] == 1:  # Long position
                    axs[0].axvspan(plot_df['time'].iloc[i], plot_df['time'].iloc[i+1], 
                                alpha=0.2, color='green')
                elif plot_df['position'].iloc[i] == -1:  # Short position
                    axs[0].axvspan(plot_df['time'].iloc[i], plot_df['time'].iloc[i+1], 
                                alpha=0.2, color='red')
            
            # Add markers for buy/sell actions
            buy_indices = plot_df[plot_df['action'] == 1].index
            sell_indices = plot_df[plot_df['action'] == 2].index
            
            axs[0].scatter(plot_df.loc[buy_indices, 'time'], plot_df.loc[buy_indices, 'price'], 
                         color='green', marker='^', s=100, label='Buy')
            axs[0].scatter(plot_df.loc[sell_indices, 'time'], plot_df.loc[sell_indices, 'price'], 
                         color='red', marker='v', s=100, label='Sell')
            
            axs[0].set_title('Time & Sales Price with Actions')
            axs[0].set_ylabel('Price')
            axs[0].grid(True)
            axs[0].legend()
            
            # Plot volume/size
            axs[1].bar(plot_df['time'], plot_df['size'], color='blue', alpha=0.6)
            axs[1].set_title('Trade Size')
            axs[1].set_ylabel('Size')
            axs[1].grid(True)
            
            # Plot rewards
            axs[2].plot(plot_df['time'], plot_df['cumulative_reward'], label='Cumulative Reward', color='purple')
            axs[2].set_title('Cumulative Reward')
            axs[2].set_ylabel('Reward')
            axs[2].set_xlabel('Time')
            axs[2].grid(True)
            
            # Format time axis
            plt.gcf().autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved visualization to {output_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing predictions: {str(e)}")
            return None, None
