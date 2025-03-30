"""
Entry Controller Module

This module specializes in making option entry decisions as part of
the hierarchical reinforcement learning system. It determines the
optimal timing and conditions for entering option trades.
"""

import logging
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

class EntryController:
    """
    Reinforcement learning controller specialized for option entry decisions.
    Takes high-level strategy guidance from the MetaController and determines
    specific entry conditions and timing.
    """
    
    def __init__(self, config, reward_shaper=None, drive_manager=None):
        """
        Initialize the entry controller.
        
        Args:
            config (dict): Configuration dictionary
            reward_shaper: RewardShaper instance for reward shaping
            drive_manager: GoogleDriveManager for saving models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("sub_controllers", {}).get("entry_controller", {})
        self.reward_shaper = reward_shaper
        self.drive_manager = drive_manager
        
        # Extract configuration parameters
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.gamma = self.config.get("gamma", 0.95)  # Discount factor
        self.epsilon = self.config.get("initial_epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.1)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 64)
        self.update_target_every = self.config.get("update_target_every", 100)
        self.save_model_every = self.config.get("save_model_every", 500)
        self.memory_size = self.config.get("memory_size", 20000)
        self.time_steps = self.config.get("time_steps", 10)  # For sequence modeling
        
        # Define state and action spaces
        self.state_dim = self.config.get("state_dim", 30)  # Features per time step
        
        # Action space: 
        # 0: No action
        # 1: Enter call option
        # 2: Enter put option
        self.action_dim = 3
        
        # Set up memory buffer for experience replay
        self.memory = []
        self.memory_counter = 0
        
        # Build the models
        self._build_models()
        
        # Dictionary to track training stats
        self.training_stats = {
            "episodes": 0,
            "total_rewards": [],
            "avg_rewards": [],
            "losses": [],
            "epsilons": [],
            "action_distribution": {0: 0, 1: 0, 2: 0}
        }
        
        # Tracking current training state
        self.update_counter = 0
        self.last_state = None
        self.last_action = None
        self.episode_actions = []
        
        # Directory for models
        self.model_dir = "data/hierarchical_rl/sub_controllers"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing model if available
        self._load_models()
        
        # Strategy-specific models
        self.strategy_models = {}
        
        self.logger.info("EntryController initialized")
    
    def _build_models(self):
        """Build the primary and target networks."""
        try:
            # Main model for training
            self.main_model = self._create_model()
            
            # Target model for stable Q-targets
            self.target_model = self._create_model()
            self.target_model.set_weights(self.main_model.get_weights())
            
            self.logger.info("Built entry controller models")
            
        except Exception as e:
            self.logger.error(f"Error building entry controller models: {str(e)}")
            raise
    
    def _create_model(self):
        """Create a sequential deep Q-network model."""
        # For time-series data, use LSTM
        input_layer = Input(shape=(self.time_steps, self.state_dim))
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        # Strategy and current price inputs for additional context
        strategy_input = Input(shape=(5,))  # One-hot encoded strategy
        price_input = Input(shape=(3,))  # Current, historic avg, volatility
        
        # Concatenate LSTM output with strategy and price inputs
        combined = Concatenate()([x, strategy_input, price_input])
        
        # Dense layers
        x = Dense(64, activation='relu')(combined)
        x = Dense(32, activation='relu')(x)
        
        # Output layer (Q-values for actions)
        output_layer = Dense(self.action_dim, activation='linear')(x)
        
        # Create model with multiple inputs
        model = Model(
            inputs=[input_layer, strategy_input, price_input], 
            outputs=output_layer
        )
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _load_models(self):
        """Load saved models if they exist."""
        try:
            model_path = f"{self.model_dir}/entry_controller_model.h5"
            
            if os.path.exists(model_path):
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded saved entry controller model")
                
                # Load training stats
                stats_path = f"{self.model_dir}/entry_controller_stats.json"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.training_stats = json.load(f)
            
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/sub_controllers/entry_controller_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary("hierarchical_rl/sub_controllers/entry_controller_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded entry controller model from Google Drive")
                
                # Load training stats
                if self.drive_manager.file_exists("hierarchical_rl/sub_controllers/entry_controller_stats.json"):
                    stats_data = self.drive_manager.download_file("hierarchical_rl/sub_controllers/entry_controller_stats.json")
                    self.training_stats = json.loads(stats_data)
                    
                    with open(f"{self.model_dir}/entry_controller_stats.json", 'w') as f:
                        f.write(stats_data)
                        
        except Exception as e:
            self.logger.error(f"Error loading entry controller model: {str(e)}")
            self.logger.info("Starting with new entry controller model")
    
    def _save_models(self):
        """Save the models and training stats."""
        try:
            model_path = f"{self.model_dir}/entry_controller_model.h5"
            stats_path = f"{self.model_dir}/entry_controller_stats.json"
            
            # Save model locally
            self.main_model.save(model_path)
            
            # Save training stats
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(model_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/entry_controller_model.h5",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
                with open(stats_path, 'r') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/entry_controller_stats.json",
                        f.read(),
                        mime_type="application/json"
                    )
            
            self.logger.info(f"Saved entry controller model after {self.training_stats['episodes']} episodes")
            
        except Exception as e:
            self.logger.error(f"Error saving entry controller model: {str(e)}")
    
    def _format_state(self, state_features, strategy):
        """
        Format state features into the format required by the model.
        
        Args:
            state_features (dict): Dictionary of state features
            strategy (str): Current trading strategy
            
        Returns:
            tuple: (time_series_data, strategy_encoded, price_features)
        """
        # Extract time series data (assumes features include time series history)
        time_series = state_features.get("time_series", [])
        
        # If not enough time steps, pad with zeros
        if len(time_series) < self.time_steps:
            padding = [np.zeros(self.state_dim) for _ in range(self.time_steps - len(time_series))]
            time_series = padding + time_series
        
        # If too many time steps, take the most recent ones
        if len(time_series) > self.time_steps:
            time_series = time_series[-self.time_steps:]
        
        # Convert to numpy array
        time_series_data = np.array(time_series).reshape(1, self.time_steps, self.state_dim)
        
        # One-hot encode strategy
        strategies = ["momentum", "mean_reversion", "volatility_breakout", "gamma_scalping", "theta_harvesting"]
        strategy_encoded = np.zeros((1, 5))
        if strategy in strategies:
            strategy_encoded[0, strategies.index(strategy)] = 1
        
        # Extract current price features
        price_features = np.array([
            state_features.get("current_price", 0.0),
            state_features.get("historical_avg_price", 0.0),
            state_features.get("price_volatility", 0.0)
        ]).reshape(1, 3)
        
        return time_series_data, strategy_encoded, price_features
    
    def decide_entry(self, state_features, strategy, evaluate=False):
        """
        Decide whether to enter a trade based on current conditions.
        
        Args:
            state_features (dict): Features describing the current market state
            strategy (str): Current trading strategy from meta controller
            evaluate (bool): If True, use greedy policy (no exploration)
            
        Returns:
            int: Action (0: no action, 1: enter call, 2: enter put)
        """
        # Format state for model input
        time_series, strategy_enc, price_features = self._format_state(state_features, strategy)
        
        # Store for learning
        self.last_state = (time_series.copy(), strategy_enc.copy(), price_features.copy())
        
        # Epsilon-greedy action selection
        if not evaluate and np.random.rand() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Exploitation: select best action
            q_values = self.main_model.predict([time_series, strategy_enc, price_features])[0]
            action = np.argmax(q_values)
        
        # Store selected action
        self.last_action = action
        self.episode_actions.append(action)
        
        # Update action distribution stats
        self.training_stats["action_distribution"][action] += 1
        
        return action
    
    def record_experience(self, next_state_features, strategy, reward, done):
        """
        Record experience for learning.
        
        Args:
            next_state_features (dict): Next state features
            strategy (str): Current trading strategy
            reward (float): Reward received
            done (bool): Whether episode is done
        """
        if self.last_state is None or self.last_action is None:
            return
        
        # Format next state
        next_time_series, next_strategy_enc, next_price_features = self._format_state(next_state_features, strategy)
        
        # Apply reward shaping if available
        if self.reward_shaper:
            shaped_reward = self.reward_shaper.shape_entry_reward(
                self.last_state, self.last_action, 
                (next_time_series, next_strategy_enc, next_price_features), 
                reward, strategy
            )
        else:
            shaped_reward = reward
        
        # Store in replay memory
        experience = (
            self.last_state, 
            self.last_action, 
            shaped_reward, 
            (next_time_series, next_strategy_enc, next_price_features), 
            done
        )
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.memory_counter % self.memory_size] = experience
            
        self.memory_counter += 1
    
    def train(self):
        """
        Train the model using experience replay.
        
        Returns:
            float: Loss value if trained, None otherwise
        """
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch from memory
        indices = np.random.choice(min(len(self.memory), self.memory_size), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Extract components
        time_series_batch = np.vstack([experience[0][0] for experience in batch])
        strategy_batch = np.vstack([experience[0][1] for experience in batch])
        price_batch = np.vstack([experience[0][2] for experience in batch])
        
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        
        next_time_series_batch = np.vstack([experience[3][0] for experience in batch])
        next_strategy_batch = np.vstack([experience[3][1] for experience in batch])
        next_price_batch = np.vstack([experience[3][2] for experience in batch])
        
        dones = np.array([experience[4] for experience in batch], dtype=bool)
        
        # Compute Q-targets
        q_targets = self.main_model.predict([time_series_batch, strategy_batch, price_batch])
        q_next = self.target_model.predict([next_time_series_batch, next_strategy_batch, next_price_batch])
        
        # Update Q-targets for selected actions
        for i in range(self.batch_size):
            if dones[i]:
                q_targets[i, actions[i]] = rewards[i]
            else:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        # Train the model
        history = self.main_model.fit(
            [time_series_batch, strategy_batch, price_batch], 
            q_targets, 
            epochs=1, 
            verbose=0
        )
        loss = history.history['loss'][0]
        
        # Update target model periodically
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_model.set_weights(self.main_model.get_weights())
            self.logger.debug("Updated target model weights")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Save model periodically
        if self.update_counter % self.save_model_every == 0:
            self._save_models()
        
        # Update training stats
        self.training_stats["losses"].append(float(loss))
        self.training_stats["epsilons"].append(float(self.epsilon))
        
        return loss
    
    def end_episode(self, total_reward):
        """
        Signal the end of an episode and update training stats.
        
        Args:
            total_reward (float): Total reward for the episode
        """
        # Update episode counter
        self.training_stats["episodes"] += 1
        
        # Record reward
        self.training_stats["total_rewards"].append(total_reward)
        
        # Calculate moving average of rewards
        window_size = min(100, len(self.training_stats["total_rewards"]))
        avg_reward = np.mean(self.training_stats["total_rewards"][-window_size:])
        self.training_stats["avg_rewards"].append(float(avg_reward))
        
        # Log progress
        episode = self.training_stats["episodes"]
        if episode % 10 == 0:
            self.logger.info(
                f"Entry Controller - Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {self.epsilon:.4f}"
            )
        
        # Reset episode variables
        self.last_state = None
        self.last_action = None
        self.episode_actions = []
        
        # Save models periodically
        if episode % self.save_model_every == 0:
            self._save_models()
    
    def get_q_values(self, state_features, strategy):
        """
        Get Q-values for all actions in the current state.
        
        Args:
            state_features (dict): Current state features
            strategy (str): Current trading strategy
            
        Returns:
            list: Q-values for each action
        """
        # Format state for model input
        time_series, strategy_enc, price_features = self._format_state(state_features, strategy)
        
        # Get Q-values from model
        q_values = self.main_model.predict([time_series, strategy_enc, price_features])[0]
        
        return q_values.tolist()
    
    def load_strategy_specific_model(self, strategy):
        """
        Load a strategy-specific model if available.
        
        Args:
            strategy (str): Strategy name
            
        Returns:
            bool: Success flag
        """
        # Check if we already have this model loaded
        if strategy in self.strategy_models:
            # Set as current main model
            self.main_model = self.strategy_models[strategy]
            self.target_model.set_weights(self.main_model.get_weights())
            self.logger.info(f"Switched to {strategy} specific entry model")
            return True
        
        # Try to load from disk
        model_path = f"{self.model_dir}/entry_{strategy}_model.h5"
        
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific entry model")
                return True
            
            elif self.drive_manager and self.drive_manager.file_exists(f"hierarchical_rl/sub_controllers/entry_{strategy}_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary(f"hierarchical_rl/sub_controllers/entry_{strategy}_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific entry model from Google Drive")
                return True
            
            else:
                self.logger.info(f"No strategy-specific model found for {strategy}, using general model")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading strategy-specific model for {strategy}: {str(e)}")
            return False
