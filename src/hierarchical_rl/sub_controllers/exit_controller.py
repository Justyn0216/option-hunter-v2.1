"""
Exit Controller Module

This module specializes in making option exit decisions as part of
the hierarchical reinforcement learning system. It determines the
optimal timing and conditions for exiting option trades.
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
from tensorflow.keras.layers import Dense, Input, LSTM, GRU, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

class ExitController:
    """
    Reinforcement learning controller specialized for option exit decisions.
    Determines optimal exit timing and conditions based on trade progression,
    market conditions, and option characteristics.
    """
    
    def __init__(self, config, reward_shaper=None, drive_manager=None):
        """
        Initialize the exit controller.
        
        Args:
            config (dict): Configuration dictionary
            reward_shaper: RewardShaper instance for reward shaping
            drive_manager: GoogleDriveManager for saving models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("sub_controllers", {}).get("exit_controller", {})
        self.reward_shaper = reward_shaper
        self.drive_manager = drive_manager
        
        # Extract configuration parameters
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.gamma = self.config.get("gamma", 0.97)  # Higher gamma for exit (future rewards matter more)
        self.epsilon = self.config.get("initial_epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 64)
        self.update_target_every = self.config.get("update_target_every", 100)
        self.save_model_every = self.config.get("save_model_every", 500)
        self.memory_size = self.config.get("memory_size", 20000)
        self.time_steps = self.config.get("time_steps", 15)  # More history for exit decisions
        
        # Define state and action spaces
        self.state_dim = self.config.get("state_dim", 35)  # Features per time step
        
        # Action space:
        # 0: Hold position
        # 1: Exit position
        # 2: Partial exit (scale out)
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
            "action_distribution": {0: 0, 1: 0, 2: 0},
            "avg_hold_time": [],
            "pnl_at_exit": []
        }
        
        # Tracking current training state
        self.update_counter = 0
        self.last_state = None
        self.last_action = None
        self.episode_actions = []
        self.hold_time = 0
        
        # Directory for models
        self.model_dir = "data/hierarchical_rl/sub_controllers"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing model if available
        self._load_models()
        
        # Strategy-specific models
        self.strategy_models = {}
        
        self.logger.info("ExitController initialized")
    
    def _build_models(self):
        """Build the primary and target networks."""
        try:
            # Main model for training
            self.main_model = self._create_model()
            
            # Target model for stable Q-targets
            self.target_model = self._create_model()
            self.target_model.set_weights(self.main_model.get_weights())
            
            self.logger.info("Built exit controller models")
            
        except Exception as e:
            self.logger.error(f"Error building exit controller models: {str(e)}")
            raise
    
    def _create_model(self):
        """Create a model for option exit timing."""
        # For time-series data of price and indicators
        price_input = Input(shape=(self.time_steps, self.state_dim))
        
        # Use GRU for sequence modeling with attention to recent values
        x = GRU(128, return_sequences=True)(price_input)
        x = Dropout(0.2)(x)
        x = GRU(64)(x)
        x = Dropout(0.2)(x)
        
        # Trade-specific features
        trade_input = Input(shape=(10,))  # Entry price, current P&L, time in trade, etc.
        
        # Option characteristics
        option_input = Input(shape=(8,))  # Greeks, days to expiration, etc.
        
        # Strategy context
        strategy_input = Input(shape=(5,))  # One-hot encoded strategy
        
        # Combine all inputs
        combined = Concatenate()([x, trade_input, option_input, strategy_input])
        
        # Dense layers
        x = Dense(128, activation='relu')(combined)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer (Q-values for actions: hold, exit, partial exit)
        output_layer = Dense(self.action_dim, activation='linear')(x)
        
        # Create model with multiple inputs
        model = Model(
            inputs=[price_input, trade_input, option_input, strategy_input], 
            outputs=output_layer
        )
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _load_models(self):
        """Load saved models if they exist."""
        try:
            model_path = f"{self.model_dir}/exit_controller_model.h5"
            
            if os.path.exists(model_path):
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded saved exit controller model")
                
                # Load training stats
                stats_path = f"{self.model_dir}/exit_controller_stats.json"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.training_stats = json.load(f)
            
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/sub_controllers/exit_controller_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary("hierarchical_rl/sub_controllers/exit_controller_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded exit controller model from Google Drive")
                
                # Load training stats
                if self.drive_manager.file_exists("hierarchical_rl/sub_controllers/exit_controller_stats.json"):
                    stats_data = self.drive_manager.download_file("hierarchical_rl/sub_controllers/exit_controller_stats.json")
                    self.training_stats = json.loads(stats_data)
                    
                    with open(f"{self.model_dir}/exit_controller_stats.json", 'w') as f:
                        f.write(stats_data)
                        
        except Exception as e:
            self.logger.error(f"Error loading exit controller model: {str(e)}")
            self.logger.info("Starting with new exit controller model")
    
    def _save_models(self):
        """Save the models and training stats."""
        try:
            model_path = f"{self.model_dir}/exit_controller_model.h5"
            stats_path = f"{self.model_dir}/exit_controller_stats.json"
            
            # Save model locally
            self.main_model.save(model_path)
            
            # Save training stats
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(model_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/exit_controller_model.h5",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
                with open(stats_path, 'r') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/exit_controller_stats.json",
                        f.read(),
                        mime_type="application/json"
                    )
            
            self.logger.info(f"Saved exit controller model after {self.training_stats['episodes']} episodes")
            
        except Exception as e:
            self.logger.error(f"Error saving exit controller model: {str(e)}")
    
    def _format_state(self, state_features, strategy, trade_info):
        """
        Format state features into the format required by the model.
        
        Args:
            state_features (dict): Dictionary of state features
            strategy (str): Current trading strategy
            trade_info (dict): Information about the current trade
            
        Returns:
            tuple: (price_series, trade_features, option_features, strategy_encoded)
        """
        # Extract time series data (assumes features include time series history)
        price_series = state_features.get("price_series", [])
        
        # If not enough time steps, pad with zeros
        if len(price_series) < self.time_steps:
            padding = [np.zeros(self.state_dim) for _ in range(self.time_steps - len(price_series))]
            price_series = padding + price_series
        
        # If too many time steps, take the most recent ones
        if len(price_series) > self.time_steps:
            price_series = price_series[-self.time_steps:]
        
        # Convert to numpy array
        price_series = np.array(price_series).reshape(1, self.time_steps, self.state_dim)
        
        # Extract trade-specific features
        trade_features = np.array([
            trade_info.get("entry_price", 0.0),
            trade_info.get("current_price", 0.0),
            trade_info.get("current_pnl", 0.0),
            trade_info.get("current_pnl_percent", 0.0),
            trade_info.get("max_pnl_seen", 0.0),
            trade_info.get("min_pnl_seen", 0.0),
            trade_info.get("time_in_trade", 0.0),
            trade_info.get("quantity", 1.0),
            trade_info.get("risk_amount", 0.0),
            trade_info.get("entry_score", 0.0)
        ]).reshape(1, 10)
        
        # Extract option-specific features
        option_features = np.array([
            trade_info.get("delta", 0.0),
            trade_info.get("gamma", 0.0),
            trade_info.get("theta", 0.0),
            trade_info.get("vega", 0.0),
            trade_info.get("rho", 0.0),
            trade_info.get("implied_volatility", 0.0),
            trade_info.get("days_to_expiration", 0.0),
            trade_info.get("dte_at_entry", 0.0)
        ]).reshape(1, 8)
        
        # One-hot encode strategy
        strategies = ["momentum", "mean_reversion", "volatility_breakout", "gamma_scalping", "theta_harvesting"]
        strategy_encoded = np.zeros((1, 5))
        if strategy in strategies:
            strategy_encoded[0, strategies.index(strategy)] = 1
        
        return price_series, trade_features, option_features, strategy_encoded
    
    def decide_exit(self, state_features, strategy, trade_info, evaluate=False):
        """
        Decide whether to exit a trade based on current conditions.
        
        Args:
            state_features (dict): Features describing the current market state
            strategy (str): Current trading strategy from meta controller
            trade_info (dict): Information about the current trade
            evaluate (bool): If True, use greedy policy (no exploration)
            
        Returns:
            int: Action (0: hold, 1: exit, 2: partial exit)
        """
        # Increment hold time counter
        self.hold_time += 1
        
        # Format state for model input
        price_series, trade_features, option_features, strategy_enc = self._format_state(
            state_features, strategy, trade_info
        )
        
        # Store for learning
        self.last_state = (price_series.copy(), trade_features.copy(), option_features.copy(), strategy_enc.copy())
        
        # Check for mandatory exit conditions first (these override RL decision)
        # 1. Approaching expiration with ITM option
        dte = trade_info.get("days_to_expiration", 0)
        is_itm = (trade_info.get("option_type") == "call" and 
                  trade_info.get("underlying_price", 0) > trade_info.get("strike", 0)) or \
                 (trade_info.get("option_type") == "put" and 
                  trade_info.get("underlying_price", 0) < trade_info.get("strike", 0))
                  
        if dte <= 1 and is_itm:
            self.logger.info("Mandatory exit: Option expiring tomorrow and is in-the-money")
            action = 1  # Exit
            self.last_action = action
            self.episode_actions.append(action)
            self.training_stats["action_distribution"][action] += 1
            return action
            
        # 2. Stop loss hit
        if trade_info.get("current_pnl_percent", 0) <= trade_info.get("stop_loss_threshold", -50):
            self.logger.info("Mandatory exit: Stop loss triggered")
            action = 1  # Exit
            self.last_action = action
            self.episode_actions.append(action)
            self.training_stats["action_distribution"][action] += 1
            return action
            
        # 3. Take profit hit
        if trade_info.get("current_pnl_percent", 0) >= trade_info.get("take_profit_threshold", 100):
            self.logger.info("Mandatory exit: Take profit triggered")
            action = 1  # Exit
            self.last_action = action
            self.episode_actions.append(action)
            self.training_stats["action_distribution"][action] += 1
            return action
        
        # Epsilon-greedy action selection for normal conditions
        if not evaluate and np.random.rand() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Exploitation: select best action
            q_values = self.main_model.predict([price_series, trade_features, option_features, strategy_enc])[0]
            action = np.argmax(q_values)
        
        # Store selected action
        self.last_action = action
        self.episode_actions.append(action)
        
        # Update action distribution stats
        self.training_stats["action_distribution"][action] += 1
        
        return action
    
    def record_experience(self, next_state_features, strategy, trade_info, reward, done):
        """
        Record experience for learning.
        
        Args:
            next_state_features (dict): Next state features
            strategy (str): Current trading strategy
            trade_info (dict): Information about the current trade
            reward (float): Reward received
            done (bool): Whether episode is done
        """
        if self.last_state is None or self.last_action is None:
            return
        
        # Format next state
        next_price_series, next_trade_features, next_option_features, next_strategy_enc = self._format_state(
            next_state_features, strategy, trade_info
        )
        
        # Apply reward shaping if available
        if self.reward_shaper:
            shaped_reward = self.reward_shaper.shape_exit_reward(
                self.last_state, self.last_action, 
                (next_price_series, next_trade_features, next_option_features, next_strategy_enc), 
                reward, strategy, self.hold_time
            )
        else:
            shaped_reward = reward
        
        # Store in replay memory
        experience = (
            self.last_state, 
            self.last_action, 
            shaped_reward, 
            (next_price_series, next_trade_features, next_option_features, next_strategy_enc), 
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
        price_batch = np.vstack([experience[0][0] for experience in batch])
        trade_batch = np.vstack([experience[0][1] for experience in batch])
        option_batch = np.vstack([experience[0][2] for experience in batch])
        strategy_batch = np.vstack([experience[0][3] for experience in batch])
        
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        
        next_price_batch = np.vstack([experience[3][0] for experience in batch])
        next_trade_batch = np.vstack([experience[3][1] for experience in batch])
        next_option_batch = np.vstack([experience[3][2] for experience in batch])
        next_strategy_batch = np.vstack([experience[3][3] for experience in batch])
        
        dones = np.array([experience[4] for experience in batch], dtype=bool)
        
        # Compute Q-targets
        q_targets = self.main_model.predict([price_batch, trade_batch, option_batch, strategy_batch])
        q_next = self.target_model.predict([next_price_batch, next_trade_batch, next_option_batch, next_strategy_batch])
        
        # Update Q-targets for selected actions
        for i in range(self.batch_size):
            if dones[i]:
                q_targets[i, actions[i]] = rewards[i]
            else:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        # Train the model
        history = self.main_model.fit(
            [price_batch, trade_batch, option_batch, strategy_batch], 
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
    
    def end_episode(self, total_reward, final_pnl=None):
        """
        Signal the end of an episode and update training stats.
        
        Args:
            total_reward (float): Total reward for the episode
            final_pnl (float, optional): Final P&L percentage for the trade
        """
        # Update episode counter
        self.training_stats["episodes"] += 1
        
        # Record reward
        self.training_stats["total_rewards"].append(total_reward)
        
        # Record hold time and final P&L
        self.training_stats["avg_hold_time"].append(self.hold_time)
        if final_pnl is not None:
            self.training_stats["pnl_at_exit"].append(final_pnl)
        
        # Calculate moving average of rewards
        window_size = min(100, len(self.training_stats["total_rewards"]))
        avg_reward = np.mean(self.training_stats["total_rewards"][-window_size:])
        self.training_stats["avg_rewards"].append(float(avg_reward))
        
        # Log progress
        episode = self.training_stats["episodes"]
        if episode % 10 == 0:
            avg_hold = np.mean(self.training_stats["avg_hold_time"][-window_size:])
            self.logger.info(
                f"Exit Controller - Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Avg Hold Time: {avg_hold:.1f}, Epsilon: {self.epsilon:.4f}"
            )
        
        # Reset episode variables
        self.last_state = None
        self.last_action = None
        self.episode_actions = []
        self.hold_time = 0
        
        # Save models periodically
        if episode % self.save_model_every == 0:
            self._save_models()
    
    def get_q_values(self, state_features, strategy, trade_info):
        """
        Get Q-values for all actions in the current state.
        
        Args:
            state_features (dict): Current state features
            strategy (str): Current trading strategy
            trade_info (dict): Information about the current trade
            
        Returns:
            list: Q-values for each action (hold, exit, partial exit)
        """
        # Format state for model input
        price_series, trade_features, option_features, strategy_enc = self._format_state(
            state_features, strategy, trade_info
        )
        
        # Get Q-values from model
        q_values = self.main_model.predict([price_series, trade_features, option_features, strategy_enc])[0]
        
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
            self.logger.info(f"Switched to {strategy} specific exit model")
            return True
        
        # Try to load from disk
        model_path = f"{self.model_dir}/exit_{strategy}_model.h5"
        
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific exit model")
                return True
            
            elif self.drive_manager and self.drive_manager.file_exists(f"hierarchical_rl/sub_controllers/exit_{strategy}_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary(f"hierarchical_rl/sub_controllers/exit_{strategy}_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific exit model from Google Drive")
                return True
            
            else:
                self.logger.info(f"No strategy-specific model found for {strategy}, using general model")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading strategy-specific model for {strategy}: {str(e)}")
            return False
