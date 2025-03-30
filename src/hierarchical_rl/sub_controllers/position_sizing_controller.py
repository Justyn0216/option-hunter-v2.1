"""
Position Sizing Controller Module

This module specializes in determining optimal position sizes as part of
the hierarchical reinforcement learning system, based on risk metrics,
market conditions, and account parameters.
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
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

class PositionSizingController:
    """
    Reinforcement learning controller specialized for position sizing decisions.
    Optimizes trade allocation based on risk management principles, account
    size, market conditions, and option characteristics.
    """
    
    def __init__(self, config, reward_shaper=None, drive_manager=None):
        """
        Initialize the position sizing controller.
        
        Args:
            config (dict): Configuration dictionary
            reward_shaper: RewardShaper instance for reward shaping
            drive_manager: GoogleDriveManager for saving models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("sub_controllers", {}).get("position_sizing_controller", {})
        self.reward_shaper = reward_shaper
        self.drive_manager = drive_manager
        
        # Extract configuration parameters
        self.learning_rate = self.config.get("learning_rate", 0.0005)
        self.gamma = self.config.get("gamma", 0.95)
        self.epsilon = self.config.get("initial_epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 64)
        self.update_target_every = self.config.get("update_target_every", 100)
        self.save_model_every = self.config.get("save_model_every", 500)
        self.memory_size = self.config.get("memory_size", 10000)
        
        # Position sizing ranges
        self.max_account_risk_pct = self.config.get("max_account_risk_pct", 2.0)
        self.min_account_risk_pct = self.config.get("min_account_risk_pct", 0.1)
        self.risk_increment = self.config.get("risk_increment", 0.1)
        
        # Define sizing levels (in % of account)
        self.position_sizes = np.arange(
            self.min_account_risk_pct, 
            self.max_account_risk_pct + self.risk_increment, 
            self.risk_increment
        ).tolist()
        
        # Define state and action spaces
        self.state_dim = self.config.get("state_dim", 20)
        self.action_dim = len(self.position_sizes)  # Number of position size options
        
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
            "avg_position_size": [],
            "risk_adjusted_returns": []
        }
        
        # Tracking current training state
        self.update_counter = 0
        self.last_state = None
        self.last_action = None
        self.episode_sizes = []
        
        # Directory for models
        self.model_dir = "data/hierarchical_rl/sub_controllers"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing model if available
        self._load_models()
        
        # Strategy-specific models
        self.strategy_models = {}
        
        self.logger.info(f"PositionSizingController initialized with {len(self.position_sizes)} position size levels")
    
    def _build_models(self):
        """Build the primary and target networks."""
        try:
            # Main model for training
            self.main_model = self._create_model()
            
            # Target model for stable Q-targets
            self.target_model = self._create_model()
            self.target_model.set_weights(self.main_model.get_weights())
            
            self.logger.info("Built position sizing controller models")
            
        except Exception as e:
            self.logger.error(f"Error building position sizing controller models: {str(e)}")
            raise
    
    def _create_model(self):
        """Create a neural network model for position sizing."""
        # Market state input
        market_input = Input(shape=(10,))  # Market regime, volatility, etc.
        
        # Account state input
        account_input = Input(shape=(5,))  # Account balance, open positions, equity curve, etc.
        
        # Option/trade specific input
        trade_input = Input(shape=(12,))  # Option characteristics, expected trade parameters
        
        # Strategy input
        strategy_input = Input(shape=(5,))  # One-hot encoded strategy
        
        # Process each input stream separately initially
        x1 = Dense(32, activation='relu')(market_input)
        x1 = BatchNormalization()(x1)
        
        x2 = Dense(16, activation='relu')(account_input)
        x2 = BatchNormalization()(x2)
        
        x3 = Dense(32, activation='relu')(trade_input)
        x3 = BatchNormalization()(x3)
        
        x4 = Dense(16, activation='relu')(strategy_input)
        x4 = BatchNormalization()(x4)
        
        # Combine all streams
        combined = Concatenate()([x1, x2, x3, x4])
        
        # Dense layers
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer (Q-values for each position size option)
        output_layer = Dense(self.action_dim, activation='linear')(x)
        
        # Create model with multiple inputs
        model = Model(
            inputs=[market_input, account_input, trade_input, strategy_input], 
            outputs=output_layer
        )
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _load_models(self):
        """Load saved models if they exist."""
        try:
            model_path = f"{self.model_dir}/position_sizing_controller_model.h5"
            
            if os.path.exists(model_path):
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded saved position sizing controller model")
                
                # Load training stats
                stats_path = f"{self.model_dir}/position_sizing_controller_stats.json"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/position_sizing_controller_stats.json",
                        f.read(),
                        mime_type="application/json"
                    )
            
            self.logger.info(f"Saved position sizing controller model after {self.training_stats['episodes']} episodes")
            
        except Exception as e:
            self.logger.error(f"Error saving position sizing controller model: {str(e)}")
    
    def _format_state(self, state_features, strategy, account_info):
        """
        Format state features into the format required by the model.
        
        Args:
            state_features (dict): Dictionary of state features
            strategy (str): Current trading strategy
            account_info (dict): Account balance and position information
            
        Returns:
            tuple: (market_features, account_features, trade_features, strategy_encoded)
        """
        # Extract market-related features
        market_features = np.array([
            state_features.get("market_regime_value", 0.0),
            state_features.get("volatility_regime_value", 0.0),
            state_features.get("vix_level", 0.0),
            state_features.get("market_trend", 0.0),
            state_features.get("sector_performance", 0.0),
            state_features.get("overall_sentiment", 0.0),
            state_features.get("market_breadth", 0.0),
            state_features.get("put_call_ratio", 0.0),
            state_features.get("atr_percentage", 0.0),
            state_features.get("recent_market_volatility", 0.0)
        ]).reshape(1, 10)
        
        # Extract account-related features
        account_features = np.array([
            account_info.get("account_balance", 0.0),
            account_info.get("used_margin", 0.0),
            account_info.get("open_positions_count", 0.0),
            account_info.get("daily_pnl_percentage", 0.0),
            account_info.get("equity_drawdown", 0.0)
        ]).reshape(1, 5)
        
        # Extract trade-specific features
        trade_features = np.array([
            state_features.get("option_price", 0.0),
            state_features.get("strike_distance_percentage", 0.0),
            state_features.get("days_to_expiration", 0.0),
            state_features.get("implied_volatility", 0.0),
            state_features.get("delta", 0.0),
            state_features.get("gamma", 0.0),
            state_features.get("theta", 0.0),
            state_features.get("vega", 0.0),
            state_features.get("expected_move", 0.0),
            state_features.get("underlying_volatility", 0.0),
            state_features.get("spread_percentage", 0.0),
            state_features.get("liquidity_score", 0.0)
        ]).reshape(1, 12)
        
        # One-hot encode strategy
        strategies = ["momentum", "mean_reversion", "volatility_breakout", "gamma_scalping", "theta_harvesting"]
        strategy_encoded = np.zeros((1, 5))
        if strategy in strategies:
            strategy_encoded[0, strategies.index(strategy)] = 1
        
        return market_features, account_features, trade_features, strategy_encoded
    
    def decide_position_size(self, state_features, strategy, account_info, evaluate=False):
        """
        Decide the position size for a trade based on current conditions.
        
        Args:
            state_features (dict): Features describing the current market state
            strategy (str): Current trading strategy from meta controller
            account_info (dict): Account balance and position information
            evaluate (bool): If True, use greedy policy (no exploration)
            
        Returns:
            float: Position size as percentage of account
        """
        # Format state for model input
        market_features, account_features, trade_features, strategy_enc = self._format_state(
            state_features, strategy, account_info
        )
        
        # Store for learning
        self.last_state = (market_features.copy(), account_features.copy(), trade_features.copy(), strategy_enc.copy())
        
        # Apply position sizing constraints from state if available
        max_allowed_risk = min(
            state_features.get("max_position_risk", self.max_account_risk_pct),
            self.max_account_risk_pct
        )
        
        # Filter available actions based on max allowed risk
        valid_actions = [i for i, size in enumerate(self.position_sizes) if size <= max_allowed_risk]
        
        # Epsilon-greedy action selection
        if not evaluate and np.random.rand() < self.epsilon:
            # Exploration: random action (only from valid actions)
            action = np.random.choice(valid_actions)
        else:
            # Exploitation: select best valid action
            q_values = self.main_model.predict([market_features, account_features, trade_features, strategy_enc])[0]
            
            # Only consider valid actions
            valid_q_values = [(i, q_values[i]) for i in valid_actions]
            action = max(valid_q_values, key=lambda x: x[1])[0]
        
        # Store selected action
        self.last_action = action
        self.episode_sizes.append(self.position_sizes[action])
        
        # Update stats
        self.training_stats["avg_position_size"].append(self.position_sizes[action])
        
        return self.position_sizes[action]
    
    def record_experience(self, next_state_features, strategy, account_info, reward, done):
        """
        Record experience for learning.
        
        Args:
            next_state_features (dict): Next state features
            strategy (str): Current trading strategy
            account_info (dict): Account balance and position information
            reward (float): Reward received
            done (bool): Whether episode is done
        """
        if self.last_state is None or self.last_action is None:
            return
        
        # Format next state
        next_market_features, next_account_features, next_trade_features, next_strategy_enc = self._format_state(
            next_state_features, strategy, account_info
        )
        
        # Apply reward shaping if available
        if self.reward_shaper:
            # Adjust reward based on risk - higher reward for smaller positions that succeed
            current_size = self.position_sizes[self.last_action]
            
            shaped_reward = self.reward_shaper.shape_position_sizing_reward(
                self.last_state, self.last_action, 
                (next_market_features, next_account_features, next_trade_features, next_strategy_enc), 
                reward, strategy, current_size
            )
        else:
            # Basic reward shaping - adjust for risk-adjusted return
            current_size = self.position_sizes[self.last_action]
            base_reward = reward
            
            # Sharpe-like adjustment: reward / risk
            if current_size > 0:
                shaped_reward = base_reward / np.sqrt(current_size)
                
                # Record risk-adjusted return
                if base_reward != 0:
                    self.training_stats["risk_adjusted_returns"].append(float(shaped_reward))
            else:
                shaped_reward = base_reward
        
        # Store in replay memory
        experience = (
            self.last_state, 
            self.last_action, 
            shaped_reward, 
            (next_market_features, next_account_features, next_trade_features, next_strategy_enc), 
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
        market_batch = np.vstack([experience[0][0] for experience in batch])
        account_batch = np.vstack([experience[0][1] for experience in batch])
        trade_batch = np.vstack([experience[0][2] for experience in batch])
        strategy_batch = np.vstack([experience[0][3] for experience in batch])
        
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        
        next_market_batch = np.vstack([experience[3][0] for experience in batch])
        next_account_batch = np.vstack([experience[3][1] for experience in batch])
        next_trade_batch = np.vstack([experience[3][2] for experience in batch])
        next_strategy_batch = np.vstack([experience[3][3] for experience in batch])
        
        dones = np.array([experience[4] for experience in batch], dtype=bool)
        
        # Compute Q-targets
        q_targets = self.main_model.predict([market_batch, account_batch, trade_batch, strategy_batch])
        q_next = self.target_model.predict([next_market_batch, next_account_batch, next_trade_batch, next_strategy_batch])
        
        # Update Q-targets for selected actions
        for i in range(self.batch_size):
            if dones[i]:
                q_targets[i, actions[i]] = rewards[i]
            else:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        # Train the model
        history = self.main_model.fit(
            [market_batch, account_batch, trade_batch, strategy_batch], 
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
            avg_size = np.mean(self.training_stats["avg_position_size"][-window_size:])
            self.logger.info(
                f"Position Sizing - Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Avg Size: {avg_size:.2f}%, Epsilon: {self.epsilon:.4f}"
            )
        
        # Reset episode variables
        self.last_state = None
        self.last_action = None
        self.episode_sizes = []
        
        # Save models periodically
        if episode % self.save_model_every == 0:
            self._save_models()
    
    def get_sizing_options(self):
        """
        Get the available position sizing options.
        
        Returns:
            list: Available position sizes as percentages
        """
        return self.position_sizes.copy()
    
    def calculate_dollar_amount(self, size_percentage, account_balance, option_price):
        """
        Calculate the dollar amount and number of contracts for a trade.
        
        Args:
            size_percentage (float): Position size as percentage of account
            account_balance (float): Current account balance
            option_price (float): Price per option contract
            
        Returns:
            tuple: (dollar_amount, num_contracts)
        """
        # Calculate dollar amount based on risk percentage
        dollar_amount = account_balance * (size_percentage / 100.0)
        
        # Calculate number of contracts (1 contract = 100 shares)
        contracts_possible = max(1, int(dollar_amount / (option_price * 100)))
        
        # Recalculate actual dollar amount based on whole contracts
        actual_dollar_amount = contracts_possible * option_price * 100
        
        return actual_dollar_amount, contracts_possible
    
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
            self.logger.info(f"Switched to {strategy} specific position sizing model")
            return True
        
        # Try to load from disk
        model_path = f"{self.model_dir}/position_sizing_{strategy}_model.h5"
        
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific position sizing model")
                return True
            
            elif self.drive_manager and self.drive_manager.file_exists(f"hierarchical_rl/sub_controllers/position_sizing_{strategy}_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary(f"hierarchical_rl/sub_controllers/position_sizing_{strategy}_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                model = load_model(model_path)
                self.strategy_models[strategy] = model
                
                # Set as current main model
                self.main_model = model
                self.target_model.set_weights(self.main_model.get_weights())
                
                self.logger.info(f"Loaded {strategy} specific position sizing model from Google Drive")
                return True
            
            else:
                self.logger.info(f"No strategy-specific model found for {strategy}, using general model")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading strategy-specific model for {strategy}: {str(e)}")
            return False
                        self.training_stats = json.load(f)
            
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/sub_controllers/position_sizing_controller_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary("hierarchical_rl/sub_controllers/position_sizing_controller_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded position sizing controller model from Google Drive")
                
                # Load training stats
                if self.drive_manager.file_exists("hierarchical_rl/sub_controllers/position_sizing_controller_stats.json"):
                    stats_data = self.drive_manager.download_file("hierarchical_rl/sub_controllers/position_sizing_controller_stats.json")
                    self.training_stats = json.loads(stats_data)
                    
                    with open(f"{self.model_dir}/position_sizing_controller_stats.json", 'w') as f:
                        f.write(stats_data)
                        
        except Exception as e:
            self.logger.error(f"Error loading position sizing controller model: {str(e)}")
            self.logger.info("Starting with new position sizing controller model")
    
    def _save_models(self):
        """Save the models and training stats."""
        try:
            model_path = f"{self.model_dir}/position_sizing_controller_model.h5"
            stats_path = f"{self.model_dir}/position_sizing_controller_stats.json"
            
            # Save model locally
            self.main_model.save(model_path)
            
            # Save training stats
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(model_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/sub_controllers/position_sizing_controller_model.h5",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
                with open(stats_path, 'r') as f:
