"""
Meta Controller Module

This module implements the high-level strategic decision maker for
the hierarchical reinforcement learning system. It coordinates
sub-controllers and makes high-level trading strategy decisions.
"""

import logging
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

class MetaController:
    """
    High-level reinforcement learning controller that coordinates
    sub-controllers for entry, exit, and position sizing decisions.
    """
    
    def __init__(self, config, state_abstractor, reward_shaper=None, drive_manager=None):
        """
        Initialize the meta controller.
        
        Args:
            config (dict): Configuration dictionary
            state_abstractor: StateAbstractor instance for state representation
            reward_shaper: RewardShaper instance for reward shaping
            drive_manager: GoogleDriveManager for saving models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("meta_controller", {})
        self.state_abstractor = state_abstractor
        self.reward_shaper = reward_shaper
        self.drive_manager = drive_manager
        
        # Extract configuration parameters
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.gamma = self.config.get("gamma", 0.95)  # Discount factor
        self.epsilon = self.config.get("initial_epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.1)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.999)
        self.batch_size = self.config.get("batch_size", 32)
        self.update_target_every = self.config.get("update_target_every", 100)
        self.save_model_every = self.config.get("save_model_every", 1000)
        self.memory_size = self.config.get("memory_size", 10000)
        
        # Strategy selection parameters
        self.strategies = self.config.get("strategies", [
            "momentum", "mean_reversion", "volatility_breakout", 
            "gamma_scalping", "theta_harvesting"
        ])
        
        # Initialize state and observation space
        self.state_dim = self.config.get("state_dim", 20)  # Abstract state dimension
        self.action_dim = len(self.strategies)  # Number of strategies
        
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
            "epsilons": []
        }
        
        # Tracking current training state
        self.update_counter = 0
        self.current_strategy = None
        self.last_state = None
        self.last_action = None
        
        # Training directory
        self.model_dir = "data/hierarchical_rl"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing model if available
        self._load_models()
        
        self.logger.info(f"MetaController initialized with {len(self.strategies)} strategies")
    
    def _build_models(self):
        """Build the primary and target Q-networks."""
        try:
            # Main model for training
            self.main_model = self._create_model()
            
            # Target model for stable Q-targets
            self.target_model = self._create_model()
            self.target_model.set_weights(self.main_model.get_weights())
            
            self.logger.info("Built meta-controller models")
            
        except Exception as e:
            self.logger.error(f"Error building meta-controller models: {str(e)}")
            raise
    
    def _create_model(self):
        """Create a deep Q-network model."""
        # Input layer
        input_layer = Input(shape=(self.state_dim,))
        
        # Hidden layers
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        
        # Output layer (Q-values for each strategy)
        output_layer = Dense(self.action_dim, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _load_models(self):
        """Load saved models if they exist."""
        try:
            model_path = f"{self.model_dir}/meta_controller_model.h5"
            
            if os.path.exists(model_path):
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded saved meta-controller model")
                
                # Load training stats
                stats_path = f"{self.model_dir}/meta_controller_stats.json"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.training_stats = json.load(f)
            
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/meta_controller_model.h5"):
                # Download from Google Drive
                model_data = self.drive_manager.download_file_binary("hierarchical_rl/meta_controller_model.h5")
                
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info("Loaded meta-controller model from Google Drive")
                
                # Load training stats
                if self.drive_manager.file_exists("hierarchical_rl/meta_controller_stats.json"):
                    stats_data = self.drive_manager.download_file("hierarchical_rl/meta_controller_stats.json")
                    self.training_stats = json.loads(stats_data)
                    
                    with open(f"{self.model_dir}/meta_controller_stats.json", 'w') as f:
                        f.write(stats_data)
                        
        except Exception as e:
            self.logger.error(f"Error loading meta-controller model: {str(e)}")
            self.logger.info("Starting with new model")
    
    def _save_models(self):
        """Save the models and training stats."""
        try:
            model_path = f"{self.model_dir}/meta_controller_model.h5"
            stats_path = f"{self.model_dir}/meta_controller_stats.json"
            
            # Save model locally
            self.main_model.save(model_path)
            
            # Save training stats
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(model_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/meta_controller_model.h5",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
                with open(stats_path, 'r') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/meta_controller_stats.json",
                        f.read(),
                        mime_type="application/json"
                    )
            
            self.logger.info(f"Saved meta-controller model after {self.training_stats['episodes']} episodes")
            
        except Exception as e:
            self.logger.error(f"Error saving meta-controller model: {str(e)}")
    
    def select_strategy(self, state_features, evaluate=False):
        """
        Select the best trading strategy for the current market conditions.
        
        Args:
            state_features (dict): Dictionary of current market state features
            evaluate (bool): If True, use greedy policy (no exploration)
            
        Returns:
            str: Selected strategy name
        """
        # Get abstract state representation
        state = self.state_abstractor.get_abstract_state(state_features)
        
        # Store for learning
        self.last_state = state
        
        # Epsilon-greedy strategy selection
        if not evaluate and np.random.rand() < self.epsilon:
            # Exploration: random strategy
            action = np.random.randint(0, self.action_dim)
        else:
            # Exploitation: select best strategy
            state_tensor = np.array(state).reshape(1, -1)
            q_values = self.main_model.predict(state_tensor)[0]
            action = np.argmax(q_values)
        
        # Store selected action
        self.last_action = action
        self.current_strategy = self.strategies[action]
        
        self.logger.debug(f"Selected strategy: {self.current_strategy}")
        return self.current_strategy
    
    def record_experience(self, next_state_features, reward, done):
        """
        Record experience for learning.
        
        Args:
            next_state_features (dict): Next state features
            reward (float): Reward received
            done (bool): Whether episode is done
        """
        if self.last_state is None or self.last_action is None:
            return
            
        # Get abstract representation of next state
        next_state = self.state_abstractor.get_abstract_state(next_state_features)
        
        # Apply reward shaping if available
        if self.reward_shaper:
            shaped_reward = self.reward_shaper.shape_meta_reward(
                self.last_state, self.last_action, next_state, reward
            )
        else:
            shaped_reward = reward
        
        # Store in replay memory
        experience = (self.last_state, self.last_action, shaped_reward, next_state, done)
        
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
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch], dtype=bool)
        
        # Compute Q-targets
        q_targets = self.main_model.predict(states)
        q_next = self.target_model.predict(next_states)
        
        # Update Q-targets for selected actions
        for i in range(self.batch_size):
            if dones[i]:
                q_targets[i, actions[i]] = rewards[i]
            else:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        # Train the model
        history = self.main_model.fit(states, q_targets, epochs=1, verbose=0)
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
                f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Epsilon: {self.epsilon:.4f}"
            )
        
        # Reset episode variables
        self.last_state = None
        self.last_action = None
        self.current_strategy = None
        
        # Save models periodically
        if episode % self.save_model_every == 0:
            self._save_models()
    
    def get_q_values(self, state_features):
        """
        Get Q-values for all strategies in the current state.
        
        Args:
            state_features (dict): Current state features
            
        Returns:
            dict: Mapping of strategy names to Q-values
        """
        # Get abstract state
        state = self.state_abstractor.get_abstract_state(state_features)
        state_tensor = np.array(state).reshape(1, -1)
        
        # Get Q-values from model
        q_values = self.main_model.predict(state_tensor)[0]
        
        # Map to strategies
        strategy_q_values = {
            strategy: float(q_values[i])
            for i, strategy in enumerate(self.strategies)
        }
        
        return strategy_q_values
    
    def get_training_stats(self):
        """
        Get training statistics.
        
        Returns:
            dict: Training statistics
        """
        # Create a copy to avoid modifying the original
        stats = self.training_stats.copy()
        
        # Truncate long lists for readability
        for key in ["total_rewards", "avg_rewards", "losses", "epsilons"]:
            if key in stats and len(stats[key]) > 100:
                stats[key] = stats[key][-100:]  # Keep only the last 100 entries
        
        return stats
    
    def load_pretrained_model(self, model_path):
        """
        Load a pretrained model from a specific path.
        
        Args:
            model_path (str): Path to the pretrained model
            
        Returns:
            bool: Success flag
        """
        try:
            if os.path.exists(model_path):
                self.main_model = load_model(model_path)
                self.target_model.set_weights(self.main_model.get_weights())
                self.logger.info(f"Loaded pretrained meta-controller model from {model_path}")
                return True
            else:
                self.logger.warning(f"Pretrained model not found at {model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading pretrained model: {str(e)}")
            return False
