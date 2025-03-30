"""
Reward Shaper Module

This module provides reward shaping functionality for the hierarchical
reinforcement learning system, helping to guide learning by providing
more informative rewards to the RL agents.
"""

import logging
import numpy as np
from datetime import datetime
import pandas as pd
import json
import os

class RewardShaper:
    """
    Shapes rewards for the hierarchical reinforcement learning system
    to improve learning efficiency and guide agents toward better behavior.
    """
    
    def __init__(self, config):
        """
        Initialize the RewardShaper.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("reward_shaper", {})
        
        # Extract configuration parameters
        self.meta_reward_scale = self.config.get("meta_reward_scale", 1.0)
        self.entry_reward_scale = self.config.get("entry_reward_scale", 1.0)
        self.exit_reward_scale = self.config.get("exit_reward_scale", 1.0)
        self.position_sizing_reward_scale = self.config.get("position_sizing_reward_scale", 1.0)
        
        # Parameters for reward shaping
        self.early_exit_penalty = self.config.get("early_exit_penalty", -0.2)
        self.late_exit_penalty = self.config.get("late_exit_penalty", -0.3)
        self.patience_bonus = self.config.get("patience_bonus", 0.1)
        self.diversification_bonus = self.config.get("diversification_bonus", 0.15)
        self.consistency_bonus = self.config.get("consistency_bonus", 0.2)
        
        # Temporal difference parameters
        self.use_temporal_difference = self.config.get("use_temporal_difference", True)
        self.td_lambda = self.config.get("td_lambda", 0.9)
        
        # Tracking recent rewards by strategy
        self.strategy_rewards = {}
        
        # Directory for storing reward statistics
        self.stats_dir = "data/hierarchical_rl/reward_stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        
        self.logger.info("RewardShaper initialized")
    
    def shape_meta_reward(self, state, action, next_state, original_reward):
        """
        Shape the reward for the meta controller.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            original_reward: Original reward received
            
        Returns:
            float: Shaped reward
        """
        # Scale the original reward
        shaped_reward = original_reward * self.meta_reward_scale
        
        # Track strategy performance
        strategy = self.get_strategy_name(action)
        if strategy not in self.strategy_rewards:
            self.strategy_rewards[strategy] = []
        
        # Add strategy consistency bonus
        if len(self.strategy_rewards[strategy]) >= 3:
            recent_rewards = self.strategy_rewards[strategy][-3:]
            if all(r > 0 for r in recent_rewards):
                # Bonus for consistently good strategy
                shaped_reward += self.consistency_bonus
        
        # Record this reward
        self.strategy_rewards[strategy].append(original_reward)
        
        return shaped_reward
    
    def shape_entry_reward(self, state, action, next_state, original_reward, strategy):
        """
        Shape the reward for the entry controller.
        
        Args:
            state: Current state (time_series, strategy_enc, price_features)
            action: Action taken (0: no action, 1: enter call, 2: enter put)
            next_state: Next state
            original_reward: Original reward received
            strategy: Current trading strategy
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = original_reward * self.entry_reward_scale
        
        # No entry action (action = 0)
        if action == 0:
            # If market moved favorably after not entering, slight penalty for missed opportunity
            if original_reward > 0:
                shaped_reward *= 0.5  # Reduce reward for passive behavior that worked out
            else:
                shaped_reward *= 0.8  # Reduce penalty for correctly avoiding a bad trade
        
        # Entry action (action = 1 or 2)
        else:
            # Penalty for entering right before major loss
            if original_reward < -0.5:
                shaped_reward *= 1.2  # Increase penalty for bad timing
            
            # Bonus for entering before major gain
            if original_reward > 0.5:
                shaped_reward *= 1.1  # Increase reward for good timing
        
        # Check for entry signal quality (using time series data)
        if action > 0:  # If entered a position
            signal_quality_bonus = self._evaluate_entry_signal_quality(state, action, strategy)
            shaped_reward += signal_quality_bonus
        
        return shaped_reward
    
    def _evaluate_entry_signal_quality(self, state, action, strategy):
        """
        Evaluate quality of entry signal based on technical indicators in the state.
        
        Args:
            state: Current state
            action: Action taken
            strategy: Current trading strategy
            
        Returns:
            float: Signal quality bonus
        """
        # Extract time series data
        time_series = state[0][0]  # First element of tuple, first batch
        
        # Calculate bonus based on strategy and action
        bonus = 0.0
        
        # Extract last few values of relevant indicators from state
        # Assuming state contains indicators like RSI, MACD, etc.
        try:
            # Example for momentum strategy
            if strategy == "momentum":
                # Last value of momentum indicator (e.g., column 5 in time series)
                momentum = time_series[-1, 5]
                
                if action == 1:  # Call option
                    # Bonus for strong positive momentum
                    if momentum > 0.7:
                        bonus += 0.1
                elif action == 2:  # Put option
                    # Bonus for strong negative momentum
                    if momentum < -0.7:
                        bonus += 0.1
            
            # Example for mean reversion strategy
            elif strategy == "mean_reversion":
                # Assuming overbought/oversold indicator (e.g., RSI) is in column 6
                overbought_oversold = time_series[-1, 6]
                
                if action == 1:  # Call option
                    # Bonus for oversold conditions
                    if overbought_oversold < 0.3:
                        bonus += 0.1
                elif action == 2:  # Put option
                    # Bonus for overbought conditions
                    if overbought_oversold > 0.7:
                        bonus += 0.1
            
            # Add more strategy-specific evaluations here
            
        except (IndexError, TypeError) as e:
            # Handle cases where expected indicators aren't in state
            pass
        
        return bonus
    
    def shape_exit_reward(self, state, action, next_state, original_reward, strategy, hold_time):
        """
        Shape the reward for the exit controller.
        
        Args:
            state: Current state (price_series, trade_features, option_features, strategy_enc)
            action: Action taken (0: hold, 1: exit, 2: partial exit)
            next_state: Next state
            original_reward: Original reward received
            strategy: Current trading strategy
            hold_time: How long the position has been held
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = original_reward * self.exit_reward_scale
        
        # Hold action (action = 0)
        if action == 0:
            # If held while position loses value
            if original_reward < 0:
                # Increase penalty for holding a losing position
                shaped_reward -= abs(original_reward) * 0.1
            else:
                # Small bonus for patience that paid off
                shaped_reward += self.patience_bonus * min(hold_time / 20, 1.0)
        
        # Exit action (action = 1)
        elif action == 1:
            # Extract trade features to see if we're exiting near the peak
            trade_features = state[1][0]  # First element of tuple, first batch
            current_pnl = trade_features[2]  # Current P&L (assumed to be index 2)
            max_pnl_seen = trade_features[4]  # Max P&L seen (assumed to be index 4)
            
            # Penalty for exiting far from peak if P&L is positive
            if current_pnl > 0 and max_pnl_seen > current_pnl:
                exit_quality = current_pnl / max_pnl_seen  # 1.0 = perfect exit
                shaped_reward *= (0.5 + 0.5 * exit_quality)  # Scale between 0.5x and 1.0x
            
            # Check for exit timing based on option features
            option_features = state[2][0]  # First element of tuple, first batch
            days_to_expiration = option_features[6]  # DTE (assumed to be index 6)
            
            # If exiting near expiration with little time value left
            if days_to_expiration < 5:
                theta = option_features[2]  # Theta (assumed to be index 2)
                if abs(theta) > 0.1:  # High theta decay
                    shaped_reward += 0.1  # Bonus for good exit timing
        
        # Partial exit action (action = 2)
        elif action == 2:
            # Bonus for locking in some profits in volatile conditions
            if original_reward > 0:
                shaped_reward += 0.05
        
        return shaped_reward
    
    def shape_position_sizing_reward(self, state, action, next_state, original_reward, strategy, position_size):
        """
        Shape the reward for the position sizing controller.
        
        Args:
            state: Current state (market_features, account_features, trade_features, strategy_enc)
            action: Action taken (index of position size)
            next_state: Next state
            original_reward: Original reward received
            strategy: Current trading strategy
            position_size: Selected position size as percentage
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = original_reward * self.position_sizing_reward_scale
        
        # Extract market features to check volatility
        market_features = state[0][0]  # First element of tuple, first batch
        volatility = market_features[1]  # Volatility regime (assumed to be index 1)
        vix_level = market_features[2]  # VIX level (assumed to be index 2)
        
        # Extract account features
        account_features = state[1][0]  # First element of tuple, first batch
        open_positions = account_features[2]  # Number of open positions (assumed to be index 2)
        
        # Risk-adjusted reward - divide by square root of position size
        if position_size > 0:
            risk_adjusted_reward = original_reward / np.sqrt(position_size / 100.0)
            shaped_reward = risk_adjusted_reward
        
        # Add bonus/penalty based on market conditions
        
        # In high volatility, smaller positions should get a bonus
        if volatility > 0.7 or vix_level > 25:
            if position_size < 1.0:
                shaped_reward += 0.2  # Bonus for appropriate caution
            elif position_size > 1.5:
                shaped_reward -= 0.2  # Penalty for excessive risk
        
        # In low volatility, reward medium-sized positions
        elif volatility < 0.3 and vix_level < 15:
            if 0.8 <= position_size <= 1.5:
                shaped_reward += 0.1  # Bonus for appropriate sizing
        
        # Diversification bonus - smaller positions when already have many open trades
        if open_positions > 3 and position_size < 1.0:
            shaped_reward += self.diversification_bonus
        
        # Strategy-specific adjustments
        if strategy == "theta_harvesting" and position_size > 1.5:
            shaped_reward -= 0.15  # Penalty for large sizes in theta strategies (should be many small positions)
            
        if strategy == "gamma_scalping" and position_size < 0.5:
            shaped_reward -= 0.1  # Penalty for too small positions in gamma strategies (need enough gamma exposure)
        
        return shaped_reward
    
    def get_strategy_name(self, action_index):
        """
        Get strategy name from action index.
        
        Args:
            action_index (int): Index of the strategy action
            
        Returns:
            str: Strategy name
        """
        strategies = ["momentum", "mean_reversion", "volatility_breakout", "gamma_scalping", "theta_harvesting"]
        if 0 <= action_index < len(strategies):
            return strategies[action_index]
        return "unknown"
    
    def save_reward_statistics(self):
        """
        Save reward statistics to disk.
        """
        try:
            stats = {
                "strategy_rewards": self.strategy_rewards,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(f"{self.stats_dir}/reward_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
                
            self.logger.info("Saved reward statistics")
            
        except Exception as e:
            self.logger.error(f"Error saving reward statistics: {str(e)}")
    
    def load_reward_statistics(self):
        """
        Load reward statistics from disk.
        
        Returns:
            bool: Success flag
        """
        try:
            stats_file = f"{self.stats_dir}/reward_stats.json"
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                
                self.strategy_rewards = stats.get("strategy_rewards", {})
                self.logger.info("Loaded reward statistics")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading reward statistics: {str(e)}")
            return False
    
    def get_reward_statistics(self, strategy=None):
        """
        Get reward statistics for a specific strategy or all strategies.
        
        Args:
            strategy (str, optional): Strategy name
            
        Returns:
            dict: Reward statistics
        """
        if strategy:
            return {
                "rewards": self.strategy_rewards.get(strategy, []),
                "mean": np.mean(self.strategy_rewards.get(strategy, [0])),
                "std": np.std(self.strategy_rewards.get(strategy, [0])),
                "max": np.max(self.strategy_rewards.get(strategy, [0])),
                "min": np.min(self.strategy_rewards.get(strategy, [0])),
                "count": len(self.strategy_rewards.get(strategy, []))
            }
        else:
            stats = {}
            for strategy, rewards in self.strategy_rewards.items():
                if rewards:
                    stats[strategy] = {
                        "mean": float(np.mean(rewards)),
                        "std": float(np.std(rewards)),
                        "max": float(np.max(rewards)),
                        "min": float(np.min(rewards)),
                        "count": len(rewards)
                    }
            return stats
    
    def clear_strategy_rewards(self):
        """
        Clear stored strategy rewards.
        """
        self.strategy_rewards = {}
        self.logger.info("Cleared strategy rewards history")
    
    def create_custom_reward_function(self, strategy):
        """
        Create a custom reward function for a specific strategy.
        
        Args:
            strategy (str): Strategy name
            
        Returns:
            function: Custom reward function
        """
        # Define strategy-specific reward functions
        if strategy == "momentum":
            def momentum_reward(original_reward, state, action, next_state):
                # Enhance rewards for momentum trends that continue
                shaped_reward = original_reward * 1.1
                
                # Add time-series analysis if available
                if hasattr(state, "__getitem__") and len(state) > 0:
                    try:
                        # Assuming time series data is available
                        time_series = state[0]
                        
                        # Check for trend continuation
                        if len(time_series) > 2:
                            # Simple momentum check (last point vs point before)
                            momentum_continuing = (time_series[-1] - time_series[-2]) > 0
                            
                            if momentum_continuing and original_reward > 0:
                                shaped_reward *= 1.15  # Bonus for trades aligned with momentum
                    except (IndexError, TypeError, AttributeError):
                        pass
                
                return shaped_reward
            
            return momentum_reward
            
        elif strategy == "mean_reversion":
            def mean_reversion_reward(original_reward, state, action, next_state):
                # Enhance rewards for mean reversion that works
                shaped_reward = original_reward * 1.05
                
                # Check for overbought/oversold conditions
                if hasattr(state, "__getitem__") and len(state) > 1:
                    try:
                        # Assuming overbought/oversold indicator is available
                        indicators = state[1]
                        
                        # Check if we took the right action for mean reversion
                        if "overbought" in indicators and action == 2:  # Put on overbought
                            shaped_reward *= 1.1
                        elif "oversold" in indicators and action == 1:  # Call on oversold
                            shaped_reward *= 1.1
                    except (IndexError, TypeError, AttributeError):
                        pass
                
                return shaped_reward
            
            return mean_reversion_reward
            
        # Default reward function
        return lambda original_reward, state, action, next_state: original_reward
    
    def apply_custom_reward_function(self, strategy, original_reward, state, action, next_state):
        """
        Apply a custom reward function for a specific strategy.
        
        Args:
            strategy (str): Strategy name
            original_reward (float): Original reward
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Shaped reward
        """
        reward_func = self.create_custom_reward_function(strategy)
        return reward_func(original_reward, state, action, next_state)
