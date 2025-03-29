"""
RL Exit Timing Module

This module uses reinforcement learning to determine optimal exit timing for options trades.
It balances maximizing profits, minimizing losses, and managing time decay.
"""

import logging
import numpy as np
import random
from datetime import datetime, timedelta
import os
import json

class RLExitTiming:
    """
    Reinforcement learning-based system for determining optimal option trade exit timing.
    """
    
    def __init__(self, config, tradier_api):
        """
        Initialize the RLExitTiming component.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        
        # Extract RL model configuration
        self.rl_config = config.get("exit_strategies", {}).get("rl_exit_timing", {})
        
        # Initialize Q-table
        self.q_table = self._initialize_q_table()
        
        # Parameters
        self.learning_rate = self.rl_config.get("learning_rate", 0.1)
        self.discount_factor = self.rl_config.get("discount_factor", 0.95)
        self.exploration_rate = self.rl_config.get("exploration_rate", 0.2)
        
        # Load Q-table from file if exists
        self._load_q_table()
        
        self.logger.info("RLExitTiming initialized")
    
    def _initialize_q_table(self):
        """
        Initialize the Q-table for the RL model.
        
        Returns:
            dict: Q-table structure
        """
        # State space: Profit %, Days to Expiration, IV Percentile, Delta, Theta Decay Rate
        # Each state maps to action values (hold, exit)
        
        q_table = {}
        
        # Pre-populate with zeroes for common states to avoid cold-start
        for profit_bucket in range(-5, 6):  # -5 to +5, representing profit buckets
            for dte_bucket in range(6):     # 0-5, representing days to expiration buckets
                for iv_percentile in range(0, 101, 20):  # 0, 20, 40, 60, 80, 100
                    for delta_bucket in range(5):  # 0-4, representing delta buckets
                        state = (profit_bucket, dte_bucket, iv_percentile // 20, delta_bucket)
                        q_table[state] = {
                            'hold': 0.0,
                            'exit': 0.0
                        }
        
        return q_table
    
    def _load_q_table(self):
        """Load Q-table from file if it exists."""
        q_table_file = "data/q_tables/exit_timing_q_table.json"
        
        try:
            if os.path.exists(q_table_file):
                with open(q_table_file, 'r') as f:
                    # Convert string tuple keys back to actual tuples
                    str_q_table = json.load(f)
                    self.q_table = {}
                    
                    for str_key, value in str_q_table.items():
                        # Parse tuple from string representation
                        # Strip parentheses and split by comma
                        tuple_key = tuple(int(x.strip()) for x in str_key.strip('()').split(','))
                        self.q_table[tuple_key] = value
                        
                self.logger.info(f"Loaded Q-table with {len(self.q_table)} states")
            else:
                self.logger.info("No existing Q-table found, using initialized table")
        except Exception as e:
            self.logger.error(f"Error loading Q-table: {str(e)}")
    
    def _save_q_table(self):
        """Save Q-table to file."""
        q_table_file = "data/q_tables/exit_timing_q_table.json"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(q_table_file), exist_ok=True)
            
            # Convert tuple keys to strings
            str_q_table = {str(k): v for k, v in self.q_table.items()}
            
            with open(q_table_file, 'w') as f:
                json.dump(str_q_table, f, indent=2)
                
            self.logger.info(f"Saved Q-table with {len(self.q_table)} states")
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {str(e)}")
    
    def _get_state(self, trade, quote):
        """
        Convert trade data into a discrete state representation.
        
        Args:
            trade (dict): Trade information
            quote (dict): Current option quote
            
        Returns:
            tuple: State representation (profit_bucket, dte_bucket, iv_percentile_bucket, delta_bucket)
        """
        try:
            # Calculate profit percentage
            entry_price = trade['entry_price']
            current_price = (float(quote['bid']) + float(quote['ask'])) / 2
            profit_pct = ((current_price / entry_price) - 1) * 100
            
            # Map profit percentage to buckets
            if profit_pct <= -50:
                profit_bucket = -5
            elif profit_pct <= -30:
                profit_bucket = -4
            elif profit_pct <= -20:
                profit_bucket = -3
            elif profit_pct <= -10:
                profit_bucket = -2
            elif profit_pct < 0:
                profit_bucket = -1
            elif profit_pct == 0:
                profit_bucket = 0
            elif profit_pct < 10:
                profit_bucket = 1
            elif profit_pct < 25:
                profit_bucket = 2
            elif profit_pct < 50:
                profit_bucket = 3
            elif profit_pct < 100:
                profit_bucket = 4
            else:
                profit_bucket = 5
            
            # Calculate days to expiration
            expiration = datetime.strptime(trade['expiration'], '%Y-%m-%d').date()
            today = datetime.now().date()
            days_to_exp = (expiration - today).days
            
            # Map DTE to buckets
            if days_to_exp <= 1:
                dte_bucket = 0
            elif days_to_exp <= 3:
                dte_bucket = 1
            elif days_to_exp <= 7:
                dte_bucket = 2
            elif days_to_exp <= 14:
                dte_bucket = 3
            elif days_to_exp <= 30:
                dte_bucket = 4
            else:
                dte_bucket = 5
            
            # Get IV percentile if available
            iv_percentile_bucket = 2  # Default middle bucket
            if 'greeks' in quote and quote['greeks'] is not None:
                if 'mid_iv' in quote['greeks']:
                    iv = float(quote['greeks']['mid_iv'])
                    
                    # Compare to IV range
                    if 'initial_iv' in trade:
                        initial_iv = trade['initial_iv']
                        if iv <= initial_iv * 0.6:
                            iv_percentile_bucket = 0
                        elif iv <= initial_iv * 0.8:
                            iv_percentile_bucket = 1
                        elif iv <= initial_iv * 1.2:
                            iv_percentile_bucket = 2
                        elif iv <= initial_iv * 1.4:
                            iv_percentile_bucket = 3
                        else:
                            iv_percentile_bucket = 4
            
            # Get delta bucket if available
            delta_bucket = 2  # Default middle bucket
            if 'greeks' in quote and quote['greeks'] is not None:
                if 'delta' in quote['greeks']:
                    delta = abs(float(quote['greeks']['delta']))
                    
                    if delta < 0.2:
                        delta_bucket = 0
                    elif delta < 0.4:
                        delta_bucket = 1
                    elif delta < 0.6:
                        delta_bucket = 2
                    elif delta < 0.8:
                        delta_bucket = 3
                    else:
                        delta_bucket = 4
            
            # Return state as tuple
            return (profit_bucket, dte_bucket, iv_percentile_bucket, delta_bucket)
            
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            # Return default state on error
            return (0, 2, 2, 2)
    
    def get_exit_decision(self, trade, option_quote):
        """
        Decide whether to exit a trade based on RL model.
        
        Args:
            trade (dict): Trade information
            option_quote (dict): Current option quote
            
        Returns:
            tuple: (should_exit, confidence, reason)
        """
        # Get current state
        state = self._get_state(trade, option_quote)
        
        # Exploration: sometimes make a random decision to gather more data
        if random.random() < self.exploration_rate:
            action = random.choice(['hold', 'exit'])
            self.logger.debug(f"Exploration action for {trade['trade_id']}: {action}")
            
            # Low confidence when exploring
            confidence = 0.5
        else:
            # Get Q-values for this state
            if state in self.q_table:
                q_values = self.q_table[state]
            else:
                # Initialize if state not in table
                self.q_table[state] = {'hold': 0.0, 'exit': 0.0}
                q_values = self.q_table[state]
            
            # Choose action with highest Q-value
            if q_values['exit'] > q_values['hold']:
                action = 'exit'
            else:
                action = 'hold'
            
            # Calculate confidence based on Q-value difference
            diff = abs(q_values['exit'] - q_values['hold'])
            confidence = min(0.5 + (diff / 2.0), 0.99)  # Scale to 0.5-0.99
        
        # Formulate reason
        entry_price = trade['entry_price']
        current_price = (float(option_quote['bid']) + float(option_quote['ask'])) / 2
        profit_pct = ((current_price / entry_price) - 1) * 100
        
        expiration = datetime.strptime(trade['expiration'], '%Y-%m-%d').date()
        today = datetime.now().date()
        days_to_exp = (expiration - today).days
        
        if action == 'exit':
            reason = f"RL model recommends exit (profit: {profit_pct:.2f}%, DTE: {days_to_exp})"
        else:
            reason = None  # No reason needed for hold
        
        return (action == 'exit', confidence, reason)
    
    def update_model(self, trade, action_taken, reward):
        """
        Update the RL model based on action outcome.
        
        Args:
            trade (dict): Trade information
            action_taken (str): The action that was taken ('hold' or 'exit')
            reward (float): The reward received for the action
        """
        try:
            # Get the last state
            if 'last_rl_state' not in trade:
                self.logger.warning(f"No last state found for trade {trade['trade_id']}, skipping update")
                return
            
            state = trade['last_rl_state']
            
            # Make sure state is in Q-table
            if state not in self.q_table:
                self.q_table[state] = {'hold': 0.0, 'exit': 0.0}
            
            # Get current Q-value
            current_q = self.q_table[state][action_taken]
            
            # Update Q-value using reward
            # For terminal state (trade closed), no future reward
            new_q = current_q + self.learning_rate * (reward - current_q)
            
            # Update Q-table
            self.q_table[state][action_taken] = new_q
            
            self.logger.debug(
                f"Updated Q-value for state {state}, action {action_taken}: "
                f"{current_q:.4f} -> {new_q:.4f} (reward: {reward:.4f})"
            )
            
            # Periodically save Q-table
            if random.random() < 0.1:  # 10% chance to save after update
                self._save_q_table()
                
        except Exception as e:
            self.logger.error(f"Error updating RL model: {str(e)}")
    
    def calculate_reward(self, trade, exit_price=None):
        """
        Calculate reward for RL model based on trade outcome.
        
        Args:
            trade (dict): Trade information
            exit_price (float, optional): Exit price if different from trade exit_price
            
        Returns:
            float: Calculated reward
        """
        try:
            entry_price = trade['entry_price']
            price = exit_price if exit_price is not None else trade.get('exit_price')
            
            if price is None:
                self.logger.warning(f"No exit price for trade {trade['trade_id']}, using current price")
                # Try to get current price
                option_quote = self.tradier_api.get_option_quote(trade['option_symbol'])
                if option_quote:
                    price = (float(option_quote['bid']) + float(option_quote['ask'])) / 2
                else:
                    return 0.0  # Can't calculate reward without price
            
            # Calculate profit percentage
            profit_pct = ((price / entry_price) - 1) * 100
            
            # Calculate days held
            entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            exit_time = datetime.now() if exit_price is None else datetime.strptime(
                trade.get('exit_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                '%Y-%m-%d %H:%M:%S'
            )
            
            days_held = (exit_time - entry_time).days + (exit_time - entry_time).seconds / 86400
            
            # Calculate days to expiration at exit
            expiration = datetime.strptime(trade['expiration'], '%Y-%m-%d').date()
            days_to_exp = (expiration - exit_time.date()).days
            
            # Base reward on profit
            if profit_pct >= 0:
                # Positive reward for profits
                reward = min(1.0, profit_pct / 100.0)  # Scale to 0-1
            else:
                # Negative reward for losses, more severe for larger losses
                reward = max(-1.0, profit_pct / 50.0)  # Scale to -1-0
            
            # Adjust for time decay - penalize holding too close to expiration
            time_factor = 1.0
            if days_to_exp < 3:
                time_factor = 0.8  # Penalty for getting very close to expiration
            
            # Adjust for theta decay if available
            if 'greeks' in trade and trade['greeks'] and 'theta' in trade['greeks']:
                theta = abs(float(trade['greeks']['theta']))
                theta_factor = max(0.7, 1.0 - (theta / 0.1))  # Larger negative theta = bigger penalty
                time_factor *= theta_factor
            
            # Final reward
            final_reward = reward * time_factor
            
            self.logger.debug(
                f"Calculated reward for {trade['trade_id']}: {final_reward:.4f} "
                f"(profit: {profit_pct:.2f}%, time factor: {time_factor:.2f})"
            )
            
            return final_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
