"""
Reinforcement Feedback Module

This module provides feedback to reinforcement learning components
based on trade outcomes and prediction accuracy.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class ReinforcementFeedback:
    """
    Provides structured feedback to RL components based on trade outcomes,
    creating reward signals that align with trading objectives.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the ReinforcementFeedback.
        
        Args:
            config (dict): Configuration settings
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.feedback_config = config.get("meta_learning", {}).get("reinforcement_feedback", {})
        
        # Reward scaling factors
        self.reward_factors = self.feedback_config.get("reward_factors", {
            "pnl_percent": 0.05,        # Scale factor for PnL percentage
            "profit_threshold": 10.0,   # Profit threshold for full reward
            "loss_threshold": -20.0,    # Loss threshold for full penalty
            "accuracy_weight": 0.3,     # Weight for prediction accuracy
            "time_weight": 0.2,         # Weight for timing efficiency
            "risk_weight": 0.2          # Weight for risk management
        })
        
        # Feedback storage
        self.feedback_history = []
        
        # Create directory
        os.makedirs("data/meta_learning", exist_ok=True)
        
        # Load feedback history
        self._load_feedback_history()
        
        self.logger.info("ReinforcementFeedback initialized")
    
    def _load_feedback_history(self):
        """Load feedback history from storage."""
        history_file = "data/meta_learning/reinforcement_feedback.json"
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.feedback_history = json.load(f)
                
                self.logger.info(f"Loaded {len(self.feedback_history)} feedback records")
                
            elif self.drive_manager and self.drive_manager.file_exists("reinforcement_feedback.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("reinforcement_feedback.json")
                
                # Parse JSON
                self.feedback_history = json.loads(file_data)
                
                # Save locally
                with open(history_file, 'w') as f:
                    json.dump(self.feedback_history, f, indent=2)
                
                self.logger.info(f"Loaded {len(self.feedback_history)} feedback records from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading feedback history: {str(e)}")
            # Start with empty history
            self.feedback_history = []
    
    def _save_feedback_history(self):
        """Save feedback history to storage."""
        history_file = "data/meta_learning/reinforcement_feedback.json"
        
        try:
            # Limit size of history
            max_history = self.feedback_config.get("max_history", 1000)
            
            if len(self.feedback_history) > max_history:
                self.feedback_history = self.feedback_history[-max_history:]
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "reinforcement_feedback.json",
                    json.dumps(self.feedback_history),
                    mime_type="application/json"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving feedback history: {str(e)}")
    
    def calculate_reward(self, trade_result, prediction=None, model_type=None):
        """
        Calculate reward signal for a completed trade.
        
        Args:
            trade_result (dict): Trade outcome data
            prediction (dict, optional): Original prediction
            model_type (str, optional): Type of model ('entry', 'exit', 'sizing')
            
        Returns:
            float: Calculated reward value
        """
        try:
            # Default reward components
            pnl_reward = 0.0
            accuracy_reward = 0.0
            timing_reward = 0.0
            risk_reward = 0.0
            
            # Calculate PnL reward component
            if "pnl_percent" in trade_result:
                pnl_percent = trade_result["pnl_percent"]
                
                # Scale PnL to reward
                # Positive reward for profit, negative for loss
                profit_threshold = self.reward_factors.get("profit_threshold", 10.0)
                loss_threshold = self.reward_factors.get("loss_threshold", -20.0)
                
                if pnl_percent >= 0:
                    # Scale from 0 to 1 based on profit
                    pnl_reward = min(1.0, pnl_percent / profit_threshold)
                else:
                    # Scale from 0 to -1 based on loss
                    pnl_reward = max(-1.0, pnl_percent / abs(loss_threshold))
            
            # Different reward components based on model type
            if model_type == "entry":
                # Entry models are primarily judged on PnL
                # Add accuracy component if prediction available
                if prediction and "signal" in prediction:
                    predicted_direction = prediction["signal"]
                    
                    # Determine actual direction
                    if "pnl_percent" in trade_result:
                        actual_direction = "bullish" if trade_result["pnl_percent"] > 0 else "bearish"
                    elif "direction" in trade_result:
                        actual_direction = trade_result["direction"]
                    else:
                        actual_direction = "unknown"
                    
                    # Calculate accuracy reward
                    if predicted_direction == actual_direction:
                        accuracy_reward = 1.0
                    elif predicted_direction == "neutral" or actual_direction == "neutral":
                        accuracy_reward = 0.3
                    else:
                        accuracy_reward = -0.5
                
                # Add risk management component
                if "max_drawdown" in trade_result:
                    # Lower drawdown = better risk management
                    max_dd = trade_result["max_drawdown"]
                    risk_reward = max(0, 1.0 - (max_dd / 30.0))  # Scale based on 30% max drawdown
                
                # Calculate final reward with weights
                final_reward = (
                    pnl_reward * (1.0 - self.reward_factors.get("accuracy_weight", 0.3) - self.reward_factors.get("risk_weight", 0.2)) +
                    accuracy_reward * self.reward_factors.get("accuracy_weight", 0.3) +
                    risk_reward * self.reward_factors.get("risk_weight", 0.2)
                )
                
            elif model_type == "exit":
                # Exit models are judged on timing and opportunity cost
                
                # Add timing component
                if "timing_efficiency" in trade_result:
                    timing_reward = trade_result["timing_efficiency"]
                elif "exit_efficiency" in trade_result:
                    timing_reward = trade_result["exit_efficiency"]
                elif "optimal_exit_pct" in trade_result and "actual_exit_pct" in trade_result:
                    # Calculate how close to optimal the exit was
                    optimal = trade_result["optimal_exit_pct"]
                    actual = trade_result["actual_exit_pct"]
                    timing_reward = max(0, 1.0 - min(1.0, abs(optimal - actual) / optimal))
                
                # Add accuracy component
                if prediction and "should_exit" in prediction and "optimal_exit" in trade_result:
                    accuracy_reward = 1.0 if prediction["should_exit"] == trade_result["optimal_exit"] else -0.5
                
                # Calculate final reward with weights
                final_reward = (
                    pnl_reward * (1.0 - self.reward_factors.get("time_weight", 0.3) - self.reward_factors.get("accuracy_weight", 0.3)) +
                    timing_reward * self.reward_factors.get("time_weight", 0.3) +
                    accuracy_reward * self.reward_factors.get("accuracy_weight", 0.3)
                )
                
            elif model_type == "sizing":
                # Sizing models are judged on risk-adjusted returns
                
                # Add sizing accuracy component
                if "optimal_size" in trade_result and "actual_size" in trade_result:
                    optimal = trade_result["optimal_size"]
                    actual = trade_result["actual_size"]
                    
                    if optimal > 0:
                        # Calculate sizing error (lower is better)
                        size_error = abs(actual - optimal) / optimal
                        accuracy_reward = max(0, 1.0 - min(1.0, size_error))
                    else:
                        # If optimal size is 0, accuracy depends on how small actual size was
                        accuracy_reward = 1.0 if actual < 100 else 0.0
                
                # Add risk component
                if "risk_adjusted_return" in trade_result:
                    risk_reward = min(1.0, max(-1.0, trade_result["risk_adjusted_return"]))
                elif "sharpe_ratio" in trade_result:
                    risk_reward = min(1.0, max(-1.0, trade_result["sharpe_ratio"] / 3.0))
                
                # Calculate final reward with weights
                final_reward = (
                    pnl_reward * (1.0 - self.reward_factors.get("accuracy_weight", 0.4) - self.reward_factors.get("risk_weight", 0.4)) +
                    accuracy_reward * self.reward_factors.get("accuracy_weight", 0.4) +
                    risk_reward * self.reward_factors.get("risk_weight", 0.4)
                )
                
            else:
                # Generic model reward focused on PnL and accuracy
                if prediction and "accuracy" in trade_result:
                    accuracy_reward = trade_result["accuracy"]
                
                # Calculate final reward
                final_reward = (
                    pnl_reward * (1.0 - self.reward_factors.get("accuracy_weight", 0.3)) +
                    accuracy_reward * self.reward_factors.get("accuracy_weight", 0.3)
                )
            
            # Ensure reward is within proper range
            final_reward = max(-1.0, min(1.0, final_reward))
            
            # Log reward calculation
            self.logger.debug(
                f"Calculated reward {final_reward:.4f} (PnL: {pnl_reward:.4f}, "
                f"Accuracy: {accuracy_reward:.4f}, Timing: {timing_reward:.4f}, "
                f"Risk: {risk_reward:.4f})"
            )
            
            return final_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
    
    def provide_feedback(self, model_name, state, action, reward, next_state=None, done=False):
        """
        Provide feedback to a reinforcement learning model.
        
        Args:
            model_name (str): Name of the model
            state: State representation when action was taken
            action: Action that was taken
            reward (float): Reward signal
            next_state: Resulting state (optional)
            done (bool): Whether this is the final step in an episode
            
        Returns:
            bool: True if feedback was successfully recorded
        """
        try:
            # Create feedback record
            feedback = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_name,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }
            
            # Add to history
            self.feedback_history.append(feedback)
            
            # Periodically save history
            if len(self.feedback_history) % 50 == 0:
                self._save_feedback_history()
            
            self.logger.debug(f"Provided feedback to {model_name}: reward={reward:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error providing feedback: {str(e)}")
            return False
    
    def get_feedback_batch(self, model_name, batch_size=32):
        """
        Get a batch of feedback for model training.
        
        Args:
            model_name (str): Name of the model
            batch_size (int): Number of feedback records to return
            
        Returns:
            list: Batch of feedback records
        """
        try:
            # Filter to relevant model
            model_feedback = [f for f in self.feedback_history if f["model_name"] == model_name]
            
            if not model_feedback:
                return []
            
            # Return random batch (or all if less than batch size)
            if len(model_feedback) <= batch_size:
                return model_feedback
            else:
                return np.random.choice(model_feedback, size=batch_size, replace=False).tolist()
                
        except Exception as e:
            self.logger.error(f"Error getting feedback batch: {str(e)}")
            return []
    
    def get_reward_distribution(self, model_name, days=30):
        """
        Get the distribution of rewards for a model.
        
        Args:
            model_name (str): Name of the model
            days (int): Number of days to include
            
        Returns:
            dict: Reward distribution statistics
        """
        try:
            # Get cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            
            # Filter feedback
            recent_feedback = [
                f for f in self.feedback_history 
                if f["model_name"] == model_name and f.get("timestamp", "") >= cutoff_str
            ]
            
            if not recent_feedback:
                return {
                    "model_name": model_name,
                    "days": days,
                    "count": 0,
                    "message": "No feedback data found"
                }
            
            # Extract rewards
            rewards = [f["reward"] for f in recent_feedback]
            
            # Calculate statistics
            avg_reward = np.mean(rewards)
            median_reward = np.median(rewards)
            std_reward = np.std(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)
            
            # Calculate percentiles
            percentiles = {
                "p10": np.percentile(rewards, 10),
                "p25": np.percentile(rewards, 25),
                "p50": np.percentile(rewards, 50),
                "p75": np.percentile(rewards, 75),
                "p90": np.percentile(rewards, 90)
            }
            
            # Count positive and negative rewards
            positive_count = sum(1 for r in rewards if r > 0)
            negative_count = sum(1 for r in rewards if r < 0)
            
            return {
                "model_name": model_name,
                "days": days,
                "count": len(rewards),
                "avg_reward": avg_reward,
                "median_reward": median_reward,
                "std_reward": std_reward,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "percentiles": percentiles,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_ratio": positive_count / len(rewards) if rewards else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting reward distribution: {str(e)}")
            return {"error": str(e)}
    
    def get_reward_trend(self, model_name, days=30, window_size=20):
        """
        Get the trend of rewards over time for a model.
        
        Args:
            model_name (str): Name of the model
            days (int): Number of days to include
            window_size (int): Window size for rolling average
            
        Returns:
            dict: Reward trend data
        """
        try:
            # Get cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            
            # Filter feedback
            recent_feedback = [
                f for f in self.feedback_history 
                if f["model_name"] == model_name and f.get("timestamp", "") >= cutoff_str
            ]
            
            if not recent_feedback:
                return {
                    "model_name": model_name,
                    "days": days,
                    "count": 0,
                    "message": "No feedback data found"
                }
            
            # Sort by timestamp
            sorted_feedback = sorted(recent_feedback, key=lambda x: x.get("timestamp", ""))
            
            # Extract timestamps and rewards
            timestamps = [f.get("timestamp", "") for f in sorted_feedback]
            rewards = [f["reward"] for f in sorted_feedback]
            
            # Calculate rolling average
            if len(rewards) >= window_size:
                rolling_rewards = []
                
                for i in range(len(rewards) - window_size + 1):
                    window_avg = sum(rewards[i:i+window_size]) / window_size
                    rolling_rewards.append(window_avg)
                
                rolling_timestamps = timestamps[window_size-1:]
            else:
                rolling_rewards = rewards
                rolling_timestamps = timestamps
            
            # Calculate trend
            if len(rolling_rewards) >= 2:
                # Simple linear regression
                x = np.arange(len(rolling_rewards))
                coeffs = np.polyfit(x, rolling_rewards, 1)
                trend = coeffs[0]
            else:
                trend = 0.0
            
            return {
                "model_name": model_name,
                "days": days,
                "count": len(rewards),
                "window_size": window_size,
                "timestamps": timestamps,
                "rewards": rewards,
                "rolling_timestamps": rolling_timestamps,
                "rolling_rewards": rolling_rewards,
                "trend": trend
            }
            
        except Exception as e:
            self.logger.error(f"Error getting reward trend: {str(e)}")
            return {"error": str(e)}
    
    def calculate_state_values(self, model_name):
        """
        Calculate the average value of different states for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: State value mapping
        """
        try:
            # Filter to relevant model
            model_feedback = [f for f in self.feedback_history if f["model_name"] == model_name]
            
            if not model_feedback:
                return {"message": "No feedback data found"}
            
            # Group by state
            state_values = {}
            state_counts = {}
            
            for feedback in model_feedback:
                # Convert state to string for dictionary key
                state_str = str(feedback["state"])
                reward = feedback["reward"]
                
                if state_str not in state_values:
                    state_values[state_str] = 0.0
                    state_counts[state_str] = 0
                
                state_values[state_str] += reward
                state_counts[state_str] += 1
            
            # Calculate average values
            avg_values = {
                state: value / state_counts[state]
                for state, value in state_values.items()
            }
            
            # Sort by value (descending)
            sorted_values = sorted(
                avg_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                "model_name": model_name,
                "state_count": len(avg_values),
                "highest_value_states": sorted_values[:10],
                "lowest_value_states": sorted_values[-10:],
                "state_values": avg_values
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating state values: {str(e)}")
            return {"error": str(e)}
    
    def calculate_action_values(self, model_name):
        """
        Calculate the average value of different actions for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Action value mapping
        """
        try:
            # Filter to relevant model
            model_feedback = [f for f in self.feedback_history if f["model_name"] == model_name]
            
            if not model_feedback:
                return {"message": "No feedback data found"}
            
            # Group by action
            action_values = {}
            action_counts = {}
            
            for feedback in model_feedback:
                # Convert action to string for dictionary key
                action_str = str(feedback["action"])
                reward = feedback["reward"]
                
                if action_str not in action_values:
                    action_values[action_str] = 0.0
                    action_counts[action_str] = 0
                
                action_values[action_str] += reward
                action_counts[action_str] += 1
            
            # Calculate average values
            avg_values = {
                action: value / action_counts[action]
                for action, value in action_values.items()
            }
            
            # Sort by value (descending)
            sorted_values = sorted(
                avg_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                "model_name": model_name,
                "action_count": len(avg_values),
                "best_actions": sorted_values[:5],
                "worst_actions": sorted_values[-5:],
                "action_values": avg_values
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating action values: {str(e)}")
            return {"error": str(e)}
    
    def get_feedback_summary(self, days=30):
        """
        Get summary of feedback across all models.
        
        Args:
            days (int): Number of days to include
            
        Returns:
            dict: Feedback summary
        """
        try:
            # Get cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            
            # Filter feedback
            recent_feedback = [
                f for f in self.feedback_history 
                if f.get("timestamp", "") >= cutoff_str
            ]
            
            if not recent_feedback:
                return {"days": days, "count": 0, "message": "No feedback data found"}
            
            # Group by model
            model_groups = {}
            
            for feedback in recent_feedback:
                model_name = feedback["model_name"]
                
                if model_name not in model_groups:
                    model_groups[model_name] = []
                
                model_groups[model_name].append(feedback)
            
            # Calculate summary for each model
            model_summaries = {}
            
            for model_name, model_feedback in model_groups.items():
                rewards = [f["reward"] for f in model_feedback]
                
                model_summaries[model_name] = {
                    "count": len(rewards),
                    "avg_reward": np.mean(rewards),
                    "positive_ratio": sum(1 for r in rewards if r > 0) / len(rewards),
                    "latest_reward": model_feedback[-1]["reward"] if model_feedback else 0
                }
            
            # Calculate overall statistics
            all_rewards = [f["reward"] for f in recent_feedback]
            
            overall_stats = {
                "count": len(all_rewards),
                "avg_reward": np.mean(all_rewards),
                "median_reward": np.median(all_rewards),
                "positive_ratio": sum(1 for r in all_rewards if r > 0) / len(all_rewards),
                "models": len(model_groups)
            }
            
            return {
                "days": days,
                "overall": overall_stats,
                "models": model_summaries
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feedback summary: {str(e)}")
            return {"error": str(e)}
    
    def clear_feedback_history(self, model_name=None, days_to_keep=30):
        """
        Clear old feedback history to save space.
        
        Args:
            model_name (str, optional): Only clear for specific model
            days_to_keep (int): Number of days of recent history to keep
            
        Returns:
            int: Number of records removed
        """
        try:
            # Get cutoff date
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            
            # Count records before clearing
            original_count = len(self.feedback_history)
            
            # Filter feedback to keep
            if model_name:
                # Only clear specific model
                self.feedback_history = [
                    f for f in self.feedback_history 
                    if f["model_name"] != model_name or f.get("timestamp", "") >= cutoff_str
                ]
            else:
                # Clear all models
                self.feedback_history = [
                    f for f in self.feedback_history 
                    if f.get("timestamp", "") >= cutoff_str
                ]
            
            # Count records removed
            removed_count = original_count - len(self.feedback_history)
            
            # Save updated history
            self._save_feedback_history()
            
            self.logger.info(f"Cleared {removed_count} feedback records")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error clearing feedback history: {str(e)}")
            return 0
    
    def set_reward_factors(self, factors):
        """
        Update reward scaling factors.
        
        Args:
            factors (dict): New reward factors
            
        Returns:
            dict: Updated reward factors
        """
        try:
            # Update with provided factors
            for key, value in factors.items():
                if key in self.reward_factors:
                    self.reward_factors[key] = value
            
            self.logger.info(f"Updated reward factors: {self.reward_factors}")
            return self.reward_factors
            
        except Exception as e:
            self.logger.error(f"Error setting reward factors: {str(e)}")
            return self.reward_factors
