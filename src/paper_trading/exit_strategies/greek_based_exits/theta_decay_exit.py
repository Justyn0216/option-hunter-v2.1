"""
Theta Decay Exit Module

This module monitors option theta decay and provides theta-based
exit signals for the options trading system.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class ThetaDecayExit:
    """
    Monitors option theta decay and provides exit signals based on
    accelerated time decay, theta-to-premium ratio, and optimal theta
    exit points.
    
    Theta-based exits are useful for:
    - Avoiding rapid time decay near expiration
    - Optimizing premium collection in theta-positive trades
    - Managing overnight/weekend decay
    """
    
    def __init__(self, config, threshold_optimizer=None):
        """
        Initialize the theta decay exit monitor.
        
        Args:
            config (dict): Configuration dictionary
            threshold_optimizer: Optional GreekThresholdOptimizer instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("theta_decay_exit", {})
        self.threshold_optimizer = threshold_optimizer
        
        # Default thresholds (will be overridden by optimizer if available)
        self.high_theta_ratio_threshold = self.config.get("high_theta_ratio_threshold", 0.04)  # Theta/price ratio
        self.theta_acceleration_threshold = self.config.get("theta_acceleration_threshold", 0.005)
        self.theta_percentage_threshold = self.config.get("theta_percentage_threshold", 3.0)  # % of option price
        
        # Special thresholds for near-expiration options
        self.near_expiry_days = self.config.get("near_expiry_days", 5)
        self.weekend_decay_threshold = self.config.get("weekend_decay_threshold", 4.0)  # % for weekend decay
        
        # Tracking theta history for positions
        self.theta_history = {}
        self.max_history_length = self.config.get("max_history_length", 20)
        
        # Track exit signals for logging/analysis
        self.exit_signals = []
        
        self.logger.info("ThetaDecayExit initialized")
    
    def update_thresholds(self, optimization_results=None):
        """
        Update theta thresholds, either from optimizer or configuration.
        
        Args:
            optimization_results (dict, optional): Results from optimizer
        """
        if optimization_results is not None:
            # Update from optimizer results
            self.high_theta_ratio_threshold = optimization_results.get("high_theta_ratio", self.high_theta_ratio_threshold)
            self.theta_acceleration_threshold = optimization_results.get("theta_acceleration", self.theta_acceleration_threshold)
            self.theta_percentage_threshold = optimization_results.get("theta_percentage", self.theta_percentage_threshold)
            
            self.logger.info(f"Updated theta thresholds from optimizer: " +
                             f"Theta ratio: {self.high_theta_ratio_threshold}, " +
                             f"Acceleration: {self.theta_acceleration_threshold}, " +
                             f"Percentage: {self.theta_percentage_threshold}%")
            
        elif self.threshold_optimizer:
            # Get latest optimized thresholds
            optimal_thresholds = self.threshold_optimizer.get_optimal_thresholds("theta")
            if optimal_thresholds:
                self.update_thresholds(optimal_thresholds)
        else:
            # Use configuration values
            self.high_theta_ratio_threshold = self.config.get("high_theta_ratio_threshold", 0.04)
            self.theta_acceleration_threshold = self.config.get("theta_acceleration_threshold", 0.005)
            self.theta_percentage_threshold = self.config.get("theta_percentage_threshold", 3.0)
    
    def update_theta(self, trade_id, current_theta, trade_data):
        """
        Update theta history for a position.
        
        Args:
            trade_id (str): Trade identifier
            current_theta (float): Current theta value (typically negative)
            trade_data (dict): Current trade data
            
        Returns:
            list: Theta history for the position
        """
        # Initialize history if not exists
        if trade_id not in self.theta_history:
            self.theta_history[trade_id] = []
        
        # Add timestamp and metadata
        theta_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "theta": current_theta,  # Typically negative
            "option_price": trade_data.get("current_price", 0.0),
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "days_to_expiration": trade_data.get("days_to_expiration", 0.0),
            "pnl_percent": trade_data.get("current_pnl_percent", 0.0),
            # Calculate theta as percentage of option price
            "theta_percent": (abs(current_theta) / trade_data.get("current_price", 0.01)) * 100 if trade_data.get("current_price", 0) > 0 else 0.0
        }
        
        # Add to history
        self.theta_history[trade_id].append(theta_entry)
        
        # Limit history length
        if len(self.theta_history[trade_id]) > self.max_history_length:
            self.theta_history[trade_id] = self.theta_history[trade_id][-self.max_history_length:]
        
        return self.theta_history[trade_id]
    
    def check_exit_signal(self, trade_id, trade_data):
        """
        Check if current theta indicates an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Current trade data
            
        Returns:
            dict: Exit signal if detected, None otherwise
        """
        if trade_id not in self.theta_history or not self.theta_history[trade_id]:
            # No history yet, update and return
            current_theta = trade_data.get("greeks", {}).get("theta", 0.0)
            self.update_theta(trade_id, current_theta, trade_data)
            return None
        
        # Get current theta (should be negative for long options)
        current_theta = trade_data.get("greeks", {}).get("theta", 0.0)
        days_to_expiration = trade_data.get("days_to_expiration", 0.0)
        option_price = trade_data.get("current_price", 0.0)
        option_type = trade_data.get("option_type", "").lower()
        
        # For short options (sellers), theta is positive income
        if trade_data.get("position_type", "long") == "short":
            # Short option sellers want theta decay, different logic needed
            return self._check_short_theta_exit(trade_id, current_theta, trade_data)
        
        # Add theta to history
        self.update_theta(trade_id, current_theta, trade_data)
        
        # Get theta history
        theta_history = self.theta_history[trade_id]
        
        # Need at least 2 data points
        if len(theta_history) < 2:
            return None
        
        # For long positions, theta is negative (cost)
        
        # Check for near-expiration exit signals
        if days_to_expiration <= self.near_expiry_days:
            signal = self._check_near_expiry_theta(trade_id, theta_history, trade_data)
            if signal:
                return signal
        
        # Check for high theta-to-premium ratio
        signal = self._check_theta_premium_ratio(trade_id, theta_history, trade_data)
        if signal:
            return signal
        
        # Check for accelerating theta decay
        if len(theta_history) >= 3:
            signal = self._check_theta_acceleration(trade_id, theta_history, trade_data)
            if signal:
                return signal
        
        # Check for weekend decay
        signal = self._check_weekend_decay(trade_id, trade_data)
        if signal:
            return signal
        
        # No exit signal
        return None
    
    def _check_near_expiry_theta(self, trade_id, theta_history, trade_data):
        """
        Check for excessive theta decay near expiration.
        
        Args:
            trade_id (str): Trade identifier
            theta_history (list): Theta history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if excessive decay detected, None otherwise
        """
        days_to_expiration = trade_data.get("days_to_expiration", 0.0)
        
        if days_to_expiration <= self.near_expiry_days:
            # Get most recent theta data
            recent_theta = theta_history[-1]
            
            # Calculate theta as percentage of option price
            theta_percent = recent_theta["theta_percent"]
            
            # Higher threshold as expiration approaches
            dynamic_threshold = self.theta_percentage_threshold * (1 + (self.near_expiry_days - days_to_expiration) / self.near_expiry_days)
            
            if theta_percent > dynamic_threshold:
                # Higher confidence with less days
                confidence_boost = max(0, min(0.3, (self.near_expiry_days - days_to_expiration) / self.near_expiry_days * 0.3))
                
                return self._create_exit_signal(
                    trade_id,
                    "near_expiry_theta_decay",
                    f"Excessive theta decay near expiration: {theta_percent:.2f}% per day with {days_to_expiration:.1f} days left",
                    recent_theta["theta"],
                    trade_data,
                    confidence_boost=confidence_boost
                )
        
        return None
    
    def _check_theta_premium_ratio(self, trade_id, theta_history, trade_data):
        """
        Check if theta represents too high a percentage of option premium.
        
        Args:
            trade_id (str): Trade identifier
            theta_history (list): Theta history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if ratio too high, None otherwise
        """
        # Get most recent theta data
        recent_theta = theta_history[-1]
        
        # Absolute theta value (theta is typically negative for long options)
        abs_theta = abs(recent_theta["theta"])
        option_price = recent_theta["option_price"]
        
        # Calculate theta to premium ratio
        if option_price > 0:
            theta_ratio = abs_theta / option_price
            
            if theta_ratio > self.high_theta_ratio_threshold:
                # Calculate daily percentage loss
                daily_decay_pct = abs_theta / option_price * 100
                
                return self._create_exit_signal(
                    trade_id,
                    "high_theta_premium_ratio",
                    f"High theta/premium ratio: {theta_ratio:.4f} ({daily_decay_pct:.2f}% decay per day)",
                    recent_theta["theta"],
                    trade_data
                )
        
        return None
    
    def _check_theta_acceleration(self, trade_id, theta_history, trade_data):
        """
        Check for acceleration in theta decay.
        
        Args:
            trade_id (str): Trade identifier
            theta_history (list): Theta history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if acceleration detected, None otherwise
        """
        # Need at least 3 data points
        if len(theta_history) < 3:
            return None
        
        # Get theta values (use absolute values since theta is typically negative)
        theta_3 = abs(theta_history[-3]["theta"])
        theta_2 = abs(theta_history[-2]["theta"])
        theta_1 = abs(theta_history[-1]["theta"])
        
        # Get option prices to normalize
        price_3 = theta_history[-3]["option_price"]
        price_2 = theta_history[-2]["option_price"]
        price_1 = theta_history[-1]["option_price"]
        
        # Normalize theta to percentage of option price
        if price_3 > 0 and price_2 > 0 and price_1 > 0:
            norm_theta_3 = theta_3 / price_3
            norm_theta_2 = theta_2 / price_2
            norm_theta_1 = theta_1 / price_1
            
            # Calculate first differences
            diff_1 = norm_theta_2 - norm_theta_3
            diff_2 = norm_theta_1 - norm_theta_2
            
            # Calculate acceleration (second difference)
            acceleration = diff_2 - diff_1
            
            if acceleration > self.theta_acceleration_threshold:
                # Theta decay is accelerating
                return self._create_exit_signal(
                    trade_id,
                    "accelerating_theta_decay",
                    f"Theta decay accelerating: {acceleration:.6f} (normalized to option price)",
                    theta_history[-1]["theta"],
                    trade_data
                )
        
        return None
    
    def _check_weekend_decay(self, trade_id, trade_data):
        """
        Check if holding over weekend would cause excessive theta decay.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if weekend decay is concerning, None otherwise
        """
        # Only check on Friday or day before holiday
        today = datetime.now()
        is_friday = today.weekday() == 4  # 0 is Monday, 4 is Friday
        
        # Check if it's Friday afternoon
        is_friday_afternoon = is_friday and today.hour >= 14  # After 2 PM
        
        if is_friday_afternoon:
            # Get current theta
            current_theta = trade_data.get("greeks", {}).get("theta", 0.0)
            option_price = trade_data.get("current_price", 0.0)
            
            # Calculate 3-day decay (Friday close to Monday open)
            weekend_decay = abs(current_theta) * 3  # 3 days of decay
            
            if option_price > 0:
                # Calculate as percentage of current price
                weekend_decay_percent = (weekend_decay / option_price) * 100
                
                if weekend_decay_percent > self.weekend_decay_threshold:
                    return self._create_exit_signal(
                        trade_id,
                        "weekend_theta_decay",
                        f"Weekend theta decay concern: {weekend_decay_percent:.2f}% of premium over weekend",
                        current_theta,
                        trade_data,
                        confidence_boost=0.1
                    )
        
        return None
    
    def _check_short_theta_exit(self, trade_id, current_theta, trade_data):
        """
        Check for theta-based exit signals for short options (sellers).
        For short options, theta is positive income, so exit logic differs.
        
        Args:
            trade_id (str): Trade identifier
            current_theta (float): Current theta value (positive for shorts)
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if conditions met, None otherwise
        """
        # For short options, we're looking for diminishing returns
        
        # Update theta history
        self.update_theta(trade_id, current_theta, trade_data)
        theta_history = self.theta_history[trade_id]
        
        if len(theta_history) < 2:
            return None
        
        days_to_expiration = trade_data.get("days_to_expiration", 0.0)
        
        # Check if we've captured most of the theta decay
        if days_to_expiration <= 5:
            # Get recent theta values
            recent_theta = abs(theta_history[-1]["theta"])
            initial_theta = abs(theta_history[0]["theta"])
            
            # If theta has significantly diminished (collected most of the decay)
            if initial_theta > 0 and recent_theta < initial_theta * 0.3:
                return self._create_exit_signal(
                    trade_id,
                    "short_option_diminishing_returns",
                    f"Short option has captured most of the theta decay: {initial_theta:.4f} -> {recent_theta:.4f}",
                    current_theta,
                    trade_data
                )
            
            # If we've captured significant profit already
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            if pnl_percent > 50:  # Captured more than 50% of potential profit
                return self._create_exit_signal(
                    trade_id,
                    "short_option_profit_target",
                    f"Short option at {pnl_percent:.2f}% profit with {days_to_expiration:.1f} days remaining",
                    current_theta,
                    trade_data,
                    confidence_boost=0.1
                )
        
        return None
    
    def _create_exit_signal(self, trade_id, signal_type, description, theta_value, trade_data, confidence_boost=0):
        """
        Create an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            signal_type (str): Type of exit signal
            description (str): Description of the signal
            theta_value (float): Current theta value
            trade_data (dict): Trade data
            confidence_boost (float): Additional confidence value
            
        Returns:
            dict: Exit signal
        """
        signal = {
            "trade_id": trade_id,
            "signal_type": signal_type,
            "greek": "theta",
            "description": description,
            "theta_value": theta_value,
            "confidence": self._calculate_signal_confidence(signal_type, theta_value, trade_data) + confidence_boost,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_data": {
                "symbol": trade_data.get("symbol"),
                "option_type": trade_data.get("option_type"),
                "strike": trade_data.get("strike"),
                "expiration": trade_data.get("expiration"),
                "days_to_expiration": trade_data.get("days_to_expiration", 0),
                "current_pnl_percent": trade_data.get("current_pnl_percent", 0.0)
            }
        }
        
        # Clamp confidence to 0-1 range
        signal["confidence"] = max(0.0, min(1.0, signal["confidence"]))
        
        # Log the signal
        self.logger.info(f"Theta exit signal for {trade_id}: {description} (confidence: {signal['confidence']:.2f})")
        
        # Add to exit signals history
        self.exit_signals.append(signal)
        if len(self.exit_signals) > 100:
            self.exit_signals = self.exit_signals[-100:]
        
        return signal
    
    def _calculate_signal_confidence(self, signal_type, theta_value, trade_data):
        """
        Calculate confidence level for the exit signal.
        
        Args:
            signal_type (str): Type of exit signal
            theta_value (float): Current theta value
            trade_data (dict): Trade data
            
        Returns:
            float: Confidence level (0-1)
        """
        base_confidence = 0.7  # Default confidence
        
        # Get option price for normalizing
        option_price = trade_data.get("current_price", 0.01)
        days_to_expiration = trade_data.get("days_to_expiration", 0.0)
        
        # Adjust based on signal type
        if signal_type == "near_expiry_theta_decay":
            # More confident with less days and higher theta
            theta_pct = abs(theta_value) / option_price * 100
            dte_factor = min(0.2, (self.near_expiry_days - days_to_expiration) / self.near_expiry_days * 0.2)
            theta_factor = min(0.15, (theta_pct - self.theta_percentage_threshold) / 5 * 0.15)
            base_confidence += dte_factor + theta_factor
            
        elif signal_type == "high_theta_premium_ratio":
            # More confident with higher ratio
            theta_ratio = abs(theta_value) / option_price
            ratio_excess = theta_ratio - self.high_theta_ratio_threshold
            base_confidence += min(0.2, ratio_excess / self.high_theta_ratio_threshold * 0.2)
            
        elif signal_type == "accelerating_theta_decay":
            # Base confidence is fine
            pass
            
        elif signal_type == "weekend_theta_decay":
            # More confident with higher decay percentage
            base_confidence += 0.1  # Weekend signals get a boost
            
        elif signal_type == "short_option_diminishing_returns":
            # More confident as we get closer to expiration
            base_confidence += min(0.2, (5 - days_to_expiration) / 5 * 0.2)
            
        elif signal_type == "short_option_profit_target":
            # More confident with higher profit percentage
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            base_confidence += min(0.25, pnl_percent / 200)  # Max +0.25 for 50%+ profit
        
        # Adjust for profit/loss situation
        pnl_percent = trade_data.get("current_pnl_percent", 0.0)
        if pnl_percent > 15:
            # More confident in taking profits with high theta decay
            base_confidence += min(0.15, pnl_percent / 100)  # Max +0.15 for 15%+ profit
        elif pnl_percent < -10:
            # Less confident in taking losses with high theta
            base_confidence -= min(0.2, abs(pnl_percent) / 50)  # Max -0.2 for big losses
        
        return base_confidence
    
    def get_theta_history(self, trade_id):
        """
        Get theta history for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            list: Theta history
        """
        return self.theta_history.get(trade_id, [])
    
    def get_recent_signals(self, limit=10):
        """
        Get recent exit signals.
        
        Args:
            limit (int): Maximum number of signals to return
            
        Returns:
            list: Recent exit signals
        """
        return self.exit_signals[-limit:]
    
    def clear_history(self, trade_id=None):
        """
        Clear theta history for a specific trade or all trades.
        
        Args:
            trade_id (str, optional): Trade to clear. If None, clear all.
        """
        if trade_id:
            if trade_id in self.theta_history:
                del self.theta_history[trade_id]
                self.logger.debug(f"Cleared theta history for trade {trade_id}")
        else:
            self.theta_history = {}
            self.logger.debug("Cleared all theta history")
    
    def get_thresholds(self):
        """
        Get current theta thresholds.
        
        Returns:
            dict: Current thresholds
        """
        return {
            "high_theta_ratio_threshold": self.high_theta_ratio_threshold,
            "theta_acceleration_threshold": self.theta_acceleration_threshold,
            "theta_percentage_threshold": self.theta_percentage_threshold,
            "weekend_decay_threshold": self.weekend_decay_threshold
        }
