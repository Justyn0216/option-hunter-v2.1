"""
Delta Monitor Module

This module monitors changes in option delta and provides delta-based
exit signals for the options trading system.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class DeltaMonitor:
    """
    Monitors the delta of option positions and provides exit signals
    based on changes in delta or when delta crosses predefined thresholds.
    
    Delta-based exits are useful for:
    - Risk management (reducing exposure as delta increases)
    - Exiting when directional exposure has changed significantly
    - Managing deep ITM/OTM positions
    """
    
    def __init__(self, config, threshold_optimizer=None):
        """
        Initialize the delta monitor.
        
        Args:
            config (dict): Configuration dictionary
            threshold_optimizer: Optional GreekThresholdOptimizer instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("delta_monitor", {})
        self.threshold_optimizer = threshold_optimizer
        
        # Default thresholds (will be overridden by optimizer if available)
        self.call_high_delta_threshold = self.config.get("call_high_delta_threshold", 0.85)
        self.call_low_delta_threshold = self.config.get("call_low_delta_threshold", 0.15)
        self.put_high_delta_threshold = self.config.get("put_high_delta_threshold", -0.85)
        self.put_low_delta_threshold = self.config.get("put_low_delta_threshold", -0.15)
        
        # Delta change thresholds
        self.delta_change_threshold = self.config.get("delta_change_threshold", 0.20)
        self.delta_acceleration_threshold = self.config.get("delta_acceleration_threshold", 0.05)
        
        # Tracking delta history for positions
        self.delta_history = {}
        self.max_history_length = self.config.get("max_history_length", 20)
        
        # Track exit signals for logging/analysis
        self.exit_signals = []
        
        self.logger.info("DeltaMonitor initialized")
    
    def update_thresholds(self, optimization_results=None):
        """
        Update delta thresholds, either from optimizer or configuration.
        
        Args:
            optimization_results (dict, optional): Results from optimizer
        """
        if optimization_results is not None:
            # Update from optimizer results
            self.call_high_delta_threshold = optimization_results.get("call_high_delta", self.call_high_delta_threshold)
            self.call_low_delta_threshold = optimization_results.get("call_low_delta", self.call_low_delta_threshold)
            self.put_high_delta_threshold = optimization_results.get("put_high_delta", self.put_high_delta_threshold)
            self.put_low_delta_threshold = optimization_results.get("put_low_delta", self.put_low_delta_threshold)
            self.delta_change_threshold = optimization_results.get("delta_change", self.delta_change_threshold)
            
            self.logger.info(f"Updated delta thresholds from optimizer: " +
                             f"Call high: {self.call_high_delta_threshold}, " +
                             f"Call low: {self.call_low_delta_threshold}, " +
                             f"Put high: {self.put_high_delta_threshold}, " +
                             f"Put low: {self.put_low_delta_threshold}")
            
        elif self.threshold_optimizer:
            # Get latest optimized thresholds
            optimal_thresholds = self.threshold_optimizer.get_optimal_thresholds("delta")
            if optimal_thresholds:
                self.update_thresholds(optimal_thresholds)
        else:
            # Use configuration values
            self.call_high_delta_threshold = self.config.get("call_high_delta_threshold", 0.85)
            self.call_low_delta_threshold = self.config.get("call_low_delta_threshold", 0.15)
            self.put_high_delta_threshold = self.config.get("put_high_delta_threshold", -0.85)
            self.put_low_delta_threshold = self.config.get("put_low_delta_threshold", -0.15)
            self.delta_change_threshold = self.config.get("delta_change_threshold", 0.20)
    
    def update_delta(self, trade_id, current_delta, trade_data):
        """
        Update delta history for a position.
        
        Args:
            trade_id (str): Trade identifier
            current_delta (float): Current delta value
            trade_data (dict): Current trade data
            
        Returns:
            list: Delta history for the position
        """
        # Initialize history if not exists
        if trade_id not in self.delta_history:
            self.delta_history[trade_id] = []
        
        # Add timestamp and metadata
        delta_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "delta": current_delta,
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "option_price": trade_data.get("current_price", 0.0),
            "pnl_percent": trade_data.get("current_pnl_percent", 0.0)
        }
        
        # Add to history
        self.delta_history[trade_id].append(delta_entry)
        
        # Limit history length
        if len(self.delta_history[trade_id]) > self.max_history_length:
            self.delta_history[trade_id] = self.delta_history[trade_id][-self.max_history_length:]
        
        return self.delta_history[trade_id]
    
    def check_exit_signal(self, trade_id, trade_data):
        """
        Check if current delta indicates an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Current trade data
            
        Returns:
            dict: Exit signal if detected, None otherwise
        """
        if trade_id not in self.delta_history or not self.delta_history[trade_id]:
            self.logger.warning(f"No delta history for trade {trade_id}")
            return None
        
        # Get current delta
        current_delta = trade_data.get("greeks", {}).get("delta", 0.0)
        option_type = trade_data.get("option_type", "").lower()
        days_to_expiration = trade_data.get("days_to_expiration", 0)
        
        # Add delta to history
        self.update_delta(trade_id, current_delta, trade_data)
        
        # Get delta history
        delta_history = self.delta_history[trade_id]
        
        # Need at least 2 data points for change detection
        if len(delta_history) < 2:
            return None
        
        # Check absolute threshold crossings
        signal = self._check_threshold_crossing(trade_id, current_delta, option_type, trade_data)
        if signal:
            return signal
        
        # Check for significant delta changes
        signal = self._check_delta_change(trade_id, delta_history, option_type, trade_data)
        if signal:
            return signal
        
        # Check for delta acceleration near expiration
        if days_to_expiration <= 5:
            signal = self._check_delta_acceleration(trade_id, delta_history, option_type, trade_data)
            if signal:
                return signal
        
        # No exit signal
        return None
    
    def _check_threshold_crossing(self, trade_id, current_delta, option_type, trade_data):
        """
        Check if delta has crossed a predefined threshold.
        
        Args:
            trade_id (str): Trade identifier
            current_delta (float): Current delta value
            option_type (str): Option type (call or put)
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if threshold crossed, None otherwise
        """
        # Check if we've crossed threshold based on option type
        if option_type == "call":
            if current_delta >= self.call_high_delta_threshold:
                return self._create_exit_signal(
                    trade_id,
                    "delta_high_threshold",
                    f"Call delta reached high threshold: {current_delta:.2f} >= {self.call_high_delta_threshold:.2f}",
                    current_delta,
                    trade_data
                )
            elif current_delta <= self.call_low_delta_threshold and current_delta > 0:
                return self._create_exit_signal(
                    trade_id,
                    "delta_low_threshold",
                    f"Call delta reached low threshold: {current_delta:.2f} <= {self.call_low_delta_threshold:.2f}",
                    current_delta,
                    trade_data
                )
        elif option_type == "put":
            if current_delta <= self.put_high_delta_threshold:
                return self._create_exit_signal(
                    trade_id,
                    "delta_high_threshold",
                    f"Put delta reached high threshold: {current_delta:.2f} <= {self.put_high_delta_threshold:.2f}",
                    current_delta,
                    trade_data
                )
            elif current_delta >= self.put_low_delta_threshold and current_delta < 0:
                return self._create_exit_signal(
                    trade_id,
                    "delta_low_threshold",
                    f"Put delta reached low threshold: {current_delta:.2f} >= {self.put_low_delta_threshold:.2f}",
                    current_delta,
                    trade_data
                )
        
        # No threshold crossing
        return None
    
    def _check_delta_change(self, trade_id, delta_history, option_type, trade_data):
        """
        Check for significant changes in delta.
        
        Args:
            trade_id (str): Trade identifier
            delta_history (list): Delta history for the position
            option_type (str): Option type (call or put)
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if significant change detected, None otherwise
        """
        # Need at least 2 data points
        if len(delta_history) < 2:
            return None
        
        # Get recent delta values
        recent_delta = delta_history[-1]["delta"]
        prev_delta = delta_history[-2]["delta"]
        
        # Calculate absolute change
        delta_change = abs(recent_delta - prev_delta)
        
        # If change is significant
        if delta_change >= self.delta_change_threshold:
            # For calls, check if delta decreased significantly
            if option_type == "call" and recent_delta < prev_delta:
                return self._create_exit_signal(
                    trade_id,
                    "delta_decrease",
                    f"Call delta decreased significantly: {prev_delta:.2f} -> {recent_delta:.2f} (Δ{delta_change:.2f})",
                    recent_delta,
                    trade_data
                )
            # For puts, check if delta increased significantly (less negative)
            elif option_type == "put" and recent_delta > prev_delta:
                return self._create_exit_signal(
                    trade_id,
                    "delta_decrease",
                    f"Put delta decreased significantly: {prev_delta:.2f} -> {recent_delta:.2f} (Δ{delta_change:.2f})",
                    recent_delta,
                    trade_data
                )
        
        # No significant change
        return None
    
    def _check_delta_acceleration(self, trade_id, delta_history, option_type, trade_data):
        """
        Check for acceleration in delta changes (second derivative).
        
        Args:
            trade_id (str): Trade identifier
            delta_history (list): Delta history for the position
            option_type (str): Option type (call or put)
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if acceleration detected, None otherwise
        """
        # Need at least 3 data points
        if len(delta_history) < 3:
            return None
        
        # Get delta values
        delta_3 = delta_history[-3]["delta"]
        delta_2 = delta_history[-2]["delta"]
        delta_1 = delta_history[-1]["delta"]
        
        # Calculate first differences
        diff_1 = delta_2 - delta_3
        diff_2 = delta_1 - delta_2
        
        # Calculate acceleration (second difference)
        acceleration = diff_2 - diff_1
        
        # Check for significant acceleration
        if abs(acceleration) >= self.delta_acceleration_threshold:
            # For calls, check negative acceleration (delta slowing/reversing)
            if option_type == "call" and acceleration < 0:
                return self._create_exit_signal(
                    trade_id,
                    "delta_deceleration",
                    f"Call delta momentum slowing: acceleration = {acceleration:.4f}",
                    delta_1,
                    trade_data
                )
            # For puts, check positive acceleration (delta slowing/reversing)
            elif option_type == "put" and acceleration > 0:
                return self._create_exit_signal(
                    trade_id,
                    "delta_deceleration",
                    f"Put delta momentum slowing: acceleration = {acceleration:.4f}",
                    delta_1,
                    trade_data
                )
        
        # No significant acceleration
        return None
    
    def _create_exit_signal(self, trade_id, signal_type, description, delta_value, trade_data):
        """
        Create an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            signal_type (str): Type of exit signal
            description (str): Description of the signal
            delta_value (float): Current delta value
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal
        """
        signal = {
            "trade_id": trade_id,
            "signal_type": signal_type,
            "greek": "delta",
            "description": description,
            "delta_value": delta_value,
            "confidence": self._calculate_signal_confidence(signal_type, delta_value, trade_data),
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
        
        # Log the signal
        self.logger.info(f"Delta exit signal for {trade_id}: {description} (confidence: {signal['confidence']:.2f})")
        
        # Add to exit signals history
        self.exit_signals.append(signal)
        if len(self.exit_signals) > 100:
            self.exit_signals = self.exit_signals[-100:]
        
        return signal
    
    def _calculate_signal_confidence(self, signal_type, delta_value, trade_data):
        """
        Calculate confidence level for the exit signal.
        
        Args:
            signal_type (str): Type of exit signal
            delta_value (float): Current delta value
            trade_data (dict): Trade data
            
        Returns:
            float: Confidence level (0-1)
        """
        base_confidence = 0.7  # Default confidence
        
        # Adjust based on signal type
        if signal_type == "delta_high_threshold":
            # More confident as delta approaches 1.0 (calls) or -1.0 (puts)
            if trade_data.get("option_type") == "call":
                base_confidence += min(0.3, (abs(delta_value) - self.call_high_delta_threshold) * 2)
            else:
                base_confidence += min(0.3, (abs(delta_value) - abs(self.put_high_delta_threshold)) * 2)
        elif signal_type == "delta_low_threshold":
            # More confident as delta approaches 0
            if trade_data.get("option_type") == "call":
                base_confidence += min(0.3, (self.call_low_delta_threshold - delta_value) * 3)
            else:
                base_confidence += min(0.3, (delta_value - self.put_low_delta_threshold) * 3)
        elif signal_type == "delta_decrease":
            # More confident with larger decreases
            delta_history = self.delta_history.get(trade_data.get("trade_id", ""), [])
            if len(delta_history) >= 2:
                change = abs(delta_history[-1]["delta"] - delta_history[-2]["delta"])
                base_confidence += min(0.3, change * 0.5)
        elif signal_type == "delta_deceleration":
            # More confident with larger acceleration
            base_confidence += min(0.2, abs(delta_value) * 0.25)
        
        # Adjust for profit/loss situation
        pnl_percent = trade_data.get("current_pnl_percent", 0.0)
        if pnl_percent > 20:
            # More confident in taking profits
            base_confidence += min(0.1, pnl_percent / 200)  # Max +0.1 for 20% profit
        elif pnl_percent < -10:
            # Less confident in taking losses (might want to wait for recovery)
            base_confidence -= min(0.2, abs(pnl_percent) / 50)  # Max -0.2 for big losses
        
        # Adjust for time to expiration
        days_to_expiration = trade_data.get("days_to_expiration", 30)
        if days_to_expiration < 5:
            # More confident when close to expiration
            base_confidence += min(0.2, (5 - days_to_expiration) * 0.04)
        
        return max(0.0, min(1.0, base_confidence))  # Clamp to 0-1 range
    
    def get_delta_history(self, trade_id):
        """
        Get delta history for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            list: Delta history
        """
        return self.delta_history.get(trade_id, [])
    
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
        Clear delta history for a specific trade or all trades.
        
        Args:
            trade_id (str, optional): Trade to clear. If None, clear all.
        """
        if trade_id:
            if trade_id in self.delta_history:
                del self.delta_history[trade_id]
                self.logger.debug(f"Cleared delta history for trade {trade_id}")
        else:
            self.delta_history = {}
            self.logger.debug("Cleared all delta history")
    
    def get_thresholds(self):
        """
        Get current delta thresholds.
        
        Returns:
            dict: Current thresholds
        """
        return {
            "call_high_delta_threshold": self.call_high_delta_threshold,
            "call_low_delta_threshold": self.call_low_delta_threshold,
            "put_high_delta_threshold": self.put_high_delta_threshold,
            "put_low_delta_threshold": self.put_low_delta_threshold,
            "delta_change_threshold": self.delta_change_threshold,
            "delta_acceleration_threshold": self.delta_acceleration_threshold
        }
