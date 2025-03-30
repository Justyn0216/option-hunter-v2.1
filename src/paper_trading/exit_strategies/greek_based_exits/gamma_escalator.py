"""
Gamma Escalator Module

This module monitors changes in option gamma and provides gamma-based
exit signals for the options trading system.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class GammaEscalator:
    """
    Monitors option gamma and provides exit signals based on gamma acceleration,
    high absolute gamma values, and changing gamma profiles.
    
    Gamma-based exits are useful for:
    - Capturing profits during periods of gamma acceleration
    - Managing risk when gamma becomes too high
    - Identifying inflection points in the trade
    """
    
    def __init__(self, config, threshold_optimizer=None):
        """
        Initialize the gamma escalator.
        
        Args:
            config (dict): Configuration dictionary
            threshold_optimizer: Optional GreekThresholdOptimizer instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("gamma_escalator", {})
        self.threshold_optimizer = threshold_optimizer
        
        # Default thresholds (will be overridden by optimizer if available)
        self.high_gamma_threshold = self.config.get("high_gamma_threshold", 0.08)
        self.gamma_acceleration_threshold = self.config.get("gamma_acceleration_threshold", 0.01)
        self.gamma_change_threshold = self.config.get("gamma_change_threshold", 0.03)
        
        # Settings for gamma profile analysis
        self.gamma_peak_detection = self.config.get("gamma_peak_detection", True)
        self.max_gamma_value = self.config.get("max_gamma_value", 0.15)  # Expected maximum gamma value
        
        # Tracking gamma history for positions
        self.gamma_history = {}
        self.max_history_length = self.config.get("max_history_length", 20)
        
        # Track exit signals for logging/analysis
        self.exit_signals = []
        
        self.logger.info("GammaEscalator initialized")
    
    def update_thresholds(self, optimization_results=None):
        """
        Update gamma thresholds, either from optimizer or configuration.
        
        Args:
            optimization_results (dict, optional): Results from optimizer
        """
        if optimization_results is not None:
            # Update from optimizer results
            self.high_gamma_threshold = optimization_results.get("high_gamma", self.high_gamma_threshold)
            self.gamma_acceleration_threshold = optimization_results.get("gamma_acceleration", self.gamma_acceleration_threshold)
            self.gamma_change_threshold = optimization_results.get("gamma_change", self.gamma_change_threshold)
            
            self.logger.info(f"Updated gamma thresholds from optimizer: " +
                             f"High gamma: {self.high_gamma_threshold}, " +
                             f"Acceleration: {self.gamma_acceleration_threshold}, " +
                             f"Change: {self.gamma_change_threshold}")
            
        elif self.threshold_optimizer:
            # Get latest optimized thresholds
            optimal_thresholds = self.threshold_optimizer.get_optimal_thresholds("gamma")
            if optimal_thresholds:
                self.update_thresholds(optimal_thresholds)
        else:
            # Use configuration values
            self.high_gamma_threshold = self.config.get("high_gamma_threshold", 0.08)
            self.gamma_acceleration_threshold = self.config.get("gamma_acceleration_threshold", 0.01)
            self.gamma_change_threshold = self.config.get("gamma_change_threshold", 0.03)
    
    def update_gamma(self, trade_id, current_gamma, trade_data):
        """
        Update gamma history for a position.
        
        Args:
            trade_id (str): Trade identifier
            current_gamma (float): Current gamma value
            trade_data (dict): Current trade data
            
        Returns:
            list: Gamma history for the position
        """
        # Initialize history if not exists
        if trade_id not in self.gamma_history:
            self.gamma_history[trade_id] = []
        
        # Add timestamp and metadata
        gamma_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gamma": current_gamma,
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "option_price": trade_data.get("current_price", 0.0),
            "pnl_percent": trade_data.get("current_pnl_percent", 0.0),
            "delta": trade_data.get("greeks", {}).get("delta", 0.0)
        }
        
        # Add to history
        self.gamma_history[trade_id].append(gamma_entry)
        
        # Limit history length
        if len(self.gamma_history[trade_id]) > self.max_history_length:
            self.gamma_history[trade_id] = self.gamma_history[trade_id][-self.max_history_length:]
        
        return self.gamma_history[trade_id]
    
    def check_exit_signal(self, trade_id, trade_data):
        """
        Check if current gamma indicates an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Current trade data
            
        Returns:
            dict: Exit signal if detected, None otherwise
        """
        if trade_id not in self.gamma_history or not self.gamma_history[trade_id]:
            # No history yet, update and return
            current_gamma = trade_data.get("greeks", {}).get("gamma", 0.0)
            self.update_gamma(trade_id, current_gamma, trade_data)
            return None
        
        # Get current gamma
        current_gamma = trade_data.get("greeks", {}).get("gamma", 0.0)
        
        # Add gamma to history
        self.update_gamma(trade_id, current_gamma, trade_data)
        
        # Get gamma history
        gamma_history = self.gamma_history[trade_id]
        
        # Need at least 2 data points for change detection
        if len(gamma_history) < 2:
            return None
        
        # Check for high absolute gamma values
        signal = self._check_high_gamma(trade_id, current_gamma, trade_data)
        if signal:
            return signal
        
        # Check for significant gamma changes
        signal = self._check_gamma_change(trade_id, gamma_history, trade_data)
        if signal:
            return signal
        
        # Check for gamma acceleration/deceleration
        if len(gamma_history) >= 3:
            signal = self._check_gamma_acceleration(trade_id, gamma_history, trade_data)
            if signal:
                return signal
        
        # Check for gamma peak detection (when gamma starts decreasing after increasing)
        if self.gamma_peak_detection and len(gamma_history) >= 3:
            signal = self._check_gamma_peak(trade_id, gamma_history, trade_data)
            if signal:
                return signal
        
        # No exit signal
        return None
    
    def _check_high_gamma(self, trade_id, current_gamma, trade_data):
        """
        Check if gamma has reached a high threshold.
        
        Args:
            trade_id (str): Trade identifier
            current_gamma (float): Current gamma value
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if threshold reached, None otherwise
        """
        if current_gamma >= self.high_gamma_threshold:
            # Check if we're profitable
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            if pnl_percent > 0:
                # Higher confidence exit signal when in profit
                confidence_boost = min(0.3, pnl_percent / 100)
                return self._create_exit_signal(
                    trade_id,
                    "high_gamma_profitable",
                    f"High gamma ({current_gamma:.4f}) with profitable position ({pnl_percent:.2f}%)",
                    current_gamma,
                    trade_data,
                    confidence_boost=confidence_boost
                )
            else:
                # Lower confidence for unprofitable positions
                return self._create_exit_signal(
                    trade_id,
                    "high_gamma",
                    f"High gamma reached: {current_gamma:.4f} >= {self.high_gamma_threshold:.4f}",
                    current_gamma,
                    trade_data
                )
        
        return None
    
    def _check_gamma_change(self, trade_id, gamma_history, trade_data):
        """
        Check for significant changes in gamma.
        
        Args:
            trade_id (str): Trade identifier
            gamma_history (list): Gamma history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if significant change detected, None otherwise
        """
        # Need at least 2 data points
        if len(gamma_history) < 2:
            return None
        
        # Get recent gamma values
        recent_gamma = gamma_history[-1]["gamma"]
        prev_gamma = gamma_history[-2]["gamma"]
        
        # Calculate absolute change
        gamma_change = abs(recent_gamma - prev_gamma)
        
        # If change is significant
        if gamma_change >= self.gamma_change_threshold:
            # Check if gamma is decreasing after being high
            if prev_gamma >= self.high_gamma_threshold * 0.7 and recent_gamma < prev_gamma:
                return self._create_exit_signal(
                    trade_id,
                    "gamma_decreasing_from_high",
                    f"Gamma decreasing from high level: {prev_gamma:.4f} -> {recent_gamma:.4f} (Δ{gamma_change:.4f})",
                    recent_gamma,
                    trade_data,
                    confidence_boost=0.1
                )
            # Check if gamma is increasing rapidly
            elif recent_gamma > prev_gamma:
                # Adjust confidence based on proximity to max expected gamma
                proximity_to_max = recent_gamma / self.max_gamma_value
                confidence_adj = min(0.2, proximity_to_max * 0.2)
                
                return self._create_exit_signal(
                    trade_id,
                    "gamma_increasing_rapidly",
                    f"Gamma increasing rapidly: {prev_gamma:.4f} -> {recent_gamma:.4f} (Δ{gamma_change:.4f})",
                    recent_gamma,
                    trade_data,
                    confidence_boost=confidence_adj
                )
        
        # No significant change
        return None
    
    def _check_gamma_acceleration(self, trade_id, gamma_history, trade_data):
        """
        Check for acceleration/deceleration in gamma changes.
        
        Args:
            trade_id (str): Trade identifier
            gamma_history (list): Gamma history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if acceleration detected, None otherwise
        """
        # Need at least 3 data points
        if len(gamma_history) < 3:
            return None
        
        # Get gamma values
        gamma_3 = gamma_history[-3]["gamma"]
        gamma_2 = gamma_history[-2]["gamma"]
        gamma_1 = gamma_history[-1]["gamma"]
        
        # Calculate first differences
        diff_1 = gamma_2 - gamma_3
        diff_2 = gamma_1 - gamma_2
        
        # Calculate acceleration (second difference)
        acceleration = diff_2 - diff_1
        
        # Check for significant acceleration
        if abs(acceleration) >= self.gamma_acceleration_threshold:
            # Gamma is accelerating upward (increasing faster)
            if acceleration > 0:
                return self._create_exit_signal(
                    trade_id,
                    "gamma_accelerating",
                    f"Gamma accelerating upward: {acceleration:.6f}",
                    gamma_1,
                    trade_data
                )
            # Gamma is decelerating (slowing down or reversing)
            else:
                return self._create_exit_signal(
                    trade_id,
                    "gamma_decelerating",
                    f"Gamma decelerating/reversing: {acceleration:.6f}",
                    gamma_1,
                    trade_data
                )
        
        # No significant acceleration
        return None
    
    def _check_gamma_peak(self, trade_id, gamma_history, trade_data):
        """
        Check for gamma peak (when gamma starts decreasing after increasing).
        
        Args:
            trade_id (str): Trade identifier
            gamma_history (list): Gamma history for the position
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if gamma peak detected, None otherwise
        """
        # Need at least 3 data points
        if len(gamma_history) < 3:
            return None
        
        # Get last three gamma values
        gamma_3 = gamma_history[-3]["gamma"]
        gamma_2 = gamma_history[-2]["gamma"]
        gamma_1 = gamma_history[-1]["gamma"]
        
        # Check for peak pattern (up then down)
        if gamma_3 < gamma_2 and gamma_2 > gamma_1:
            # Only consider significant peaks
            if gamma_2 >= self.high_gamma_threshold * 0.5:
                # Calculate peak height
                peak_height = gamma_2 - max(gamma_1, gamma_3)
                
                # Only flag significant peaks
                if peak_height >= self.gamma_change_threshold * 0.7:
                    # Check if profitable
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    confidence_boost = min(0.2, pnl_percent / 100) if pnl_percent > 0 else 0
                    
                    return self._create_exit_signal(
                        trade_id,
                        "gamma_peak_detected",
                        f"Gamma peak detected: {gamma_3:.4f} -> {gamma_2:.4f} -> {gamma_1:.4f}",
                        gamma_1,
                        trade_data,
                        confidence_boost=confidence_boost
                    )
        
        # No peak detected
        return None
    
    def _create_exit_signal(self, trade_id, signal_type, description, gamma_value, trade_data, confidence_boost=0):
        """
        Create an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            signal_type (str): Type of exit signal
            description (str): Description of the signal
            gamma_value (float): Current gamma value
            trade_data (dict): Trade data
            confidence_boost (float): Additional confidence value
            
        Returns:
            dict: Exit signal
        """
        signal = {
            "trade_id": trade_id,
            "signal_type": signal_type,
            "greek": "gamma",
            "description": description,
            "gamma_value": gamma_value,
            "confidence": self._calculate_signal_confidence(signal_type, gamma_value, trade_data) + confidence_boost,
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
        self.logger.info(f"Gamma exit signal for {trade_id}: {description} (confidence: {signal['confidence']:.2f})")
        
        # Add to exit signals history
        self.exit_signals.append(signal)
        if len(self.exit_signals) > 100:
            self.exit_signals = self.exit_signals[-100:]
        
        return signal
    
    def _calculate_signal_confidence(self, signal_type, gamma_value, trade_data):
        """
        Calculate confidence level for the exit signal.
        
        Args:
            signal_type (str): Type of exit signal
            gamma_value (float): Current gamma value
            trade_data (dict): Trade data
            
        Returns:
            float: Confidence level (0-1)
        """
        base_confidence = 0.7  # Default confidence
        
        # Adjust based on signal type
        if signal_type == "high_gamma" or signal_type == "high_gamma_profitable":
            # More confident as gamma exceeds threshold
            base_confidence += min(0.2, (gamma_value - self.high_gamma_threshold) / self.high_gamma_threshold)
        elif signal_type == "gamma_decreasing_from_high":
            # More confident when decrease is from a higher level
            base_confidence += min(0.15, gamma_value / self.high_gamma_threshold * 0.15)
        elif signal_type == "gamma_increasing_rapidly":
            # More confident when gamma is already high
            base_confidence += min(0.2, gamma_value / self.high_gamma_threshold * 0.2)
        elif signal_type == "gamma_accelerating":
            # Confidence based on proximity to max expected gamma
            proximity = gamma_value / self.max_gamma_value
            base_confidence += min(0.2, proximity * 0.2)
        elif signal_type == "gamma_peak_detected":
            # More confident for higher peaks
            base_confidence += min(0.25, gamma_value / self.high_gamma_threshold * 0.25)
        
        # Adjust for profit/loss situation
        pnl_percent = trade_data.get("current_pnl_percent", 0.0)
        if pnl_percent > 20:
            # More confident in taking profits with high gamma
            base_confidence += min(0.15, pnl_percent / 150)  # Max +0.15 for 20%+ profit
        elif pnl_percent < -10:
            # Less confident in taking losses with high gamma
            base_confidence -= min(0.2, abs(pnl_percent) / 50)  # Max -0.2 for big losses
        
        # Adjust for time to expiration
        days_to_expiration = trade_data.get("days_to_expiration", 30)
        if days_to_expiration < 5:
            # More confident when close to expiration and gamma significant
            base_confidence += min(0.15, (5 - days_to_expiration) * 0.03)
        
        # Adjust for delta (closer to 0.5 means higher gamma potential)
        delta = trade_data.get("greeks", {}).get("delta", 0.0)
        delta_from_center = abs(abs(delta) - 0.5)  # 0 for delta of 0.5 or -0.5
        if delta_from_center < 0.2:
            # Closer to 0.5 delta means we're in the gamma sweet spot
            base_confidence += min(0.1, (0.2 - delta_from_center) / 2)
        
        return base_confidence
    
    def get_gamma_history(self, trade_id):
        """
        Get gamma history for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            list: Gamma history
        """
        return self.gamma_history.get(trade_id, [])
    
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
        Clear gamma history for a specific trade or all trades.
        
        Args:
            trade_id (str, optional): Trade to clear. If None, clear all.
        """
        if trade_id:
            if trade_id in self.gamma_history:
                del self.gamma_history[trade_id]
                self.logger.debug(f"Cleared gamma history for trade {trade_id}")
        else:
            self.gamma_history = {}
            self.logger.debug("Cleared all gamma history")
    
    def get_thresholds(self):
        """
        Get current gamma thresholds.
        
        Returns:
            dict: Current thresholds
        """
        return {
            "high_gamma_threshold": self.high_gamma_threshold,
            "gamma_acceleration_threshold": self.gamma_acceleration_threshold,
            "gamma_change_threshold": self.gamma_change_threshold
        }
    
    def get_gamma_statistics(self, trade_id=None):
        """
        Get gamma statistics for a specific trade or all trades.
        
        Args:
            trade_id (str, optional): Trade identifier. If None, get for all trades.
            
        Returns:
            dict: Gamma statistics
        """
        if trade_id:
            # Get statistics for a specific trade
            history = self.gamma_history.get(trade_id, [])
            if not history:
                return {"error": "No gamma history found for trade"}
            
            gamma_values = [entry["gamma"] for entry in history]
            return {
                "trade_id": trade_id,
                "current_gamma": gamma_values[-1] if gamma_values else None,
                "max_gamma": max(gamma_values) if gamma_values else None,
                "min_gamma": min(gamma_values) if gamma_values else None,
                "avg_gamma": sum(gamma_values) / len(gamma_values) if gamma_values else None,
                "std_gamma": np.std(gamma_values) if gamma_values else None,
                "data_points": len(gamma_values)
            }
        else:
            # Get statistics for all trades
            stats = {}
            for tid, history in self.gamma_history.items():
                if history:
                    gamma_values = [entry["gamma"] for entry in history]
                    stats[tid] = {
                        "current_gamma": gamma_values[-1] if gamma_values else None,
                        "max_gamma": max(gamma_values) if gamma_values else None,
                        "avg_gamma": sum(gamma_values) / len(gamma_values) if gamma_values else None,
                        "data_points": len(gamma_values)
                    }
            return stats
