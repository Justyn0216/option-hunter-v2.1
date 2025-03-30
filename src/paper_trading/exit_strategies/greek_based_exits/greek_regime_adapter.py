"""
Greek Regime Adapter Module

This module adapts Greek thresholds based on current market regimes,
volatility environment, and option characteristics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class GreekRegimeAdapter:
    """
    Dynamically adapts Greek exit thresholds based on current market regime,
    volatility environment, and underlying asset characteristics.
    
    Ensures that exit strategies are appropriately calibrated to current
    market conditions rather than using static thresholds.
    """
    
    def __init__(self, config, threshold_optimizer=None, market_regime_detector=None):
        """
        Initialize the Greek regime adapter.
        
        Args:
            config (dict): Configuration dictionary
            threshold_optimizer: Optional GreekThresholdOptimizer instance
            market_regime_detector: Optional market regime detector
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("greek_regime_adapter", {})
        self.threshold_optimizer = threshold_optimizer
        self.market_regime_detector = market_regime_detector
        
        # Base adjustment factors for different regimes
        self.regime_adjustments = self.config.get("regime_adjustments", {
            "market": {
                "bullish": {
                    "delta": {"call_high_delta_threshold": 1.1, "put_high_delta_threshold": 1.1},
                    "gamma": {"high_gamma_threshold": 0.9},
                    "theta": {"high_theta_ratio_threshold": 0.8},
                    "vega": {"iv_drop_threshold": 1.2}
                },
                "bearish": {
                    "delta": {"call_high_delta_threshold": 0.9, "put_high_delta_threshold": 0.9},
                    "gamma": {"high_gamma_threshold": 1.1},
                    "theta": {"high_theta_ratio_threshold": 1.2},
                    "vega": {"iv_drop_threshold": 0.8}
                },
                "sideways": {
                    "delta": {"delta_change_threshold": 1.2},
                    "gamma": {"gamma_change_threshold": 1.2},
                    "theta": {"theta_percentage_threshold": 0.9},
                    "vega": {"vega_exposure_threshold": 1.1}
                },
                "volatile": {
                    "delta": {"delta_change_threshold": 0.8},
                    "gamma": {"gamma_acceleration_threshold": 0.8},
                    "theta": {"theta_acceleration_threshold": 0.8},
                    "vega": {"iv_spike_threshold": 1.2}
                }
            },
            "volatility": {
                "low": {
                    "delta": {"delta_change_threshold": 1.2},
                    "gamma": {"high_gamma_threshold": 1.2},
                    "theta": {"theta_percentage_threshold": 1.2},
                    "vega": {"vega_exposure_threshold": 1.2, "iv_spike_threshold": 0.8}
                },
                "normal": {
                    # No adjustments for normal volatility (multiplier = 1.0)
                },
                "high": {
                    "delta": {"delta_change_threshold": 0.8},
                    "gamma": {"high_gamma_threshold": 0.8},
                    "theta": {"theta_percentage_threshold": 0.8},
                    "vega": {"vega_exposure_threshold": 0.8, "iv_spike_threshold": 1.2}
                },
                "extreme": {
                    "delta": {"delta_change_threshold": 0.6},
                    "gamma": {"high_gamma_threshold": 0.6},
                    "theta": {"theta_percentage_threshold": 0.6},
                    "vega": {"vega_exposure_threshold": 0.6, "iv_spike_threshold": 1.4}
                }
            }
        })
        
        # Adjustment factors based on days to expiration
        self.dte_adjustments = self.config.get("dte_adjustments", {
            "delta": {
                "0-7": {"call_high_delta_threshold": 0.9, "put_high_delta_threshold": 0.9},
                "8-21": {"call_high_delta_threshold": 1.0, "put_high_delta_threshold": 1.0},
                "22-45": {"call_high_delta_threshold": 1.1, "put_high_delta_threshold": 1.1},
                "46+": {"call_high_delta_threshold": 1.2, "put_high_delta_threshold": 1.2}
            },
            "gamma": {
                "0-7": {"high_gamma_threshold": 0.8},
                "8-21": {"high_gamma_threshold": 1.0},
                "22-45": {"high_gamma_threshold": 1.2},
                "46+": {"high_gamma_threshold": 1.4}
            },
            "theta": {
                "0-7": {"theta_percentage_threshold": 0.8, "high_theta_ratio_threshold": 0.8},
                "8-21": {"theta_percentage_threshold": 1.0, "high_theta_ratio_threshold": 1.0},
                "22-45": {"theta_percentage_threshold": 1.2, "high_theta_ratio_threshold": 1.2},
                "46+": {"theta_percentage_threshold": 1.5, "high_theta_ratio_threshold": 1.5}
            },
            "vega": {
                "0-7": {"vega_exposure_threshold": 1.3},
                "8-21": {"vega_exposure_threshold": 1.0},
                "22-45": {"vega_exposure_threshold": 0.9},
                "46+": {"vega_exposure_threshold": 0.8}
            }
        })
        
        # Adaptive learning rate
        self.learning_rate = self.config.get("learning_rate", 0.05)
        
        # Directory for storing adaptation data
        self.data_dir = "data/exit_strategies/regime_adapter"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Recent performance tracking for feedback
        self.recent_performance = {
            "delta": {"success_count": 0, "failure_count": 0},
            "gamma": {"success_count": 0, "failure_count": 0},
            "theta": {"success_count": 0, "failure_count": 0},
            "vega": {"success_count": 0, "failure_count": 0}
        }
        
        # Load adjustment history if available
        self._load_adjustment_history()
        
        self.logger.info("GreekRegimeAdapter initialized")
    
    def _load_adjustment_history(self):
        """Load adjustment history from disk if available."""
        try:
            adjustment_file = f"{self.data_dir}/regime_adjustments.json"
            if os.path.exists(adjustment_file):
                with open(adjustment_file, "r") as f:
                    loaded_adjustments = json.load(f)
                
                # Update adjustments if loaded successfully
                if loaded_adjustments and "market" in loaded_adjustments and "volatility" in loaded_adjustments:
                    self.regime_adjustments = loaded_adjustments
                    self.logger.info("Loaded regime adjustment history")
                    
            # Also load performance history
            performance_file = f"{self.data_dir}/recent_performance.json"
            if os.path.exists(performance_file):
                with open(performance_file, "r") as f:
                    self.recent_performance = json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading adjustment history: {str(e)}")
    
    def _save_adjustment_history(self):
        """Save adjustment history to disk."""
        try:
            adjustment_file = f"{self.data_dir}/regime_adjustments.json"
            with open(adjustment_file, "w") as f:
                json.dump(self.regime_adjustments, f, indent=2)
            
            # Also save performance history
            performance_file = f"{self.data_dir}/recent_performance.json"
            with open(performance_file, "w") as f:
                json.dump(self.recent_performance, f, indent=2)
                
            self.logger.debug("Saved regime adjustment history")
            
        except Exception as e:
            self.logger.error(f"Error saving adjustment history: {str(e)}")
    
    def adapt_thresholds(self, base_thresholds, greek_type, market_conditions, option_data):
        """
        Adapt base thresholds based on current market conditions and option data.
        
        Args:
            base_thresholds (dict): Base thresholds from optimizer
            greek_type (str): Greek strategy type ('delta', 'gamma', 'theta', 'vega')
            market_conditions (dict): Current market conditions
            option_data (dict): Option characteristics
            
        Returns:
            dict: Adapted thresholds
        """
        if not base_thresholds:
            self.logger.warning(f"No base thresholds provided for {greek_type}")
            return {}
        
        # Extract market regime and volatility regime
        market_regime = market_conditions.get("market_regime", "normal")
        volatility_regime = market_conditions.get("volatility_regime", "normal")
        
        # Get days to expiration category
        days_to_expiration = option_data.get("days_to_expiration", 30)
        dte_category = self._get_dte_category(days_to_expiration)
        
        # Start with base thresholds
        adapted_thresholds = base_thresholds.copy()
        
        # Apply market regime adjustments
        self._apply_regime_adjustments(adapted_thresholds, greek_type, "market", market_regime)
        
        # Apply volatility regime adjustments
        self._apply_regime_adjustments(adapted_thresholds, greek_type, "volatility", volatility_regime)
        
        # Apply days to expiration adjustments
        self._apply_dte_adjustments(adapted_thresholds, greek_type, dte_category)
        
        # Apply any specialized adjustments based on option characteristics
        self._apply_option_specific_adjustments(adapted_thresholds, greek_type, option_data)
        
        self.logger.debug(f"Adapted {greek_type} thresholds for {market_regime}/{volatility_regime} regime and {dte_category} DTE")
        return adapted_thresholds
    
    def _apply_regime_adjustments(self, thresholds, greek_type, regime_type, regime_value):
        """
        Apply adjustments for a specific regime type.
        
        Args:
            thresholds (dict): Thresholds to adjust
            greek_type (str): Greek strategy type
            regime_type (str): 'market' or 'volatility'
            regime_value (str): Specific regime value
        """
        if regime_type not in self.regime_adjustments or regime_value not in self.regime_adjustments[regime_type]:
            return
        
        regime_data = self.regime_adjustments[regime_type][regime_value]
        if greek_type not in regime_data:
            return
        
        # Apply adjustments for each parameter
        for param, multiplier in regime_data[greek_type].items():
            if param in thresholds:
                thresholds[param] *= multiplier
    
    def _apply_dte_adjustments(self, thresholds, greek_type, dte_category):
        """
        Apply adjustments based on days to expiration.
        
        Args:
            thresholds (dict): Thresholds to adjust
            greek_type (str): Greek strategy type
            dte_category (str): DTE category ('0-7', '8-21', '22-45', '46+')
        """
        if greek_type not in self.dte_adjustments or dte_category not in self.dte_adjustments[greek_type]:
            return
        
        # Apply adjustments for each parameter
        for param, multiplier in self.dte_adjustments[greek_type][dte_category].items():
            if param in thresholds:
                thresholds[param] *= multiplier
    
    def _apply_option_specific_adjustments(self, thresholds, greek_type, option_data):
        """
        Apply option-specific adjustments based on characteristics.
        
        Args:
            thresholds (dict): Thresholds to adjust
            greek_type (str): Greek strategy type
            option_data (dict): Option characteristics
        """
        # Extract option characteristics
        option_type = option_data.get("option_type", "call").lower()
        moneyness = option_data.get("moneyness", 0)  # ATM = 0, ITM > 0, OTM < 0
        implied_volatility = option_data.get("implied_volatility", 0)
        
        # Apply adjustments based on option type and moneyness
        if greek_type == "delta":
            # For calls, adjust based on moneyness
            if option_type == "call":
                if moneyness > 0.05:  # ITM calls
                    if "call_high_delta_threshold" in thresholds:
                        thresholds["call_high_delta_threshold"] *= 0.95  # Lower threshold (easier trigger)
                elif moneyness < -0.05:  # OTM calls
                    if "call_low_delta_threshold" in thresholds:
                        thresholds["call_low_delta_threshold"] *= 1.2  # Raise threshold (harder trigger)
            
            # For puts, adjust based on moneyness
            elif option_type == "put":
                if moneyness < -0.05:  # ITM puts (strike below price)
                    if "put_high_delta_threshold" in thresholds:
                        thresholds["put_high_delta_threshold"] *= 0.95  # Make more negative (easier trigger)
                elif moneyness > 0.05:  # OTM puts
                    if "put_low_delta_threshold" in thresholds:
                        thresholds["put_low_delta_threshold"] *= 1.2  # Raise threshold (harder trigger)
        
        elif greek_type == "gamma":
            # Gamma is highest near the money, adjust based on moneyness
            if abs(moneyness) < 0.02:  # Very close to ATM
                if "high_gamma_threshold" in thresholds:
                    thresholds["high_gamma_threshold"] *= 0.9  # Lower threshold for ATM options
            elif abs(moneyness) > 0.1:  # Far ITM or OTM
                if "high_gamma_threshold" in thresholds:
                    thresholds["high_gamma_threshold"] *= 1.3  # Higher threshold for far ITM/OTM
        
        elif greek_type == "theta":
            # Theta affects OTM options more aggressively near expiration
            if abs(moneyness) > 0.05 and option_data.get("days_to_expiration", 30) < 14:
                if "theta_percentage_threshold" in thresholds:
                    thresholds["theta_percentage_threshold"] *= 0.9  # Lower threshold for OTM near expiry
        
        elif greek_type == "vega":
            # Vega is highest for ATM options with longer expiration
            if abs(moneyness) < 0.05 and option_data.get("days_to_expiration", 30) > 30:
                if "vega_exposure_threshold" in thresholds:
                    thresholds["vega_exposure_threshold"] *= 0.9  # Lower threshold for ATM with long expiry
            
            # Adjust IV thresholds based on current IV level
            if implied_volatility > 0:
                historical_vol = option_data.get("historical_volatility", implied_volatility * 0.8)
                if historical_vol > 0:
                    iv_ratio = implied_volatility / historical_vol
                    
                    # If IV is already much higher than historical, make IV spike threshold higher
                    if iv_ratio > 1.5 and "iv_spike_threshold" in thresholds:
                        thresholds["iv_spike_threshold"] *= 1.2  # Higher threshold when IV already elevated
                    
                    # If IV is close to historical, make IV drop threshold more sensitive
                    if 0.9 < iv_ratio < 1.1 and "iv_drop_threshold" in thresholds:
                        thresholds["iv_drop_threshold"] *= 0.9  # Lower threshold when IV near historical
    
    def _get_dte_category(self, days_to_expiration):
        """
        Get days to expiration category.
        
        Args:
            days_to_expiration (int): Number of days to expiration
            
        Returns:
            str: DTE category
        """
        if days_to_expiration <= 7:
            return "0-7"
        elif days_to_expiration <= 21:
            return "8-21"
        elif days_to_expiration <= 45:
            return "22-45"
        else:
            return "46+"
    
    def record_exit_performance(self, greek_type, exit_result, market_conditions, option_data):
        """
        Record performance of an exit signal to adapt future thresholds.
        
        Args:
            greek_type (str): Greek strategy type
            exit_result (dict): Exit result including success/failure
            market_conditions (dict): Market conditions at exit
            option_data (dict): Option data at exit
        """
        # Extract market regime and volatility regime
        market_regime = market_conditions.get("market_regime", "normal")
        volatility_regime = market_conditions.get("volatility_regime", "normal")
        
        # Track success/failure for this Greek
        was_successful = exit_result.get("success", False)
        if was_successful:
            self.recent_performance[greek_type]["success_count"] += 1
        else:
            self.recent_performance[greek_type]["failure_count"] += 1
        
        # Adapt thresholds based on performance feedback
        self._adapt_thresholds_from_feedback(
            greek_type, was_successful, 
            market_regime, volatility_regime,
            option_data, exit_result
        )
        
        # Save updated adjustments
        self._save_adjustment_history()
    
    def _adapt_thresholds_from_feedback(self, greek_type, successful, market_regime, volatility_regime, option_data, exit_result):
        """
        Adapt threshold adjustments based on exit performance.
        
        Args:
            greek_type (str): Greek strategy type
            successful (bool): Whether exit was successful
            market_regime (str): Market regime at exit
            volatility_regime (str): Volatility regime at exit
            option_data (dict): Option data at exit
            exit_result (dict): Exit result details
        """
        # Get the adjustment direction based on success/failure
        direction = 1 if successful else -1
        
        # Get days to expiration category
        days_to_expiration = option_data.get("days_to_expiration", 30)
        dte_category = self._get_dte_category(days_to_expiration)
        
        # Get signal type and threshold that triggered the exit
        signal_type = exit_result.get("signal_type", "")
        threshold_param = self._get_threshold_param_from_signal(greek_type, signal_type)
        
        if not threshold_param:
            return
        
        # Apply learning to market regime adjustments
        if market_regime in self.regime_adjustments["market"]:
            market_adj = self.regime_adjustments["market"][market_regime].get(greek_type, {})
            if threshold_param in market_adj:
                # Adjust the multiplier (increase if successful, decrease if not)
                current = market_adj[threshold_param]
                # Calculate new value bounded between 0.5 and 1.5
                new_value = current + (direction * self.learning_rate)
                new_value = max(0.5, min(new_value, 1.5))
                
                # Update the adjustment
                market_adj[threshold_param] = new_value
                self.regime_adjustments["market"][market_regime][greek_type] = market_adj
        
        # Apply learning to volatility regime adjustments
        if volatility_regime in self.regime_adjustments["volatility"]:
            vol_adj = self.regime_adjustments["volatility"][volatility_regime].get(greek_type, {})
            if threshold_param in vol_adj:
                # Adjust the multiplier (increase if successful, decrease if not)
                current = vol_adj[threshold_param]
                # Calculate new value bounded between 0.5 and 1.5
                new_value = current + (direction * self.learning_rate)
                new_value = max(0.5, min(new_value, 1.5))
                
                # Update the adjustment
                vol_adj[threshold_param] = new_value
                self.regime_adjustments["volatility"][volatility_regime][greek_type] = vol_adj
        
        # Apply learning to DTE adjustments
        if greek_type in self.dte_adjustments and dte_category in self.dte_adjustments[greek_type]:
            dte_adj = self.dte_adjustments[greek_type][dte_category]
            if threshold_param in dte_adj:
                # Adjust the multiplier (increase if successful, decrease if not)
                current = dte_adj[threshold_param]
                # Calculate new value bounded between 0.5 and 1.5
                new_value = current + (direction * self.learning_rate)
                new_value = max(0.5, min(new_value, 1.5))
                
                # Update the adjustment
                dte_adj[threshold_param] = new_value
                self.dte_adjustments[greek_type][dte_category] = dte_adj
    
    def _get_threshold_param_from_signal(self, greek_type, signal_type):
        """
        Map signal type to corresponding threshold parameter.
        
        Args:
            greek_type (str): Greek strategy type
            signal_type (str): Signal type that triggered exit
            
        Returns:
            str: Corresponding threshold parameter
        """
        # Delta thresholds
        if greek_type == "delta":
            if signal_type in ["delta_high_threshold", "delta_high"]:
                return "call_high_delta_threshold" if "call" in signal_type else "put_high_delta_threshold"
            elif signal_type in ["delta_low_threshold", "delta_low"]:
                return "call_low_delta_threshold" if "call" in signal_type else "put_low_delta_threshold"
            elif "delta_change" in signal_type or "delta_decrease" in signal_type:
                return "delta_change_threshold"
            elif "delta_acceleration" in signal_type or "delta_deceleration" in signal_type:
                return "delta_acceleration_threshold"
        
        # Gamma thresholds
        elif greek_type == "gamma":
            if "high_gamma" in signal_type:
                return "high_gamma_threshold"
            elif "gamma_change" in signal_type or "gamma_increasing" in signal_type or "gamma_decreasing" in signal_type:
                return "gamma_change_threshold"
            elif "gamma_acceleration" in signal_type or "gamma_deceleration" in signal_type:
                return "gamma_acceleration_threshold"
        
        # Theta thresholds
        elif greek_type == "theta":
            if "high_theta_ratio" in signal_type:
                return "high_theta_ratio_threshold"
            elif "theta_percentage" in signal_type or "near_expiry_theta" in signal_type:
                return "theta_percentage_threshold"
            elif "theta_acceleration" in signal_type or "accelerating_theta" in signal_type:
                return "theta_acceleration_threshold"
        
        # Vega thresholds
        elif greek_type == "vega":
            if "iv_drop" in signal_type:
                return "iv_drop_threshold"
            elif "iv_spike" in signal_type:
                return "iv_spike_threshold"
            elif "vega_exposure" in signal_type or "high_vega" in signal_type:
                return "vega_exposure_threshold"
        
        return None
    
    def get_current_adjustments(self, greek_type=None):
        """
        Get current adjustment factors.
        
        Args:
            greek_type (str, optional): Greek strategy type to filter by
            
        Returns:
            dict: Current adjustment factors
        """
        if greek_type:
            # Return adjustments for specific Greek
            adjustments = {
                "market_regime": {regime: data.get(greek_type, {}) 
                              for regime, data in self.regime_adjustments["market"].items() 
                              if greek_type in data},
                "volatility_regime": {regime: data.get(greek_type, {}) 
                                  for regime, data in self.regime_adjustments["volatility"].items() 
                                  if greek_type in data},
                "dte": self.dte_adjustments.get(greek_type, {})
            }
            return adjustments
        else:
            # Return all adjustments
            return {
                "market_regime": self.regime_adjustments["market"],
                "volatility_regime": self.regime_adjustments["volatility"],
                "dte": self.dte_adjustments
            }
    
    def get_performance_stats(self, greek_type=None):
        """
        Get performance statistics.
        
        Args:
            greek_type (str, optional): Greek strategy type to filter by
            
        Returns:
            dict: Performance statistics
        """
        stats = {}
        
        if greek_type:
            # Stats for specific Greek
            greek_stats = self.recent_performance.get(greek_type, {})
            success_count = greek_stats.get("success_count", 0)
            failure_count = greek_stats.get("failure_count", 0)
            total = success_count + failure_count
            
            stats[greek_type] = {
                "success_count": success_count,
                "failure_count": failure_count,
                "total_exits": total,
                "success_rate": (success_count / total * 100) if total > 0 else 0
            }
        else:
            # Stats for all Greeks
            for greek, perf in self.recent_performance.items():
                success_count = perf.get("success_count", 0)
                failure_count = perf.get("failure_count", 0)
                total = success_count + failure_count
                
                stats[greek] = {
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "total_exits": total,
                    "success_rate": (success_count / total * 100) if total > 0 else 0
                }
        
        return stats
    
    def reset_performance_tracking(self, greek_type=None):
        """
        Reset performance tracking statistics.
        
        Args:
            greek_type (str, optional): Greek strategy type to reset, or all if None
        """
        if greek_type:
            if greek_type in self.recent_performance:
                self.recent_performance[greek_type] = {"success_count": 0, "failure_count": 0}
                self.logger.info(f"Reset performance tracking for {greek_type}")
        else:
            for greek in self.recent_performance:
                self.recent_performance[greek] = {"success_count": 0, "failure_count": 0}
            self.logger.info("Reset all performance tracking")
            
        # Save updated stats
        self._save_adjustment_history()
