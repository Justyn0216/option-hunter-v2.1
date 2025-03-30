"""
Vega Volatility Exit Module

This module monitors option vega and implied volatility changes to provide
volatility-based exit signals for the options trading system.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class VegaVolatilityExit:
    """
    Monitors option vega and implied volatility changes to provide exit signals.
    
    Vega/IV-based exits are useful for:
    - Capturing profits from volatility expansion
    - Exiting positions after volatility contraction
    - Timing exits based on volatility regime changes
    - Managing vega exposure in the portfolio
    """
    
    def __init__(self, config, threshold_optimizer=None):
        """
        Initialize the vega volatility exit monitor.
        
        Args:
            config (dict): Configuration dictionary
            threshold_optimizer: Optional GreekThresholdOptimizer instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("vega_volatility_exit", {})
        self.threshold_optimizer = threshold_optimizer
        
        # Default thresholds (will be overridden by optimizer if available)
        self.iv_drop_threshold = self.config.get("iv_drop_threshold", 15.0)  # % IV drop
        self.iv_spike_threshold = self.config.get("iv_spike_threshold", 25.0)  # % IV spike
        self.vega_exposure_threshold = self.config.get("vega_exposure_threshold", 0.5)  # Vega/price ratio
        self.sustained_iv_contraction_days = self.config.get("sustained_iv_contraction_days", 3)
        
        # Volatility reversion expectations
        self.iv_reversion_trigger = self.config.get("iv_reversion_trigger", 30.0)  # % above historical IV
        self.iv_reversion_exit_pct = self.config.get("iv_reversion_exit_pct", 50.0)  # Exit after capturing % of move
        
        # Tracking vega and IV history for positions
        self.vega_history = {}
        self.iv_history = {}
        self.max_history_length = self.config.get("max_history_length", 30)
        
        # Track exit signals for logging/analysis
        self.exit_signals = []
        
        self.logger.info("VegaVolatilityExit initialized")
    
    def update_thresholds(self, optimization_results=None):
        """
        Update volatility thresholds, either from optimizer or configuration.
        
        Args:
            optimization_results (dict, optional): Results from optimizer
        """
        if optimization_results is not None:
            # Update from optimizer results
            self.iv_drop_threshold = optimization_results.get("iv_drop", self.iv_drop_threshold)
            self.iv_spike_threshold = optimization_results.get("iv_spike", self.iv_spike_threshold)
            self.vega_exposure_threshold = optimization_results.get("vega_exposure", self.vega_exposure_threshold)
            
            self.logger.info(f"Updated volatility thresholds from optimizer: " +
                             f"IV drop: {self.iv_drop_threshold}%, " +
                             f"IV spike: {self.iv_spike_threshold}%, " +
                             f"Vega exposure: {self.vega_exposure_threshold}")
            
        elif self.threshold_optimizer:
            # Get latest optimized thresholds
            optimal_thresholds = self.threshold_optimizer.get_optimal_thresholds("vega")
            if optimal_thresholds:
                self.update_thresholds(optimal_thresholds)
        else:
            # Use configuration values
            self.iv_drop_threshold = self.config.get("iv_drop_threshold", 15.0)
            self.iv_spike_threshold = self.config.get("iv_spike_threshold", 25.0)
            self.vega_exposure_threshold = self.config.get("vega_exposure_threshold", 0.5)
    
    def update_vega_iv(self, trade_id, vega, implied_volatility, trade_data):
        """
        Update vega and IV history for a position.
        
        Args:
            trade_id (str): Trade identifier
            vega (float): Current vega value
            implied_volatility (float): Current implied volatility
            trade_data (dict): Current trade data
            
        Returns:
            tuple: (vega_history, iv_history)
        """
        # Initialize histories if not exist
        if trade_id not in self.vega_history:
            self.vega_history[trade_id] = []
        if trade_id not in self.iv_history:
            self.iv_history[trade_id] = []
        
        # Record vega data
        vega_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vega": vega,
            "option_price": trade_data.get("current_price", 0.0),
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "vega_ratio": vega / trade_data.get("current_price", 0.01) if trade_data.get("current_price", 0) > 0 else 0.0,
            "pnl_percent": trade_data.get("current_pnl_percent", 0.0)
        }
        
        # Record IV data
        iv_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iv": implied_volatility,
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "days_to_expiration": trade_data.get("days_to_expiration", 0),
            "historical_vol": trade_data.get("historical_volatility", implied_volatility * 0.8)  # Estimate if not provided
        }
        
        # Add to history
        self.vega_history[trade_id].append(vega_entry)
        self.iv_history[trade_id].append(iv_entry)
        
        # Limit history length
        if len(self.vega_history[trade_id]) > self.max_history_length:
            self.vega_history[trade_id] = self.vega_history[trade_id][-self.max_history_length:]
        if len(self.iv_history[trade_id]) > self.max_history_length:
            self.iv_history[trade_id] = self.iv_history[trade_id][-self.max_history_length:]
        
        return self.vega_history[trade_id], self.iv_history[trade_id]
    
    def check_exit_signal(self, trade_id, trade_data):
        """
        Check if current vega/IV indicates an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Current trade data
            
        Returns:
            dict: Exit signal if detected, None otherwise
        """
        # Get vega and IV from trade data
        vega = trade_data.get("greeks", {}).get("vega", 0.0)
        iv = trade_data.get("implied_volatility", 0.0)
        option_type = trade_data.get("option_type", "").lower()
        
        # If no IV data, can't generate signals
        if iv <= 0:
            return None
        
        # Update histories
        self.update_vega_iv(trade_id, vega, iv, trade_data)
        
        # Check for various exit signals
        
        # 1. IV Drop Exit
        signal = self._check_iv_drop(trade_id, trade_data)
        if signal:
            return signal
        
        # 2. IV Spike Profit Taking
        signal = self._check_iv_spike_profit(trade_id, trade_data)
        if signal:
            return signal
        
        # 3. Vega Exposure Exit
        signal = self._check_vega_exposure(trade_id, trade_data)
        if signal:
            return signal
        
        # 4. IV Reversion Exit
        signal = self._check_iv_reversion(trade_id, trade_data)
        if signal:
            return signal
        
        # No exit signal
        return None
    
    def _check_iv_drop(self, trade_id, trade_data):
        """
        Check for significant IV drop that might warrant an exit.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if IV drop detected, None otherwise
        """
        if trade_id not in self.iv_history or len(self.iv_history[trade_id]) < 2:
            return None
        
        # Get IV history
        iv_history = self.iv_history[trade_id]
        
        # Get initial and current IV
        initial_iv = iv_history[0]["iv"]
        current_iv = iv_history[-1]["iv"]
        
        # If long options (especially calls), IV drop is bad
        if trade_data.get("position_type", "long") == "long":
            # Calculate percentage drop
            if initial_iv > 0:
                iv_change_pct = ((current_iv - initial_iv) / initial_iv) * 100
                
                # If significant drop
                if iv_change_pct <= -self.iv_drop_threshold:
                    # Factor in trade profitability in decision
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    
                    # For profitable trades, preserve profits
                    if pnl_percent > 0:
                        return self._create_exit_signal(
                            trade_id,
                            "iv_drop_with_profit",
                            f"IV dropped {abs(iv_change_pct):.1f}% with {pnl_percent:.1f}% profit",
                            iv_history[-1]["iv"],
                            trade_data,
                            confidence_boost=min(0.2, pnl_percent / 100)
                        )
                    # For unprofitable trades, cut losses if IV drops
                    else:
                        # Check if IV is still dropping (comparing recent values)
                        if len(iv_history) >= 3 and iv_history[-1]["iv"] < iv_history[-2]["iv"] < iv_history[-3]["iv"]:
                            return self._create_exit_signal(
                                trade_id,
                                "continued_iv_drop_with_loss",
                                f"Continued IV drop to {current_iv:.1f}% with {pnl_percent:.1f}% loss",
                                iv_history[-1]["iv"],
                                trade_data
                            )
        
        # For short options, IV drop is good but may indicate profit-taking time
        elif trade_data.get("position_type", "long") == "short":
            # Calculate percentage drop
            if initial_iv > 0:
                iv_change_pct = ((current_iv - initial_iv) / initial_iv) * 100
                
                # If significant drop and profitable, consider exiting to lock in gains
                if iv_change_pct <= -self.iv_drop_threshold:
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    if pnl_percent > 30:  # If good profit already
                        return self._create_exit_signal(
                            trade_id,
                            "iv_drop_short_profit_taking",
                            f"IV contraction objective achieved: {abs(iv_change_pct):.1f}% drop with {pnl_percent:.1f}% profit",
                            iv_history[-1]["iv"],
                            trade_data,
                            confidence_boost=0.1
                        )
        
        return None
    
    def _check_iv_spike_profit(self, trade_id, trade_data):
        """
        Check for IV spike profit-taking opportunity.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if IV spike detected, None otherwise
        """
        if trade_id not in self.iv_history or len(self.iv_history[trade_id]) < 2:
            return None
        
        # Get IV history
        iv_history = self.iv_history[trade_id]
        
        # Get initial and current IV
        initial_iv = iv_history[0]["iv"]
        current_iv = iv_history[-1]["iv"]
        
        # If long options, IV spike is good for profit-taking
        if trade_data.get("position_type", "long") == "long":
            # Calculate percentage spike
            if initial_iv > 0:
                iv_change_pct = ((current_iv - initial_iv) / initial_iv) * 100
                
                # If significant spike and profitable
                if iv_change_pct >= self.iv_spike_threshold:
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    
                    if pnl_percent > 20:  # If good profit
                        # Higher confidence the higher the profit
                        confidence_boost = min(0.3, pnl_percent / 100)
                        
                        return self._create_exit_signal(
                            trade_id,
                            "iv_spike_profit_taking",
                            f"IV spike profit-taking: {iv_change_pct:.1f}% spike with {pnl_percent:.1f}% profit",
                            iv_history[-1]["iv"],
                            trade_data,
                            confidence_boost=confidence_boost
                        )
        
        # For short options, IV spike is bad
        elif trade_data.get("position_type", "long") == "short":
            # Calculate percentage spike
            if initial_iv > 0:
                iv_change_pct = ((current_iv - initial_iv) / initial_iv) * 100
                
                # If significant spike, consider exiting to prevent larger losses
                if iv_change_pct >= self.iv_spike_threshold / 2:  # Lower threshold for shorts
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    
                    return self._create_exit_signal(
                        trade_id,
                        "iv_spike_short_defense",
                        f"Defensive exit due to {iv_change_pct:.1f}% IV spike with {pnl_percent:.1f}% P&L",
                        iv_history[-1]["iv"],
                        trade_data
                    )
        
        return None
    
    def _check_vega_exposure(self, trade_id, trade_data):
        """
        Check for excessive vega exposure relative to option price.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if vega exposure is too high, None otherwise
        """
        if trade_id not in self.vega_history:
            return None
        
        # Get vega history
        vega_history = self.vega_history[trade_id]
        
        if not vega_history:
            return None
        
        # Get current vega and option price
        current_vega = vega_history[-1]["vega"]
        option_price = vega_history[-1]["option_price"]
        
        if option_price <= 0:
            return None
        
        # Calculate vega exposure ratio
        vega_ratio = current_vega / option_price
        
        # If high vega exposure
        if vega_ratio > self.vega_exposure_threshold:
            # Check if profitable
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            days_to_expiration = trade_data.get("days_to_expiration", 30)
            
            # For longer-dated options, higher vega is expected
            adjusted_threshold = self.vega_exposure_threshold * (1 + days_to_expiration / 60)
            
            if vega_ratio > adjusted_threshold:
                # Higher confidence for profitable trades or very high ratios
                confidence_boost = 0.0
                if pnl_percent > 20:
                    confidence_boost = min(0.2, pnl_percent / 100)
                elif vega_ratio > adjusted_threshold * 1.5:
                    confidence_boost = 0.15
                
                return self._create_exit_signal(
                    trade_id,
                    "high_vega_exposure",
                    f"High vega exposure: {vega_ratio:.4f} ratio with {pnl_percent:.1f}% P&L",
                    current_vega,
                    trade_data,
                    confidence_boost=confidence_boost
                )
        
        return None
    
    def _check_iv_reversion(self, trade_id, trade_data):
        """
        Check for IV reversion opportunity (IV returning to normal after spike).
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Trade data
            
        Returns:
            dict: Exit signal if IV reversion detected, None otherwise
        """
        if trade_id not in self.iv_history or len(self.iv_history[trade_id]) < 3:
            return None
        
        # Get IV history
        iv_history = self.iv_history[trade_id]
        
        # Get historical volatility if available
        historical_vol = iv_history[-1]["historical_vol"]
        
        if historical_vol <= 0:
            return None
        
        # Calculate IV premium over historical
        current_iv = iv_history[-1]["iv"]
        iv_premium_pct = ((current_iv - historical_vol) / historical_vol) * 100
        
        # Find maximum IV in history
        max_iv = max(entry["iv"] for entry in iv_history)
        max_iv_premium = ((max_iv - historical_vol) / historical_vol) * 100
        
        # If we had a significant IV premium that's now decreasing
        if max_iv_premium > self.iv_reversion_trigger and current_iv < max_iv:
            # Calculate how much of the spike we've captured
            if max_iv > historical_vol:
                reversion_pct = (max_iv - current_iv) / (max_iv - historical_vol) * 100
                
                # If we've captured significant portion of the move
                if reversion_pct > self.iv_reversion_exit_pct:
                    # Check profitability
                    pnl_percent = trade_data.get("current_pnl_percent", 0.0)
                    
                    if pnl_percent > 0:
                        return self._create_exit_signal(
                            trade_id,
                            "iv_reversion_profit_taking",
                            f"IV reverting to normal: captured {reversion_pct:.1f}% of spike with {pnl_percent:.1f}% profit",
                            current_iv,
                            trade_data,
                            confidence_boost=min(0.2, pnl_percent / 100)
                        )
        
        return None
    
    def _create_exit_signal(self, trade_id, signal_type, description, iv_value, trade_data, confidence_boost=0):
        """
        Create an exit signal.
        
        Args:
            trade_id (str): Trade identifier
            signal_type (str): Type of exit signal
            description (str): Description of the signal
            iv_value (float): Current implied volatility
            trade_data (dict): Trade data
            confidence_boost (float): Additional confidence value
            
        Returns:
            dict: Exit signal
        """
        signal = {
            "trade_id": trade_id,
            "signal_type": signal_type,
            "greek": "vega",
            "description": description,
            "iv_value": iv_value,
            "vega_value": trade_data.get("greeks", {}).get("vega", 0.0),
            "confidence": self._calculate_signal_confidence(signal_type, iv_value, trade_data) + confidence_boost,
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
        self.logger.info(f"Vega/IV exit signal for {trade_id}: {description} (confidence: {signal['confidence']:.2f})")
        
        # Add to exit signals history
        self.exit_signals.append(signal)
        if len(self.exit_signals) > 100:
            self.exit_signals = self.exit_signals[-100:]
        
        return signal
    
    def _calculate_signal_confidence(self, signal_type, iv_value, trade_data):
        """
        Calculate confidence level for the exit signal.
        
        Args:
            signal_type (str): Type of exit signal
            iv_value (float): Current implied volatility
            trade_data (dict): Trade data
            
        Returns:
            float: Confidence level (0-1)
        """
        base_confidence = 0.7  # Default confidence
        
        # Get initial IV from history if available
        initial_iv = None
        if trade_data.get("trade_id") in self.iv_history and self.iv_history[trade_data["trade_id"]]:
            initial_iv = self.iv_history[trade_data["trade_id"]][0]["iv"]
        
        # Adjust based on signal type
        if signal_type == "iv_drop_with_profit":
            # More confident with larger drops and profits
            if initial_iv:
                iv_change_pct = ((iv_value - initial_iv) / initial_iv) * 100
                base_confidence += min(0.15, abs(iv_change_pct) / 100)
            
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            base_confidence += min(0.15, pnl_percent / 100)
            
        elif signal_type == "continued_iv_drop_with_loss":
            # Less confident with larger losses
            pnl_percent = trade_data.get("current_pnl_percent", 0.0)
            if pnl_percent < 0:
                base_confidence -= min(0.2, abs(pnl_percent) / 100)
            
        elif signal_type == "iv_spike_profit_taking":
            # More confident with larger spikes
            if initial_iv:
                iv_change_pct = ((iv_value - initial_iv) / initial_iv) * 100
                base_confidence += min(0.2, iv_change_pct / 100)
            
        elif signal_type == "high_vega_exposure":
            # Confidence depends on days to expiration
            days_to_expiration = trade_data.get("days_to_expiration", 30)
            if days_to_expiration < 14:
                base_confidence += 0.1  # Higher confidence near expiration
            
        elif signal_type == "iv_reversion_profit_taking":
            # More confident when IV approaches historical norms
            historical_vol = 0
            if trade_data.get("trade_id") in self.iv_history and self.iv_history[trade_data["trade_id"]]:
                historical_vol = self.iv_history[trade_data["trade_id"]][-1]["historical_vol"]
            
            if historical_vol > 0:
                iv_premium = (iv_value - historical_vol) / historical_vol
                base_confidence += min(0.2, (1 - iv_premium) * 0.2)
        
        # Adjust for profit/loss situation
        pnl_percent = trade_data.get("current_pnl_percent", 0.0)
        if pnl_percent > 20:
            # More confident in taking profits
            base_confidence += min(0.15, pnl_percent / 150)
        elif pnl_percent < -15:
            # Less confident in taking losses
            base_confidence -= min(0.2, abs(pnl_percent) / 75)
        
        # Adjust for time to expiration
        days_to_expiration = trade_data.get("days_to_expiration", 30)
        if days_to_expiration < 7:
            # More confident when close to expiration
            base_confidence += min(0.1, (7 - days_to_expiration) / 7 * 0.1)
        
        return base_confidence
    
    def get_iv_history(self, trade_id):
        """
        Get IV history for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            list: IV history
        """
        return self.iv_history.get(trade_id, [])
    
    def get_vega_history(self, trade_id):
        """
        Get vega history for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            list: Vega history
        """
        return self.vega_history.get(trade_id, [])
    
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
        Clear history for a specific trade or all trades.
        
        Args:
            trade_id (str, optional): Trade to clear. If None, clear all.
        """
        if trade_id:
            if trade_id in self.vega_history:
                del self.vega_history[trade_id]
            if trade_id in self.iv_history:
                del self.iv_history[trade_id]
            self.logger.debug(f"Cleared vega/IV history for trade {trade_id}")
        else:
            self.vega_history = {}
            self.iv_history = {}
            self.logger.debug("Cleared all vega/IV history")
    
    def get_thresholds(self):
        """
        Get current volatility thresholds.
        
        Returns:
            dict: Current thresholds
        """
        return {
            "iv_drop_threshold": self.iv_drop_threshold,
            "iv_spike_threshold": self.iv_spike_threshold,
            "vega_exposure_threshold": self.vega_exposure_threshold,
            "iv_reversion_trigger": self.iv_reversion_trigger,
            "iv_reversion_exit_pct": self.iv_reversion_exit_pct
        }
