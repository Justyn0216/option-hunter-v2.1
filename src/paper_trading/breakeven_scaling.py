"""
Breakeven Scaling Module

This module implements auto-adjusting stop-loss systems based on ML/RL
predictions of trade success, adjusting risk management dynamically.
"""

import logging
import numpy as np
from datetime import datetime, timedelta

class BreakevenScaling:
    """
    Implements dynamic stop-loss adjustment based on ML predictions,
    moving stops to breakeven or trailing positions based on trade progress
    and prediction confidence.
    """
    
    def __init__(self, config, parameter_hub):
        """
        Initialize the BreakevenScaling component.
        
        Args:
            config (dict): Configuration settings
            parameter_hub: Instance of MasterParameterHub for parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.parameter_hub = parameter_hub
        
        # Load scaling parameters
        self.scaling_config = config.get("breakeven_scaling", {})
        
        # Track adjustments made to each trade
        self.adjustments = {}
        
        self.logger.info("BreakevenScaling initialized")
    
    def calculate_stop_loss(self, trade, prediction_score=None, market_conditions=None):
        """
        Calculate the optimal stop loss for a trade based on predictions and market conditions.
        
        Args:
            trade (dict): Trade information
            prediction_score (float, optional): ML prediction score (0-100)
            market_conditions (str, optional): Current market conditions
            
        Returns:
            float: Calculated stop loss price
        """
        # Get base stop loss percentage from parameter hub
        base_stop_loss_pct = self.parameter_hub.get_parameter('default_stop_loss_percentage', 25.0)
        
        # Adjust based on prediction score if available
        if prediction_score is not None:
            # Higher confidence = tighter stop loss
            confidence_factor = self._calculate_confidence_factor(prediction_score)
            adjusted_stop_loss_pct = base_stop_loss_pct * confidence_factor
        else:
            adjusted_stop_loss_pct = base_stop_loss_pct
        
        # Further adjust based on market conditions
        if market_conditions is not None:
            volatility_factor = self._get_volatility_factor(market_conditions)
            adjusted_stop_loss_pct *= volatility_factor
        
        # Calculate stop loss price
        entry_price = trade['entry_price']
        
        if trade['option_type'].lower() == 'call':
            stop_loss = entry_price * (1 - adjusted_stop_loss_pct / 100.0)
        else:  # Put
            stop_loss = entry_price * (1 - adjusted_stop_loss_pct / 100.0)
        
        # Ensure stop loss is positive
        stop_loss = max(0.01, stop_loss)
        
        self.logger.debug(
            f"Calculated stop loss for {trade['trade_id']}: ${stop_loss:.4f} "
            f"({adjusted_stop_loss_pct:.2f}% from entry price)"
        )
        
        return stop_loss
    
    def _calculate_confidence_factor(self, prediction_score):
        """
        Calculate confidence factor based on ML prediction score.
        
        Args:
            prediction_score (float): ML prediction score (0-100)
            
        Returns:
            float: Confidence factor (typically 0.5-1.5)
        """
        # Map prediction score (0-100) to confidence factor
        # Higher confidence = smaller factor = tighter stop
        min_factor = self.scaling_config.get("min_confidence_factor", 0.5)
        max_factor = self.scaling_config.get("max_confidence_factor", 1.5)
        
        # Normalize score between 0 and 1
        normalized_score = prediction_score / 100.0
        
        # Calculate factor
        factor = max_factor - (normalized_score * (max_factor - min_factor))
        
        return factor
    
    def _get_volatility_factor(self, market_conditions):
        """
        Get volatility adjustment factor based on market conditions.
        
        Args:
            market_conditions (str): Current market conditions
            
        Returns:
            float: Volatility factor to adjust stop loss
        """
        # Default factors for different market conditions
        volatility_factors = {
            'low_volatility': 0.8,   # Tighter stops in calm markets
            'normal': 1.0,           # Normal market conditions
            'high_volatility': 1.5,  # Wider stops in volatile markets
            'extreme_volatility': 2.0,  # Much wider stops in extreme conditions
            'bullish': 0.9,          # Slightly tighter in bullish markets
            'bearish': 1.2           # Wider in bearish markets
        }
        
        # Get custom factors from config if available
        custom_factors = self.scaling_config.get("volatility_factors", {})
        
        # Merge custom factors with defaults
        volatility_factors.update(custom_factors)
        
        # Return factor for given market conditions, default to normal
        return volatility_factors.get(market_conditions, 1.0)
    
    def update_stop_loss(self, trade, current_price, market_conditions=None):
        """
        Update stop loss based on trade progress and ML predictions.
        
        Args:
            trade (dict): Trade information
            current_price (float): Current option price
            market_conditions (str, optional): Current market conditions
            
        Returns:
            float: Updated stop loss price
        """
        trade_id = trade['trade_id']
        entry_price = trade['entry_price']
        
        # Calculate current P&L percentage
        pnl_percent = ((current_price / entry_price) - 1) * 100
        
        # Get adjustment thresholds
        breakeven_threshold = self.scaling_config.get("breakeven_threshold", 15)
        trailing_threshold = self.scaling_config.get("trailing_threshold", 30)
        trailing_stop_pct = self.scaling_config.get("trailing_stop_percentage", 15)
        
        # Get ML prediction of trade success if available
        prediction_score = trade.get('success_prediction', None)
        
        # Initialize current stop if not set
        current_stop = trade.get('current_stop_loss', trade.get('initial_stop_loss'))
        
        # Track if we're making an adjustment
        adjusted = False
        
        # Adjust based on trade progress
        if trade_id not in self.adjustments and pnl_percent >= breakeven_threshold:
            # Move to breakeven
            new_stop = entry_price
            reason = "breakeven"
            adjusted = True
            
            # Adjust threshold based on prediction if available
            if prediction_score is not None:
                # Higher prediction score = move to breakeven sooner
                confidence_factor = 1.0 - (prediction_score / 100.0 * 0.5)  # 0.5 to 1.0
                scaled_threshold = breakeven_threshold * confidence_factor
                
                if pnl_percent >= scaled_threshold:
                    new_stop = entry_price
                    reason = f"breakeven (prediction: {prediction_score})"
                    adjusted = True
        
        elif trade_id in self.adjustments and pnl_percent >= trailing_threshold:
            # Calculate trailing stop
            trailing_stop = current_price * (1 - trailing_stop_pct / 100.0)
            
            # Only adjust if new stop is higher than current
            if trailing_stop > current_stop:
                new_stop = trailing_stop
                reason = "trailing"
                adjusted = True
        
        else:
            # No adjustment
            return current_stop
        
        # Record adjustment if made
        if adjusted:
            self.adjustments[trade_id] = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': reason,
                'price': new_stop,
                'pnl_percent': pnl_percent
            }
            
            self.logger.info(
                f"Updated stop loss for {trade_id} to ${new_stop:.4f} ({reason})"
            )
            
            return new_stop
        
        return current_stop
    
    def get_adjustment_history(self, trade_id):
        """
        Get stop loss adjustment history for a trade.
        
        Args:
            trade_id (str): Trade ID
            
        Returns:
            dict: Adjustment history or None if not found
        """
        return self.adjustments.get(trade_id, None)
    
    def calculate_take_profit(self, trade, prediction_score=None):
        """
        Calculate take profit target based on ML prediction confidence.
        
        Args:
            trade (dict): Trade information
            prediction_score (float, optional): ML prediction score (0-100)
            
        Returns:
            float: Take profit price
        """
        # Get base take profit percentage from parameter hub
        base_tp_pct = self.parameter_hub.get_parameter('take_profit_percentage', 50.0)
        
        # Adjust based on prediction confidence if available
        if prediction_score is not None:
            # Higher confidence = higher target
            confidence_bonus = (prediction_score / 100.0) * 25.0  # Up to 25% bonus
            adjusted_tp_pct = base_tp_pct + confidence_bonus
        else:
            adjusted_tp_pct = base_tp_pct
        
        # Calculate take profit price
        entry_price = trade['entry_price']
        take_profit = entry_price * (1 + adjusted_tp_pct / 100.0)
        
        self.logger.debug(
            f"Calculated take profit for {trade['trade_id']}: ${take_profit:.4f} "
            f"({adjusted_tp_pct:.2f}% from entry price)"
        )
        
        return take_profit
