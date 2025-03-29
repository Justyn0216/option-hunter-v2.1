"""
Position Sizing Module

This module handles position sizing calculations based on account size,
risk tolerance, market conditions, and ML/RL-based predictions.
"""

import logging
import numpy as np
from datetime import datetime
import math

class PositionSizer:
    """
    Position sizer that determines optimal trade size based on multiple factors.
    
    Features:
    - Risk-based position sizing
    - Adaptive sizing based on market conditions
    - ML/RL integrated sizing adjustments
    - Portfolio allocation limits and constraints
    """
    
    def __init__(self, trade_params, parameter_hub):
        """
        Initialize the PositionSizer.
        
        Args:
            trade_params (dict): Trading parameters configuration
            parameter_hub: MasterParameterHub instance
        """
        self.logger = logging.getLogger(__name__)
        self.trade_params = trade_params
        self.parameter_hub = parameter_hub
        
        # Extract configuration
        self.account_size = trade_params.get("account_size", 100000.0)
        self.max_risk_per_trade_percentage = trade_params.get("max_risk_per_trade_percentage", 2.0)
        self.max_portfolio_allocation_percentage = trade_params.get("max_portfolio_allocation_percentage", 20.0)
        self.max_position_size_per_symbol_percentage = trade_params.get("max_position_size_per_symbol_percentage", 5.0)
        
        # Active position tracking
        self.active_positions = {}  # Symbol -> position size mapping
        self.total_allocated = 0.0  # Total allocated capital
        
        self.logger.info("PositionSizer initialized")
    
    def calculate_position_size(self, symbol, option_type, option_price, market_regime, volatility_regime, confidence_score=0.0):
        """
        Calculate the optimal position size for a trade.
        
        Args:
            symbol (str): Stock symbol
            option_type (str): 'call' or 'put'
            option_price (float): Current option price (per share)
            market_regime (str): Current market regime
            volatility_regime (str): Current volatility regime
            confidence_score (float): Confidence score from ML/RL (0-1)
            
        Returns:
            float: Position size in dollars
        """
        # Check if we have optimized parameters from ML
        params = self.parameter_hub.get_all_parameters()
        
        # Get risk percentage (possibly adjusted by ML)
        risk_percentage = params.get("max_risk_per_trade_percentage", self.max_risk_per_trade_percentage)
        
        # Determine risk adjustment based on market conditions
        risk_adjustment = self._calculate_risk_adjustment(
            option_type, market_regime, volatility_regime
        )
        
        # Apply confidence-based adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(confidence_score)
        
        # Calculate adjusted risk percentage
        adjusted_risk_percentage = risk_percentage * risk_adjustment * confidence_adjustment
        
        # Ensure risk percentage is within reasonable bounds
        adjusted_risk_percentage = min(self.max_risk_per_trade_percentage, max(0.1, adjusted_risk_percentage))
        
        # Base position size based on account and risk percentage
        # The 100x multiplier is to convert from per-share to per-contract (since options are for 100 shares)
        base_position_size = (self.account_size * adjusted_risk_percentage / 100.0)
        
        # Apply portfolio allocation constraints
        position_size = self._apply_allocation_constraints(base_position_size, symbol)
        
        # Ensure the position size is a multiple of contract size
        contract_value = option_price * 100  # Value of one contract
        num_contracts = math.floor(position_size / contract_value)
        position_size = max(contract_value, num_contracts * contract_value)  # At least one contract
        
        self.logger.info(
            f"Position size for {symbol} {option_type}: ${position_size:.2f} "
            f"({num_contracts} contracts @ ${option_price:.2f}/share)"
        )
        
        return position_size
    
    def _calculate_risk_adjustment(self, option_type, market_regime, volatility_regime):
        """
        Calculate risk adjustment factor based on market conditions.
        
        Args:
            option_type (str): 'call' or 'put'
            market_regime (str): Current market regime
            volatility_regime (str): Current volatility regime
            
        Returns:
            float: Risk adjustment factor (0.5-1.5)
        """
        # Start with a neutral adjustment
        adjustment = 1.0
        
        # Adjust based on market regime and option type alignment
        if option_type.lower() == 'call' and market_regime.lower() == 'bullish':
            adjustment *= 1.2  # Increase size for calls in bullish market
        elif option_type.lower() == 'put' and market_regime.lower() == 'bearish':
            adjustment *= 1.2  # Increase size for puts in bearish market
        elif (option_type.lower() == 'call' and market_regime.lower() == 'bearish') or \
             (option_type.lower() == 'put' and market_regime.lower() == 'bullish'):
            adjustment *= 0.8  # Decrease size for counter-trend trades
        
        # Adjust based on volatility regime
        if volatility_regime.lower() == 'high':
            adjustment *= 0.9  # Reduce position size in high volatility
        elif volatility_regime.lower() == 'low':
            adjustment *= 1.1  # Increase position size in low volatility
        
        # Ensure adjustment is within reasonable bounds
        return min(1.5, max(0.5, adjustment))
    
    def _calculate_confidence_adjustment(self, confidence_score):
        """
        Calculate confidence adjustment factor based on ML/RL predictions.
        
        Args:
            confidence_score (float): Confidence score from ML/RL (0-1)
            
        Returns:
            float: Confidence adjustment factor (0.5-1.5)
        """
        # If no confidence score provided, use neutral adjustment
        if confidence_score <= 0:
            return 1.0
        
        # Scale confidence score to adjustment factor (0.5-1.5)
        adjustment = 0.5 + confidence_score
        
        # Ensure adjustment is within reasonable bounds
        return min(1.5, max(0.5, adjustment))
    
    def _apply_allocation_constraints(self, position_size, symbol):
        """
        Apply portfolio allocation constraints to position size.
        
        Args:
            position_size (float): Base position size
            symbol (str): Stock symbol
            
        Returns:
            float: Adjusted position size
        """
        # Check overall portfolio allocation limit
        max_portfolio_allocation = self.account_size * (self.max_portfolio_allocation_percentage / 100.0)
        remaining_allocation = max_portfolio_allocation - self.total_allocated
        
        # Ensure we don't exceed overall portfolio allocation
        if position_size > remaining_allocation:
            self.logger.warning(
                f"Reducing position size from ${position_size:.2f} to ${remaining_allocation:.2f} "
                f"due to portfolio allocation constraint"
            )
            position_size = remaining_allocation
        
        # Check per-symbol allocation limit
        max_symbol_allocation = self.account_size * (self.max_position_size_per_symbol_percentage / 100.0)
        current_symbol_allocation = self.active_positions.get(symbol, 0.0)
        
        # Ensure we don't exceed per-symbol allocation
        if current_symbol_allocation + position_size > max_symbol_allocation:
            adjusted_size = max(0, max_symbol_allocation - current_symbol_allocation)
            self.logger.warning(
                f"Reducing position size from ${position_size:.2f} to ${adjusted_size:.2f} "
                f"due to per-symbol allocation constraint"
            )
            position_size = adjusted_size
        
        return position_size
    
    def update_position(self, symbol, amount, is_entry=True):
        """
        Update tracking of active positions.
        
        Args:
            symbol (str): Stock symbol
            amount (float): Position amount in dollars
            is_entry (bool): True if this is a new position entry, False if exit
        """
        if is_entry:
            # Add new position
            current = self.active_positions.get(symbol, 0.0)
            self.active_positions[symbol] = current + amount
            self.total_allocated += amount
        else:
            # Remove existing position
            if symbol in self.active_positions:
                self.total_allocated -= self.active_positions[symbol]
                del self.active_positions[symbol]
    
    def get_available_capital(self):
        """
        Get available capital for new positions.
        
        Returns:
            float: Available capital in dollars
        """
        max_allocation = self.account_size * (self.max_portfolio_allocation_percentage / 100.0)
        return max(0, max_allocation - self.total_allocated)
    
    def get_allocation_percentage(self):
        """
        Get the current allocation percentage.
        
        Returns:
            float: Percentage of account currently allocated
        """
        return (self.total_allocated / self.account_size) * 100.0
    
    def get_max_position_size(self):
        """
        Get maximum allowed position size for any new trade.
        
        Returns:
            float: Maximum allowable position size
        """
        # Check portfolio allocation constraint
        max_portfolio_allocation = self.account_size * (self.max_portfolio_allocation_percentage / 100.0)
        remaining_allocation = max_portfolio_allocation - self.total_allocated
        
        # Check risk-based constraint
        risk_based_max = self.account_size * (self.max_risk_per_trade_percentage / 100.0)
        
        # Return the smaller of the two constraints
        return min(remaining_allocation, risk_based_max)
    
    def update_account_size(self, new_account_size):
        """
        Update the account size (e.g., after significant P&L changes).
        
        Args:
            new_account_size (float): New account size
        """
        self.account_size = new_account_size
        self.logger.info(f"Account size updated to ${new_account_size:.2f}")
    
    def reset(self):
        """Reset active position tracking."""
        self.active_positions = {}
        self.total_allocated = 0.0
        self.logger.info("Position sizer reset")
