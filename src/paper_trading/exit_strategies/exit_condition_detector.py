"""
Exit Condition Detector Module

This module identifies optimal exit conditions for options trades
based on technical indicators, option chain dynamics, and risk parameters.
"""

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class ExitConditionDetector:
    """
    Detects optimal exit conditions for options trades based on
    multiple signals and indicators.
    """
    
    def __init__(self, config, tradier_api, parameter_hub):
        """
        Initialize the ExitConditionDetector component.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
            parameter_hub: Instance of MasterParameterHub for parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.parameter_hub = parameter_hub
        
        # Exit condition thresholds
        self.exit_config = config.get("exit_strategies", {})
        
        self.logger.info("ExitConditionDetector initialized")
    
    def check_exit_conditions(self, trade, option_quote=None, underlying_quote=None):
        """
        Check all potential exit conditions for a trade.
        
        Args:
            trade (dict): Trade information
            option_quote (dict, optional): Current option quote
            underlying_quote (dict, optional): Current underlying quote
            
        Returns:
            tuple: (should_exit, exit_confidence, exit_reason)
        """
        try:
            # Get quotes if not provided
            if option_quote is None:
                option_quote = self.tradier_api.get_option_quote(trade['option_symbol'])
                if not option_quote:
                    self.logger.warning(f"Could not get option quote for {trade['option_symbol']}")
                    return False, 0, None
            
            if underlying_quote is None:
                underlying_quotes = self.tradier_api.get_quotes(trade['symbol'])
                if underlying_quotes.empty:
                    self.logger.warning(f"Could not get underlying quote for {trade['symbol']}")
                    return False, 0, None
                underlying_quote = underlying_quotes.iloc[0].to_dict()
            
            # Store current prices
            current_option_price = (float(option_quote['bid']) + float(option_quote['ask'])) / 2
            current_underlying_price = float(underlying_quote['last'])
            
            # Update trade with current prices
            trade['current_option_price'] = current_option_price
            trade['current_underlying_price'] = current_underlying_price
            
            # Run all exit condition checks
            exit_signals = [
                self._check_profit_target(trade, current_option_price),
                self._check_stop_loss(trade, current_option_price),
                self._check_time_decay(trade, option_quote),
                self._check_technical_signals(trade, underlying_quote),
                self._check_iv_change(trade, option_quote),
                self._check_delta_threshold(trade, option_quote),
                self._check_underlying_reversal(trade, underlying_quote)
            ]
            
            # Combine exit signals (if any signal says exit, we exit)
            should_exit = any(signal[0] for signal in exit_signals)
            
            if should_exit:
                # Get the highest confidence signal
                exit_signals = [s for s in exit_signals if s[0]]
                best_signal = max(exit_signals, key=lambda x: x[1])
                
                exit_confidence = best_signal[1]
                exit_reason = best_signal[2]
                
                return True, exit_confidence, exit_reason
            
            return False, 0, None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {str(e)}")
            return False, 0, None
    
    def _check_profit_target(self, trade, current_price):
        """
        Check if profit target is reached.
        
        Args:
            trade (dict): Trade information
            current_price (float): Current option price
            
        Returns:
            tuple: (should_exit, confidence, reason)
        """
        # Get target from trade or parameter hub
        take_profit = trade.get('current_target', trade.get('initial_target'))
        
        if take_profit is None:
            return False, 0, None
        
        if current_price >= take_profit:
            # Calculate how far past target we are
            excess_percent = ((current_price / take_profit) - 1) * 100
            
            # Higher confidence if we're well past target
            confidence = min(0.95, 0.8 + (excess_percent / 100))
            
            return True, confidence, f"Take profit target reached (${take_profit:.2f})"
        
        # Check for dynamic profit-taking based on time decay
        if 'entry_time' in trade and 'expiration' in trade:
            entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            expiration = datetime.strptime(trade['expiration'], '%Y-%m-%d').date()
            days_to_exp = (expiration - datetime.now().date()).days
            
            # Calculate profit percentage
            profit_pct = ((current_price / trade['entry_price']) - 1) * 100
            
            # As we get closer to expiration, accept smaller profits
            if days_to_exp <= 1 and profit_pct >= 15:
                return True, 0.85, f"Taking profits due to pending expiration ({profit_pct:.2f}%)"
            elif days_to_exp <= 3 and profit_pct >= 25:
                return True, 0.8, f"Taking profits due to approaching expiration ({profit_pct:.2f}%)"
            elif days_to_exp <= 7 and profit_pct >= 40:
                return True, 0.75, f"Taking profits with {days_to_exp} days to expiration ({profit_pct:.2f}%)"
        
        return False, 0, None
