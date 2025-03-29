"""
Trade Entry Module

This module handles the entry logic for paper trades in the Option Hunter system.
It validates trades, calculates appropriate position sizes, and records entries.
"""

import logging
from datetime import datetime
import json
import os
import uuid

class TradeEntry:
    """
    Handles the entry of new paper trades, including validation,
    position sizing, and logging.
    """
    
    def __init__(self, config, tradier_api, parameter_hub, notification_system):
        """
        Initialize the TradeEntry component.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
            parameter_hub: Instance of MasterParameterHub for parameters
            notification_system: Instance of NotificationSystem for alerts
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.parameter_hub = parameter_hub
        self.notification_system = notification_system
        
        # Create trade log directory if it doesn't exist
        os.makedirs("logs/trades", exist_ok=True)
        
        self.logger.info("TradeEntry initialized")
    
    def enter_paper_trade(self, trade_data):
        """
        Enter a new paper trade.
        
        Args:
            trade_data (dict): Trade information including:
                - symbol: Underlying symbol
                - option_symbol: Option symbol
                - option_type: 'call' or 'put'
                - strike: Strike price
                - expiration: Expiration date
                - entry_price: Entry price per contract
                - quantity: Number of contracts
                - position_size: Total position size in dollars
                
        Returns:
            bool: True if trade entry was successful
        """
        self.logger.info(f"Entering paper trade for {trade_data['option_symbol']}")
        
        try:
            # Validate the trade
            if not self._validate_trade(trade_data):
                return False
            
            # Generate trade ID if not provided
            if 'trade_id' not in trade_data:
                trade_data['trade_id'] = f"{trade_data['symbol']}_{int(datetime.now().timestamp())}"
            
            # Add entry timestamp
            trade_data['entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Record initial risk metrics
            trade_data['initial_stop_loss'] = self._calculate_stop_loss(trade_data)
            trade_data['initial_target'] = self._calculate_take_profit(trade_data)
            
            # Add status
            trade_data['status'] = 'active'
            
            # Log the trade
            self._log_trade(trade_data)
            
            # Send notification
            self.notification_system.send_trade_entry_notification(trade_data)
            
            self.logger.info(f"Successfully entered paper trade {trade_data['trade_id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error entering paper trade: {str(e)}")
            return False
    
    def enter_live_trade(self, trade_data):
        """
        Enter a new live trade using the Tradier API.
        
        Args:
            trade_data (dict): Trade information
                
        Returns:
            bool: True if trade entry was successful
        """
        self.logger.info(f"Entering live trade for {trade_data['option_symbol']}")
        
        try:
            # Validate the trade
            if not self._validate_trade(trade_data):
                return False
            
            # Get option symbol in Tradier format
            option_symbol = trade_data['option_symbol']
            
            # Create the order
            order_result = self.tradier_api.create_paper_order(
                symbol=trade_data['symbol'],
                side='buy',
                quantity=trade_data['quantity'],
                order_type='limit',
                price=trade_data['entry_price'],
                option_symbol=option_symbol
            )
            
            if order_result.get('status') != 'ok':
                self.logger.error(f"Order creation failed: {order_result.get('message', 'Unknown error')}")
                return False
            
            # Add order ID to trade data
            trade_data['order_id'] = order_result.get('id')
            trade_data['entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trade_data['status'] = 'active'
            
            # Record initial risk metrics
            trade_data['initial_stop_loss'] = self._calculate_stop_loss(trade_data)
            trade_data['initial_target'] = self._calculate_take_profit(trade_data)
            
            # Log the trade
            self._log_trade(trade_data)
            
            # Send notification
            self.notification_system.send_trade_entry_notification(trade_data)
            
            self.logger.info(f"Successfully entered live trade {trade_data.get('trade_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error entering live trade: {str(e)}")
            return False
    
    def _validate_trade(self, trade_data):
        """
        Validate trade data before entry.
        
        Args:
            trade_data (dict): Trade information
                
        Returns:
            bool: True if trade is valid
        """
        # Required fields
        required_fields = [
            'symbol', 'option_symbol', 'option_type', 
            'strike', 'expiration', 'entry_price', 'quantity'
        ]
        
        for field in required_fields:
            if field not in trade_data:
                self.logger.error(f"Missing required field in trade data: {field}")
                return False
        
        # Validate trade against risk parameters
        try:
            # Check if position size is within allowed limits
            max_position = self.parameter_hub.get_parameter('max_position_size_per_symbol_percentage', 5.0)
            
            # Get account balance from API
            account_info = self.tradier_api.get_account_balances()
            if not account_info:
                self.logger.error("Could not get account balance")
                return False
                
            account_value = float(account_info.get('total_equity', 0))
            
            # Calculate position size
            position_size = trade_data['entry_price'] * trade_data['quantity'] * 100  # 100 shares per contract
            max_allowed_size = account_value * (max_position / 100.0)
            
            if position_size > max_allowed_size:
                self.logger.error(
                    f"Position size ${position_size:.2f} exceeds maximum allowed (${max_allowed_size:.2f})"
                )
                return False
            
            # Store position size in trade data
            trade_data['position_size'] = position_size
            
            # Validate entry price is reasonable
            if trade_data['entry_price'] <= 0:
                self.logger.error(f"Invalid entry price: {trade_data['entry_price']}")
                return False
            
            # Ensure expiration is in the future
            expiration_date = datetime.strptime(trade_data['expiration'], '%Y-%m-%d').date()
            if expiration_date <= datetime.now().date():
                self.logger.error(f"Expiration date {trade_data['expiration']} is not in the future")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False
    
    def _calculate_stop_loss(self, trade_data):
        """
        Calculate initial stop loss for the trade.
        
        Args:
            trade_data (dict): Trade information
                
        Returns:
            float: Stop loss price
        """
        # Get stop loss percentage from parameter hub
        stop_loss_pct = self.parameter_hub.get_parameter('default_stop_loss_percentage', 25.0)
        
        if trade_data['option_type'] == 'call':
            # For calls, stop loss is below entry price
            stop_loss = trade_data['entry_price'] * (1 - stop_loss_pct / 100.0)
        else:
            # For puts, stop loss is below entry price
            stop_loss = trade_data['entry_price'] * (1 - stop_loss_pct / 100.0)
        
        # Ensure stop loss is positive
        return max(0.01, stop_loss)
    
    def _calculate_take_profit(self, trade_data):
        """
        Calculate initial take profit target for the trade.
        
        Args:
            trade_data (dict): Trade information
                
        Returns:
            float: Take profit price
        """
        # Get take profit percentage from parameter hub
        take_profit_pct = self.parameter_hub.get_parameter('take_profit_percentage', 50.0)
        
        # Calculate take profit price
        take_profit = trade_data['entry_price'] * (1 + take_profit_pct / 100.0)
        
        return take_profit
    
    def _log_trade(self, trade_data):
        """
        Log trade entry to file.
        
        Args:
            trade_data (dict): Trade information
        """
        try:
            # Create trade log file for this date if it doesn't exist
            log_date = datetime.now().strftime('%Y-%m-%d')
            log_file = f"logs/trades/trades_{log_date}.json"
            
            # Load existing trades if file exists
            trades = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trades = json.load(f)
            
            # Add new trade
            trades.append(trade_data)
            
            # Save back to file
            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2)
            
            self.logger.debug(f"Trade {trade_data['trade_id']} logged to {log_file}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
