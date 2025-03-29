"""
Trade Monitor Module

This module monitors active trades, checks for exit conditions,
and manages the exit process for paper trades.
"""

import logging
from datetime import datetime, timedelta
import json
import os
import math

class TradeMonitor:
    """
    Monitors active trades, detects exit signals, and handles the trade exit process.
    """
    
    def __init__(self, config, tradier_api, notification_system, parameter_hub):
        """
        Initialize the TradeMonitor component.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
            notification_system: Instance of NotificationSystem for alerts
            parameter_hub: Instance of MasterParameterHub for parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.notification_system = notification_system
        self.parameter_hub = parameter_hub
        
        # Load exit strategies from config
        self.exit_strategies = self.config.get("exit_strategies", {})
        
        # Track stop loss adjustments
        self.stop_loss_adjustments = {}
        
        self.logger.info("TradeMonitor initialized")
    
    def check_exit_conditions(self, trade):
        """
        Check if any exit conditions are met for a trade.
        
        Args:
            trade (dict): Trade information
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        try:
            # Skip if trade is already closed
            if trade.get('status') != 'active':
                return False, None
            
            # Get current option price
            option_symbol = trade['option_symbol']
            quote = self.tradier_api.get_option_quote(option_symbol)
            
            if not quote:
                self.logger.warning(f"Could not get quote for {option_symbol}")
                return False, None
            
            # Use mid price as current price
            current_price = (float(quote['bid']) + float(quote['ask'])) / 2
            
            # Update trade with current price
            trade['current_price'] = current_price
            
            # Calculate P&L
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            current_pnl = (current_price - entry_price) * quantity * 100  # 100 shares per contract
            pnl_percent = ((current_price / entry_price) - 1) * 100
            
            trade['current_pnl'] = current_pnl
            trade['current_pnl_percent'] = pnl_percent
            
            # Check each exit strategy
            exit_checks = [
                self._check_stop_loss(trade, current_price),
                self._check_take_profit(trade, current_price),
                self._check_time_based_exit(trade),
                self._check_greek_based_exit(trade, quote),
                self._check_max_loss_exit(trade, current_pnl)
            ]
            
            # Process exit checks
            for should_exit, reason in exit_checks:
                if should_exit:
                    return True, reason
            
            # Check if we should adjust stop loss
            self._adjust_stop_loss(trade, current_price, pnl_percent)
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions for {trade.get('trade_id', 'unknown')}: {str(e)}")
            return False, None
    
    def _check_stop_loss(self, trade, current_price):
        """
        Check if stop loss is triggered.
        
        Args:
            trade (dict): Trade information
            current_price (float): Current option price
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # Get stop loss level
        stop_loss = trade.get('current_stop_loss', trade.get('initial_stop_loss'))
        
        if stop_loss is None:
            return False, None
        
        if current_price <= stop_loss:
            return True, "Stop loss triggered"
        
        return False, None
    
    def _check_take_profit(self, trade, current_price):
        """
        Check if take profit is triggered.
        
        Args:
            trade (dict): Trade information
            current_price (float): Current option price
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # Get take profit level
        take_profit = trade.get('current_target', trade.get('initial_target'))
        
        if take_profit is None:
            return False, None
        
        if current_price >= take_profit:
            return True, "Take profit triggered"
        
        return False, None
    
    def _check_time_based_exit(self, trade):
        """
        Check time-based exit conditions.
        
        Args:
            trade (dict): Trade information
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # Check expiration date
        expiration = datetime.strptime(trade['expiration'], '%Y-%m-%d').date()
        today = datetime.now().date()
        days_to_expiration = (expiration - today).days
        
        # Exit if close to expiration (default 2 days)
        min_days_to_expiration = self.exit_strategies.get("min_days_to_expiration", 2)
        if days_to_expiration <= min_days_to_expiration:
            return True, f"Close to expiration ({days_to_expiration} days left)"
        
        # Check max hold time
        if 'entry_time' in trade:
            entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            max_hold_days = self.exit_strategies.get("max_hold_days")
            
            if max_hold_days:
                max_time = entry_time + timedelta(days=max_hold_days)
                if datetime.now() > max_time:
                    return True, f"Max hold time of {max_hold_days} days exceeded"
        
        return False, None
    
    def _check_greek_based_exit(self, trade, quote):
        """
        Check Greek-based exit conditions.
        
        Args:
            trade (dict): Trade information
            quote (dict): Current option quote with Greeks
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # Check if Greeks are available
        if 'greeks' not in quote or not quote['greeks']:
            return False, None
        
        greeks = quote['greeks']
        
        # Theta decay threshold - exit if losing too much value per day
        theta_threshold = self.exit_strategies.get("theta_threshold")
        if theta_threshold and 'theta' in greeks:
            theta = float(greeks['theta'])
            if abs(theta) > theta_threshold:
                return True, f"Theta decay too high ({theta:.4f})"
        
        # Delta threshold - exit if delta moved significantly from initial
        if 'delta' in greeks and 'initial_delta' in trade:
            delta = float(greeks['delta'])
            initial_delta = trade['initial_delta']
            delta_change = abs(delta - initial_delta)
            
            delta_threshold = self.exit_strategies.get("delta_change_threshold", 0.3)
            if delta_change > delta_threshold:
                return True, f"Delta changed significantly ({delta_change:.4f} from {initial_delta:.4f} to {delta:.4f})"
        
        # IV contraction - exit if IV dropped significantly
        if 'mid_iv' in greeks and 'initial_iv' in trade:
            current_iv = float(greeks['mid_iv'])
            initial_iv = trade['initial_iv']
            iv_change_pct = ((current_iv / initial_iv) - 1) * 100
            
            iv_contraction_threshold = self.exit_strategies.get("iv_contraction_threshold", -30)
            if iv_change_pct < iv_contraction_threshold:
                return True, f"IV contracted significantly ({iv_change_pct:.2f}%)"
        
        return False, None
    
    def _check_max_loss_exit(self, trade, current_pnl):
        """
        Check if maximum loss threshold is reached.
        
        Args:
            trade (dict): Trade information
            current_pnl (float): Current P&L in dollars
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # Get max loss percentage
        max_loss_pct = self.exit_strategies.get("max_loss_percentage", 50)
        
        # Calculate entry value
        entry_value = trade['entry_price'] * trade['quantity'] * 100
        
        # Calculate loss percentage
        loss_pct = (current_pnl / entry_value) * 100
        
        if loss_pct <= -max_loss_pct:
            return True, f"Maximum loss threshold reached ({loss_pct:.2f}%)"
        
        return False, None
    
    def _adjust_stop_loss(self, trade, current_price, pnl_percent):
        """
        Adjust stop loss based on trade performance.
        
        Args:
            trade (dict): Trade information
            current_price (float): Current option price
            pnl_percent (float): Current P&L percentage
        """
        trade_id = trade['trade_id']
        
        # Get breakeven adjustment threshold
        breakeven_threshold = self.exit_strategies.get("breakeven_threshold", 15)
        
        # Get trailing stop parameters
        trailing_stop_pct = self.exit_strategies.get("trailing_stop_percentage", 15)
        
        # Move to breakeven if profit exceeds threshold
        if pnl_percent >= breakeven_threshold and trade_id not in self.stop_loss_adjustments:
            # Set stop loss to entry price
            trade['current_stop_loss'] = trade['entry_price']
            self.stop_loss_adjustments[trade_id] = 'breakeven'
            self.logger.info(f"Moved stop loss to breakeven for trade {trade_id}")
        
        # Use trailing stop if profit is significant
        elif pnl_percent >= 2 * breakeven_threshold:
            # Calculate trailing stop level
            trailing_stop = current_price * (1 - trailing_stop_pct / 100.0)
            
            # Only raise stop loss, never lower it
            current_stop = trade.get('current_stop_loss', 0)
            if trailing_stop > current_stop:
                trade['current_stop_loss'] = trailing_stop
                self.stop_loss_adjustments[trade_id] = 'trailing'
                self.logger.info(f"Raised trailing stop to {trailing_stop:.4f} for trade {trade_id}")
    
    def exit_paper_trade(self, trade):
        """
        Exit a paper trade.
        
        Args:
            trade (dict): Trade information
                
        Returns:
            bool: True if trade exit was successful
        """
        self.logger.info(f"Exiting paper trade {trade['trade_id']}")
        
        try:
            # Update trade status
            trade['status'] = 'closed'
            
            # Calculate final P&L
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            quantity = trade['quantity']
            
            pnl = (exit_price - entry_price) * quantity * 100  # 100 shares per contract
            pnl_percent = ((exit_price / entry_price) - 1) * 100
            
            trade['pnl'] = pnl
            trade['pnl_percent'] = pnl_percent
            
            # Log the updated trade
            self._log_trade(trade)
            
            # Send notification
            self.notification_system.send_trade_exit_notification(trade)
            
            self.logger.info(f"Successfully exited paper trade {trade['trade_id']} with P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exiting paper trade: {str(e)}")
            return False
    
    def exit_live_trade(self, trade):
        """
        Exit a live trade using the Tradier API.
        
        Args:
            trade (dict): Trade information
                
        Returns:
            bool: True if trade exit was successful
        """
        self.logger.info(f"Exiting live trade {trade['trade_id']}")
        
        try:
            # Get option symbol in Tradier format
            option_symbol = trade['option_symbol']
            
            # Create the sell order
            order_result = self.tradier_api.create_paper_order(
                symbol=trade['symbol'],
                side='sell',
                quantity=trade['quantity'],
                order_type='limit',
                price=trade['exit_price'],
                option_symbol=option_symbol
            )
            
            if order_result.get('status') != 'ok':
                self.logger.error(f"Order creation failed: {order_result.get('message', 'Unknown error')}")
                return False
            
            # Update trade data
            trade['exit_order_id'] = order_result.get('id')
            trade['status'] = 'closed'
            
            # Calculate final P&L
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            quantity = trade['quantity']
            
            pnl = (exit_price - entry_price) * quantity * 100  # 100 shares per contract
            pnl_percent = ((exit_price / entry_price) - 1) * 100
            
            trade['pnl'] = pnl
            trade['pnl_percent'] = pnl_percent
            
            # Log the updated trade
            self._log_trade(trade)
            
            # Send notification
            self.notification_system.send_trade_exit_notification(trade)
            
            self.logger.info(f"Successfully exited live trade {trade['trade_id']} with P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exiting live trade: {str(e)}")
            return False
    
    def _log_trade(self, trade):
        """
        Log trade exit to file.
        
        Args:
            trade (dict): Trade information
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
            
            # Find and update existing trade
            updated = False
            for i, existing_trade in enumerate(trades):
                if existing_trade.get('trade_id') == trade['trade_id']:
                    trades[i] = trade
                    updated = True
                    break
            
            # Add as new if not found
            if not updated:
                trades.append(trade)
            
            # Save back to file
            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2)
            
            self.logger.debug(f"Updated trade {trade['trade_id']} in {log_file}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade update: {str(e)}")
