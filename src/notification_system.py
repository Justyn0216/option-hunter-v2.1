"""
Notification System Module

This module handles notifications to Discord webhooks for trade entries,
exits, valuation status changes, EOD recap, and PnL milestones.
"""

import logging
import json
import time
import requests
from datetime import datetime
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plots

class NotificationSystem:
    """
    Notification system for sending alerts and updates to Discord.
    
    Features:
    - Trade entry and exit notifications
    - Daily summaries
    - Error reporting
    - Performance milestones
    - System status updates
    """
    
    def __init__(self, webhook_url, config):
        """
        Initialize the notification system.
        
        Args:
            webhook_url (str): Discord webhook URL
            config (dict): Notification configuration
        """
        self.logger = logging.getLogger(__name__)
        self.webhook_url = webhook_url
        self.config = config
        
        # Extract configuration
        self.enable_trade_notifications = config.get("enable_trade_notifications", True)
        self.enable_daily_summary = config.get("enable_daily_summary", True)
        self.enable_error_notifications = config.get("enable_error_notifications", True)
        self.enable_milestone_notifications = config.get("enable_milestone_notifications", True)
        self.milestone_pnl_thresholds = config.get("milestone_pnl_thresholds", [1000, 5000, 10000, 50000])
        self.milestone_win_streak_thresholds = config.get("milestone_win_streak_thresholds", [3, 5, 10, 15])
        
        # Track milestones to avoid duplicates
        self.reached_pnl_milestones = set()
        self.reached_streak_milestones = set()
        self.win_streak = 0
        self.loss_streak = 0
        
        self.logger.info("Notification system initialized")
    
    def _send_discord_message(self, content, embed=None, file=None):
        """
        Send a message to Discord webhook.
        
        Args:
            content (str): Message content
            embed (dict, optional): Discord embed object
            file (tuple, optional): File to upload (filename, file_content)
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.webhook_url:
            self.logger.warning("No webhook URL configured. Cannot send notification.")
            return False
        
        try:
            payload = {"content": content}
            
            if embed:
                payload["embeds"] = [embed]
            
            files = {}
            if file:
                filename, file_content = file
                files = {"file": (filename, file_content)}
            
            if files:
                # If we have files, content and embeds need to be part of form data
                multipart_data = {"payload_json": json.dumps(payload)}
                response = requests.post(self.webhook_url, data=multipart_data, files=files)
            else:
                # Otherwise, send as JSON
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.webhook_url, json=payload, headers=headers)
            
            response.raise_for_status()
            
            # Discord rate limiting
            if response.status_code == 429:
                retry_after = response.json().get('retry_after', 1)
                time.sleep(retry_after / 1000)  # Convert ms to seconds
                
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending Discord notification: {str(e)}")
            return False
    
    def send_trade_entry_notification(self, trade_data):
        """
        Send a notification for a new trade entry.
        
        Args:
            trade_data (dict): Trade entry data
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enable_trade_notifications:
            return False
        
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            option_type = trade_data.get('option_type', 'Unknown')
            strike = trade_data.get('strike', 'Unknown')
            expiration = trade_data.get('expiration', 'Unknown')
            entry_price = trade_data.get('entry_price', 0.0)
            quantity = trade_data.get('quantity', 0)
            position_size = trade_data.get('position_size', 0.0)
            timestamp = trade_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Calculate greeks if available
            delta = trade_data.get('delta', 'N/A')
            gamma = trade_data.get('gamma', 'N/A')
            theta = trade_data.get('theta', 'N/A')
            vega = trade_data.get('vega', 'N/A')
            
            # Create embed
            color = 3066993  # Green
            if option_type.lower() == 'put':
                color = 15158332  # Red
            
            embed = {
                "title": f"🚀 New Trade Entry: {symbol} {option_type.upper()} {strike} {expiration}",
                "color": color,
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "Option Type", "value": option_type.upper(), "inline": True},
                    {"name": "Strike", "value": f"${strike}", "inline": True},
                    {"name": "Expiration", "value": expiration, "inline": True},
                    {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                    {"name": "Quantity", "value": str(quantity), "inline": True},
                    {"name": "Position Size", "value": f"${position_size:.2f}", "inline": True},
                    {"name": "Delta", "value": f"{delta}", "inline": True},
                    {"name": "Gamma", "value": f"{gamma}", "inline": True},
                    {"name": "Theta", "value": f"{theta}", "inline": True},
                    {"name": "Vega", "value": f"{vega}", "inline": True}
                ],
                "footer": {"text": f"Entry Time: {timestamp}"}
            }
            
            # Add reasoning if available
            if 'entry_reason' in trade_data:
                embed["fields"].append({"name": "Entry Reason", "value": trade_data['entry_reason'], "inline": False})
                
            # Add screenshot if available
            file = None
            if 'chart_image' in trade_data and trade_data['chart_image']:
                file = (f"{symbol}_{timestamp.replace(':', '-')}.png", trade_data['chart_image'])
            
            # Send notification
            return self._send_discord_message(
                f"Option Hunter has entered a new {option_type.upper()} position on {symbol}",
                embed,
                file
            )
            
        except Exception as e:
            self.logger.error(f"Error creating trade entry notification: {str(e)}")
            return False
    
    def send_trade_exit_notification(self, trade_data):
        """
        Send a notification for a trade exit.
        
        Args:
            trade_data (dict): Trade exit data
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enable_trade_notifications:
            return False
        
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            option_type = trade_data.get('option_type', 'Unknown')
            strike = trade_data.get('strike', 'Unknown')
            expiration = trade_data.get('expiration', 'Unknown')
            entry_price = trade_data.get('entry_price', 0.0)
            exit_price = trade_data.get('exit_price', 0.0)
            quantity = trade_data.get('quantity', 0)
            pnl = trade_data.get('pnl', 0.0)
            pnl_percent = trade_data.get('pnl_percent', 0.0)
            position_size = trade_data.get('position_size', 0.0)
            timestamp = trade_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Determine if win or loss
            is_win = pnl > 0
            
            # Update streaks
            if is_win:
                self.win_streak += 1
                self.loss_streak = 0
            else:
                self.win_streak = 0
                self.loss_streak += 1
            
            # Create embed
            color = 3066993 if is_win else 15158332  # Green if win, Red if loss
            
            embed = {
                "title": f"{'🎯 Profitable' if is_win else '📉 Unprofitable'} Trade Exit: {symbol} {option_type.upper()} {strike} {expiration}",
                "color": color,
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "Option Type", "value": option_type.upper(), "inline": True},
                    {"name": "Strike", "value": f"${strike}", "inline": True},
                    {"name": "Expiration", "value": expiration, "inline": True},
                    {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                    {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
                    {"name": "Quantity", "value": str(quantity), "inline": True},
                    {"name": "P&L", "value": f"${pnl:.2f} ({pnl_percent:.2f}%)", "inline": True},
                    {"name": "Trade Duration", "value": trade_data.get('duration', 'N/A'), "inline": True}
                ],
                "footer": {"text": f"Exit Time: {timestamp} | Current Win Streak: {self.win_streak}"}
            }
            
            # Add exit reason if available
            if 'exit_reason' in trade_data:
                embed["fields"].append({"name": "Exit Reason", "value": trade_data['exit_reason'], "inline": False})
                
            # Add screenshot if available
            file = None
            if 'chart_image' in trade_data and trade_data['chart_image']:
                file = (f"{symbol}_exit_{timestamp.replace(':', '-')}.png", trade_data['chart_image'])
            
            # Send notification
            result = self._send_discord_message(
                f"Option Hunter has {'profitably' if is_win else 'unprofitably'} exited a {option_type.upper()} position on {symbol} with P&L: ${pnl:.2f} ({pnl_percent:.2f}%)",
                embed,
                file
            )
            
            # Check for milestones
            if self.enable_milestone_notifications:
                self._check_pnl_milestone(pnl)
                self._check_streak_milestone()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating trade exit notification: {str(e)}")
            return False
    
    def _check_pnl_milestone(self, pnl):
        """
        Check if a PnL milestone has been reached and send a notification if needed.
        
        Args:
            pnl (float): Profit and loss amount
        """
        if pnl <= 0:
            return
        
        for threshold in sorted(self.milestone_pnl_thresholds):
            if pnl >= threshold and threshold not in self.reached_pnl_milestones:
                self.reached_pnl_milestones.add(threshold)
                
                embed = {
                    "title": f"🏆 PnL Milestone Reached: ${threshold:,}",
                    "color": 16766720,  # Gold
                    "description": f"Congratulations! Option Hunter has reached a profit milestone of ${threshold:,} with a trade of ${pnl:,.2f}!",
                    "timestamp": datetime.now().isoformat()
                }
                
                self._send_discord_message("🏆 PnL Milestone Reached!", embed)
    
    def _check_streak_milestone(self):
        """Check if a win streak milestone has been reached and send a notification if needed."""
        if self.win_streak == 0:
            return
        
        for threshold in sorted(self.milestone_win_streak_thresholds):
            if self.win_streak == threshold:
                
                embed = {
                    "title": f"🔥 Win Streak Milestone: {threshold} Wins in a Row!",
                    "color": 16766720,  # Gold
                    "description": f"Option Hunter is on fire with {threshold} consecutive winning trades!",
                    "timestamp": datetime.now().isoformat()
                }
                
                self._send_discord_message("🔥 Win Streak Milestone!", embed)
    
    def send_valuation_change_notification(self, option_data):
        """
        Send a notification when an option's valuation status changes.
        
        Args:
            option_data (dict): Option valuation data
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enable_trade_notifications:
            return False
        
        try:
            symbol = option_data.get('symbol', 'Unknown')
            option_type = option_data.get('option_type', 'Unknown')
            strike = option_data.get('strike', 'Unknown')
            expiration = option_data.get('expiration', 'Unknown')
            market_price = option_data.get('market_price', 0.0)
            model_price = option_data.get('model_price', 0.0)
            diff_percent = option_data.get('diff_percent', 0.0)
            valuation = option_data.get('valuation', 'Unknown')
            timestamp = option_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Set color based on valuation
            color = 10181046  # Default gray
            if valuation.lower() == 'undervalued':
                color = 3066993  # Green
            elif valuation.lower() == 'overvalued':
                color = 15158332  # Red
            
            embed = {
                "title": f"{valuation.title()} Option Detected: {symbol} {option_type.upper()} {strike} {expiration}",
                "color": color,
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "Option Type", "value": option_type.upper(), "inline": True},
                    {"name": "Strike", "value": f"${strike}", "inline": True},
                    {"name": "Expiration", "value": expiration, "inline": True},
                    {"name": "Market Price", "value": f"${market_price:.2f}", "inline": True},
                    {"name": "Model Price", "value": f"${model_price:.2f}", "inline": True},
                    {"name": "Difference", "value": f"{diff_percent:.2f}%", "inline": True},
                    {"name": "Valuation", "value": valuation.title(), "inline": True}
                ],
                "footer": {"text": f"Detection Time: {timestamp}"}
            }
            
            # Add model breakdown if available
            if 'model_breakdown' in option_data:
                model_text = "\n".join([f"{model}: ${price:.4f}" for model, price in option_data['model_breakdown'].items()])
                embed["fields"].append({"name": "Pricing Model Breakdown", "value": model_text, "inline": False})
            
            # Send notification
            return self._send_discord_message(
                f"Option Hunter has detected a {valuation.lower()} option: {symbol} {option_type.upper()} {strike} {expiration}",
                embed
            )
            
        except Exception as e:
            self.logger.error(f"Error creating valuation change notification: {str(e)}")
            return False
    
    def send_daily_summary(self, summary_data):
        """
        Send an end-of-day summary notification.
        
        Args:
            summary_data (dict): Daily trading summary data
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enable_daily_summary:
            return False
        
        try:
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            losing_trades = summary_data.get('losing_trades', 0)
            win_rate = summary_data.get('win_rate', 0.0)
            total_pnl = summary_data.get('total_pnl', 0.0)
            biggest_win = summary_data.get('biggest_win', 0.0)
            biggest_loss = summary_data.get('biggest_loss', 0.0)
            avg_win = summary_data.get('avg_win', 0.0)
            avg_loss = summary_data.get('avg_loss', 0.0)
            profit_factor = summary_data.get('profit_factor', 0.0)
            
            # Create a performance chart if trade data is available
            file = None
            if 'trade_history' in summary_data and summary_data['trade_history']:
                trade_history = summary_data['trade_history']
                if len(trade_history) > 0:
                    # Create a DataFrame from trade history
                    df = pd.DataFrame(trade_history)
                    
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Plot cumulative P&L
                    if 'cumulative_pnl' in df.columns:
                        ax1.plot(df['timestamp'], df['cumulative_pnl'])
                        ax1.set_title('Cumulative P&L')
                        ax1.set_ylabel('P&L ($)')
                        ax1.grid(True)
                        
                    # Plot trade P&L as a bar chart
                    if 'pnl' in df.columns:
                        colors = ['green' if x > 0 else 'red' for x in df['pnl']]
                        ax2.bar(range(len(df)), df['pnl'], color=colors)
                        ax2.set_title('Individual Trade P&L')
                        ax2.set_xlabel('Trade #')
                        ax2.set_ylabel('P&L ($)')
                        ax2.grid(True)
                    
                    plt.tight_layout()
                    
                    # Save the figure to a bytes buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    
                    # Create file for upload
                    file = (f"daily_summary_{date}.png", buf.getvalue())
                    
                    plt.close(fig)
            
            # Create embed
            color = 3066993 if total_pnl > 0 else 15158332  # Green if profitable, Red if not
            
            embed = {
                "title": f"📊 Daily Trading Summary: {date}",
                "color": color,
                "fields": [
                    {"name": "Total Trades", "value": str(total_trades), "inline": True},
                    {"name": "Winning Trades", "value": str(winning_trades), "inline": True},
                    {"name": "Losing Trades", "value": str(losing_trades), "inline": True},
                    {"name": "Win Rate", "value": f"{win_rate:.2f}%", "inline": True},
                    {"name": "Total P&L", "value": f"${total_pnl:.2f}", "inline": True},
                    {"name": "Biggest Win", "value": f"${biggest_win:.2f}", "inline": True},
                    {"name": "Biggest Loss", "value": f"${biggest_loss:.2f}", "inline": True},
                    {"name": "Average Win", "value": f"${avg_win:.2f}", "inline": True},
                    {"name": "Average Loss", "value": f"${avg_loss:.2f}", "inline": True},
                    {"name": "Profit Factor", "value": f"{profit_factor:.2f}", "inline": True}
                ],
                "footer": {"text": f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
            }
            
            # Add top symbols if available
            if 'top_symbols' in summary_data and summary_data['top_symbols']:
                top_symbols_text = "\n".join([f"{symbol}: ${pnl:.2f}" for symbol, pnl in summary_data['top_symbols'].items()])
                embed["fields"].append({"name": "Top Performing Symbols", "value": top_symbols_text, "inline": False})
            
            # Send notification
            return self._send_discord_message(
                f"Option Hunter Daily Summary for {date}: {winning_trades}/{total_trades} winning trades, P&L: ${total_pnl:.2f}",
                embed,
                file
            )
            
        except Exception as e:
            self.logger.error(f"Error creating daily summary notification: {str(e)}")
            return False
    
    def send_error_notification(self, title, error_message, stack_trace=None):
        """
        Send an error notification.
        
        Args:
            title (str): Error title
            error_message (str): Error message
            stack_trace (str, optional): Stack trace of the error
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enable_error_notifications:
            return False
        
        try:
            embed = {
                "title": f"❌ Error: {title}",
                "color": 15158332,  # Red
                "description": error_message,
                "timestamp": datetime.now().isoformat()
            }
            
            if stack_trace:
                # Truncate stack trace if it's too long for Discord
                if len(stack_trace) > 1000:
                    stack_trace = stack_trace[:997] + "..."
                
                embed["fields"] = [
                    {"name": "Stack Trace", "value": f"```\n{stack_trace}\n```", "inline": False}
                ]
            
            return self._send_discord_message("❌ Error Alert", embed)
            
        except Exception as e:
            self.logger.error(f"Error creating error notification: {str(e)}")
            return False
    
    def send_system_notification(self, title, message):
        """
        Send a system notification.
        
        Args:
            title (str): Notification title
            message (str): Notification message
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            embed = {
                "title": f"🤖 {title}",
                "color": 3447003,  # Blue
                "description": message,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._send_discord_message("System Notification", embed)
            
        except Exception as e:
            self.logger.error(f"Error creating system notification: {str(e)}")
            return False
    
    def send_parameter_update_notification(self, old_params, new_params, update_reason=None):
        """
        Send a notification about parameter updates.
        
        Args:
            old_params (dict): Old parameter values
            new_params (dict): New parameter values
            update_reason (str, optional): Reason for the parameter update
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Create fields for parameters that changed
            fields = []
            for param, new_value in new_params.items():
                if param in old_params and old_params[param] != new_value:
                    fields.append({
                        "name": param,
                        "value": f"Old: {old_params[param]}\nNew: {new_value}",
                        "inline": True
                    })
            
            # Only send if parameters actually changed
            if not fields:
                return False
            
            embed = {
                "title": "⚙️ Trading Parameters Updated",
                "color": 10181046,  # Purple
                "fields": fields,
                "timestamp": datetime.now().isoformat()
            }
            
            if update_reason:
                embed["description"] = f"Reason: {update_reason}"
            
            return self._send_discord_message("Trading Parameters Updated", embed)
            
        except Exception as e:
            self.logger.error(f"Error creating parameter update notification: {str(e)}")
            return False
    
    def send_market_regime_notification(self, new_regime, previous_regime=None, confidence=None):
        """
        Send a notification about a market regime change.
        
        Args:
            new_regime (str): New market regime
            previous_regime (str, optional): Previous market regime
            confidence (float, optional): Confidence in the regime detection (0-1)
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            title = "Market Regime Detected" if not previous_regime else "Market Regime Change"
            
            description = f"Current Market Regime: **{new_regime}**"
            if previous_regime:
                description += f"\nPrevious Regime: {previous_regime}"
            if confidence:
                description += f"\nConfidence: {confidence * 100:.2f}%"
            
            # Set color based on regime
            color = 10181046  # Default purple
            if new_regime.lower() == 'bullish':
                color = 3066993  # Green
            elif new_regime.lower() == 'bearish':
                color = 15158332  # Red
            elif new_regime.lower() == 'volatile':
                color = 16776960  # Yellow
            
            embed = {
                "title": f"📈 {title}",
                "color": color,
                "description": description,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._send_discord_message("Market Regime Update", embed)
            
        except Exception as e:
            self.logger.error(f"Error creating market regime notification: {str(e)}")
            return False
    
    def reset_streaks(self):
        """Reset win and loss streaks."""
        self.win_streak = 0
        self.loss_streak = 0
        self.logger.debug("Win and loss streaks reset")
