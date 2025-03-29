"""
Market Hours Awareness Module

This module manages market hours and trading schedule awareness,
allowing the bot to operate appropriately during different market phases.
"""

import logging
import pytz
from datetime import datetime, time, timedelta

class MarketHoursAwareness:
    """
    Tracks market hours and provides methods to determine the current market phase.
    
    Supports:
    - Standard market hours
    - Pre-market and after-hours sessions
    - Weekend and holiday detection
    - Trading schedule management
    """
    
    # Market phases
    STANDARD_MARKET_HOURS = "STANDARD_MARKET_HOURS"
    PRE_MARKET = "PRE_MARKET"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"
    
    def __init__(self, config):
        """
        Initialize the MarketHoursAwareness object.
        
        Args:
            config (dict): Configuration dictionary with market hours settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Parse the configuration
        self.timezone = pytz.timezone(config["timezone"])
        self.open_time = datetime.strptime(config["open_time"], "%H:%M").time()
        self.close_time = datetime.strptime(config["close_time"], "%H:%M").time()
        self.pre_market_start = datetime.strptime(config["pre_market_start"], "%H:%M").time()
        self.after_hours_end = datetime.strptime(config["after_hours_end"], "%H:%M").time()
        self.weekend_days = config["weekend_days"]  # 0=Monday, 6=Sunday
        
        # Parse holidays (list of date strings in format YYYY-MM-DD)
        self.holidays = [datetime.strptime(date, "%Y-%m-%d").date() for date in config["holidays"]]
        
        self.logger.info(f"Market hours initialized: Open {self.open_time} to {self.close_time} {self.timezone}")
    
    def get_current_time(self):
        """Get the current time in the configured timezone."""
        return datetime.now(self.timezone)
    
    def get_current_phase(self):
        """
        Determine the current market phase.
        
        Returns:
            str: One of the market phase constants
        """
        current = self.get_current_time()
        current_date = current.date()
        current_time = current.time()
        current_weekday = current.weekday()
        
        # Check if today is a holiday
        if current_date in self.holidays:
            return self.HOLIDAY
        
        # Check if today is a weekend
        if current_weekday in self.weekend_days:
            return self.WEEKEND
        
        # Check market hours
        if self.open_time <= current_time < self.close_time:
            return self.STANDARD_MARKET_HOURS
        elif self.pre_market_start <= current_time < self.open_time:
            return self.PRE_MARKET
        elif self.close_time <= current_time < self.after_hours_end:
            return self.AFTER_HOURS
        else:
            return self.CLOSED
    
    def is_market_open(self, include_extended=False):
        """
        Check if the market is currently open.
        
        Args:
            include_extended (bool): If True, includes pre-market and after-hours sessions
                                    as "open" market conditions
        
        Returns:
            bool: True if market is open, False otherwise
        """
        phase = self.get_current_phase()
        
        if phase == self.STANDARD_MARKET_HOURS:
            return True
        elif include_extended and (phase == self.PRE_MARKET or phase == self.AFTER_HOURS):
            return True
        return False
    
    def time_until_market_open(self):
        """
        Calculate the time until the market opens.
        
        Returns:
            timedelta: Time until market open, or None if market is already open
        """
        if self.is_market_open():
            return None
        
        current = self.get_current_time()
        current_date = current.date()
        
        # Create datetime objects for today's open time
        open_datetime = self.timezone.localize(
            datetime.combine(current_date, self.open_time)
        )
        
        # If it's past the open time, look at the next trading day
        if current.time() > self.open_time:
            next_trading_day = self._get_next_trading_day(current_date)
            open_datetime = self.timezone.localize(
                datetime.combine(next_trading_day, self.open_time)
            )
        
        return open_datetime - current
    
    def time_until_market_close(self):
        """
        Calculate the time until the market closes.
        
        Returns:
            timedelta: Time until market close, or None if market is already closed
        """
        if not self.is_market_open():
            return None
        
        current = self.get_current_time()
        current_date = current.date()
        
        # Create datetime object for today's close time
        close_datetime = self.timezone.localize(
            datetime.combine(current_date, self.close_time)
        )
        
        return close_datetime - current
    
    def _get_next_trading_day(self, start_date):
        """
        Find the next trading day (not weekend, not holiday).
        
        Args:
            start_date (date): Starting date
        
        Returns:
            date: Next trading day
        """
        next_date = start_date + timedelta(days=1)
        
        # Keep checking until we find a trading day
        while True:
            if next_date.weekday() not in self.weekend_days and next_date not in self.holidays:
                return next_date
            next_date += timedelta(days=1)
    
    def should_trade_now(self, trade_in_extended_hours=False):
        """
        Determine if the system should be actively trading now.
        
        Args:
            trade_in_extended_hours (bool): Whether to trade during extended hours
        
        Returns:
            bool: True if trading should be active, False otherwise
        """
        phase = self.get_current_phase()
        
        if phase == self.STANDARD_MARKET_HOURS:
            return True
        elif trade_in_extended_hours and (phase == self.PRE_MARKET or phase == self.AFTER_HOURS):
            return True
        return False
    
    def get_trading_day_progress(self):
        """
        Calculate the progress through the trading day as a percentage.
        
        Returns:
            float: Percentage of the trading day that has elapsed (0-100),
                  or None if market is closed
        """
        if not self.is_market_open():
            return None
        
        current = self.get_current_time()
        
        # Convert times to seconds since midnight for calculation
        current_seconds = current.hour * 3600 + current.minute * 60 + current.second
        open_seconds = self.open_time.hour * 3600 + self.open_time.minute * 60
        close_seconds = self.close_time.hour * 3600 + self.close_time.minute * 60
        
        # Calculate progress
        total_seconds = close_seconds - open_seconds
        elapsed_seconds = current_seconds - open_seconds
        
        progress = (elapsed_seconds / total_seconds) * 100
        return max(0, min(100, progress))  # Clamp between 0 and 100

    def get_next_market_event(self):
        """
        Get information about the next market event (open, close, etc.).
        
        Returns:
            dict: Dictionary with event type, time, and description
        """
        current_phase = self.get_current_phase()
        current = self.get_current_time()
        current_date = current.date()
        current_time = current.time()
        
        event = {
            "type": None,
            "time": None,
            "description": None
        }
        
        if current_phase == self.STANDARD_MARKET_HOURS:
            # Next event is market close
            event["type"] = "close"
            event["time"] = self.timezone.localize(
                datetime.combine(current_date, self.close_time)
            )
            event["description"] = "Market Close"
        
        elif current_phase == self.PRE_MARKET:
            # Next event is market open
            event["type"] = "open"
            event["time"] = self.timezone.localize(
                datetime.combine(current_date, self.open_time)
            )
            event["description"] = "Market Open"
        
        elif current_phase == self.AFTER_HOURS:
            # Next event is after hours end
            event["type"] = "after_hours_end"
            event["time"] = self.timezone.localize(
                datetime.combine(current_date, self.after_hours_end)
            )
            event["description"] = "After Hours End"
        
        else:  # CLOSED, WEEKEND, or HOLIDAY
            # Find the next trading day
            next_trading_day = self._get_next_trading_day(
                current_date if current_time >= self.after_hours_end else current_date
            )
            
            # Next event is pre-market start
            event["type"] = "pre_market_start"
            event["time"] = self.timezone.localize(
                datetime.combine(next_trading_day, self.pre_market_start)
            )
            event["description"] = "Pre-Market Start"
        
        return event
