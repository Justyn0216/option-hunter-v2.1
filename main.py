#!/usr/bin/env python3
"""
Option Hunter v2.1 - Advanced Options Trading System
Main Entry Point

This script initializes and runs the Option Hunter trading system,
coordinating all components and managing the trading lifecycle.
"""

import os
import json
import logging
import argparse
import time
from datetime import datetime
import pytz

# Import core components
from src.market_hours_awareness import MarketHoursAwareness
from src.integrated_bot import IntegratedBot
from src.master_parameter_hub import MasterParameterHub
from src.google_drive import GoogleDriveManager
from src.notification_system import NotificationSystem

def setup_logging(config):
    """Set up logging configuration."""
    log_level = getattr(logging, config["logging"]["log_level"])
    log_dir = config["logging"]["log_directory"]
    
    # Create log directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config["logging"]["trade_log_directory"], exist_ok=True)
    os.makedirs(config["logging"]["model_prediction_directory"], exist_ok=True)
    os.makedirs(config["logging"]["performance_log_directory"], exist_ok=True)
    os.makedirs(config["logging"]["regime_transition_directory"], exist_ok=True)
    
    # Set up logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() if config["logging"]["console_logging"] else logging.NullHandler(),
            logging.FileHandler(f"{log_dir}/option_hunter_{datetime.now().strftime('%Y%m%d')}.log") 
            if config["logging"]["file_logging"] else logging.NullHandler()
        ]
    )
    
    return logging.getLogger("option_hunter")

def load_config():
    """Load configuration from config.json."""
    with open("config.json", "r") as f:
        return json.load(f)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Option Hunter v2.1 - Advanced Options Trading System")
    parser.add_argument("--paper-trading", action="store_true", default=True, 
                        help="Run in paper trading mode (default: True)")
    parser.add_argument("--backtest", action="store_true", 
                        help="Run backtesting instead of live trading")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with additional logging")
    parser.add_argument("--ticker", type=str, 
                        help="Focus on a specific ticker for testing")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file (default: config.json)")
    parser.add_argument("--no-ml", action="store_true", 
                        help="Disable ML/RL components for faster startup")
    return parser.parse_args()

def main():
    """Main function to initialize and run the Option Hunter system."""
    args = parse_arguments()
    
    try:
        # Load configuration
        config = load_config()
        
        # Set up logging
        logger = setup_logging(config)
        
        # Override log level if debug mode is enabled
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        logger.info("Starting Option Hunter v2.1")
        
        # Initialize core components
        logger.info("Initializing Google Drive integration")
        drive_manager = GoogleDriveManager(
            config["api_credentials"]["google_drive"]["credentials_file"],
            config["api_credentials"]["google_drive"]["token_file"],
            config["api_credentials"]["google_drive"]["folder_id"]
        )
        
        logger.info("Initializing notification system")
        notification_system = NotificationSystem(
            config["api_credentials"]["discord"]["webhook_url"],
            config["notification"]
        )
        
        logger.info("Initializing market hours awareness")
        market_hours = MarketHoursAwareness(config["market_hours"])
        
        logger.info("Initializing master parameter hub")
        parameter_hub = MasterParameterHub(
            config,
            drive_manager,
            disable_ml=args.no_ml
        )
        
        # Initialize the integrated bot
        logger.info("Initializing integrated bot")
        bot = IntegratedBot(
            config=config,
            parameter_hub=parameter_hub,
            drive_manager=drive_manager,
            notification_system=notification_system,
            market_hours=market_hours,
            paper_trading=args.paper_trading,
            debug_mode=args.debug,
            focus_ticker=args.ticker
        )
        
        # Send startup notification
        notification_system.send_system_notification(
            "Option Hunter v2.1 Started",
            f"System initialized in {'paper trading' if args.paper_trading else 'live trading'} mode"
        )
        
        if args.backtest:
            # Run in backtest mode
            logger.info("Running in backtest mode")
            bot.run_backtest()
        else:
            # Run the main trading loop
            logger.info("Starting main trading loop")
            bot.run()
            
    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down...")
        notification_system.send_system_notification(
            "Option Hunter v2.1 Shutdown",
            "System was manually shut down"
        )
    except Exception as e:
        logger.exception(f"Fatal error occurred: {str(e)}")
        notification_system.send_error_notification(
            "Critical Error",
            f"Option Hunter encountered a fatal error: {str(e)}"
        )
        
    logger.info("Option Hunter v2.1 shutdown complete")

if __name__ == "__main__":
    main()
