"sharpe_ratio": 0.0,
            "drawdowns": [],
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "drawdown_start_date": None,
            "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Risk metrics
        self.risk_metrics = {
            "portfolio_delta": 0.0,
            "portfolio_gamma": 0.0,
            "portfolio_theta": 0.0,
            "portfolio_vega": 0.0,
            "portfolio_beta": 0.0,
            "value_at_risk": 0.0,
            "position_concentration": 0.0,
            "option_concentration": 0.0,
            "sector_allocation": {}
        }
        
        # Directory for portfolio data storage
        self.data_dir = "data/portfolio"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load portfolio if exists
        self._load_portfolio()
        
        # Backup schedule
        self.last_backup_time = time.time()
        self.backup_interval = self.config.get("backup_interval_minutes", 30) * 60  # Convert to seconds
        
        self.logger.info(f"PortfolioUpdater initialized with ${self.portfolio['account_value']:.2f} account value")
    
    def _load_portfolio(self):
        """Load portfolio data from disk or Google Drive."""
        try:
            # Try local file first
            portfolio_file = f"{self.data_dir}/portfolio.json"
            if os.path.exists(portfolio_file):
                with open(portfolio_file, "r") as f:
                    loaded_portfolio = json.load(f)
                
                # Update portfolio with loaded data
                self.portfolio.update(loaded_portfolio)
                self.logger.info(f"Loaded portfolio data with ${self.portfolio['account_value']:.2f} account value")
                
                # Also load performance metrics
                metrics_file = f"{self.data_dir}/performance_metrics.json"
                if os.path.exists(metrics_file):
                    with open(metrics_file, "r") as f:
                        self.performance_metrics = json.load(f)
                
                # Load risk metrics
                risk_file = f"{self.data_dir}/risk_metrics.json"
                if os.path.exists(risk_file):
                    with open(risk_file, "r") as f:
                        self.risk_metrics = json.load(f)
                
            elif self.drive_manager and self.drive_manager.file_exists("portfolio/portfolio.json"):
                # Try Google Drive if local file not found
                content = self.drive_manager.download_file("portfolio/portfolio.json")
                loaded_portfolio = json.loads(content)
                
                # Update portfolio with loaded data
                self.portfolio.update(loaded_portfolio)
                
                # Save locally for future use
                with open(portfolio_file, "w") as f:
                    json.dump(self.portfolio, f, indent=2)
                
                self.logger.info(f"Loaded portfolio data from Google Drive with ${self.portfolio['account_value']:.2f} account value")
                
                # Also load performance metrics if available
                if self.drive_manager.file_exists("portfolio/performance_metrics.json"):
                    metrics_content = self.drive_manager.download_file("portfolio/performance_metrics.json")
                    self.performance_metrics = json.loads(metrics_content)
                    
                    with open(f"{self.data_dir}/performance_metrics.json", "w") as f:
                        f.write(metrics_content)
                
                # Load risk metrics if available
                if self.drive_manager.file_exists("portfolio/risk_metrics.json"):
                    risk_content = self.drive_manager.download_file("portfolio/risk_metrics.json")
                    self.risk_metrics = json.loads(risk_content)
                    
                    with open(f"{self.data_dir}/risk_metrics.json", "w") as f:
                        f.write(risk_content)
                
        except Exception as e:
            self.logger.error(f"Error loading portfolio data: {str(e)}")
            self.logger.info("Starting with new portfolio")
    
    def _save_portfolio(self, force=False):
        """
        Save portfolio data to disk and Google Drive.
        
        Args:
            force (bool): If True, save regardless of backup interval
        """
        current_time = time.time()
        time_since_backup = current_time - self.last_backup_time
        
        # Only backup at specified intervals unless forced
        if not force and time_since_backup < self.backup_interval:
            return
        
        try:
            # Save portfolio to local file
            portfolio_file = f"{self.data_dir}/portfolio.json"
            with open(portfolio_file, "w") as f:
                json.dump(self.portfolio, f, indent=2)
            
            # Save performance metrics
            metrics_file = f"{self.data_dir}/performance_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            # Save risk metrics
            risk_file = f"{self.data_dir}/risk_metrics.json"
            with open(risk_file, "w") as f:
                json.dump(self.risk_metrics, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "portfolio/portfolio.json",
                    json.dumps(self.portfolio, indent=2),
                    mime_type="application/json"
                )
                
                self.drive_manager.upload_file(
                    "portfolio/performance_metrics.json",
                    json.dumps(self.performance_metrics, indent=2),
                    mime_type="application/json"
                )
                
                self.drive_manager.upload_file(
                    "portfolio/risk_metrics.json",
                    json.dumps(self.risk_metrics, indent=2),
                    mime_type="application/json"
                )
            
            self.last_backup_time = current_time
            self.logger.debug("Saved portfolio data")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio data: {str(e)}")
    
    def add_position(self, trade_data):
        """
        Add a new position to the portfolio.
        
        Args:
            trade_data (dict): Trade data including symbol, quantity, price, etc.
            
        Returns:
            str: Position ID
        """
        try:
            # Generate position ID if not provided
            position_id = trade_data.get("trade_id", f"pos_{uuid.uuid4().hex[:8]}")
            
            # Check if position already exists
            if position_id in self.portfolio["positions"]:
                self.logger.warning(f"Position {position_id} already exists, updating instead")
                return self.update_position(position_id, trade_data)
            
            # Calculate position cost
            entry_price = trade_data.get("entry_price", 0)
            quantity = trade_data.get("quantity", 0)
            position_size = entry_price * quantity * 100  # Options are per contract (100 shares)
            
            # Check if we have enough cash
            if position_size > self.portfolio["cash"]:
                self.logger.warning(f"Insufficient cash (${self.portfolio['cash']:.2f}) for position size ${position_size:.2f}")
                # Still allow position if within a small margin of error (1%)
                if position_size > self.portfolio["cash"] * 1.01:
                    return None
            
            # Create new position
            position = {
                "position_id": position_id,
                "symbol": trade_data.get("symbol"),
                "option_symbol": trade_data.get("option_symbol"),
                "option_type": trade_data.get("option_type"),
                "strike": trade_data.get("strike"),
                "expiration": trade_data.get("expiration"),
                "entry_price": entry_price,
                "current_price": entry_price,
                "quantity": quantity,
                "position_size": position_size,
                "current_value": position_size,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_percent": 0.0,
                "realized_pnl": 0.0,
                "entry_time": trade_data.get("entry_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "days_held": 0,
                "strategy": trade_data.get("strategy", "unknown"),
                "sector": trade_data.get("sector", "unknown"),
                "greeks": trade_data.get("greeks", {}),
                "position_delta": trade_data.get("greeks", {}).get("delta", 0) * quantity * 100,
                "position_gamma": trade_data.get("greeks", {}).get("gamma", 0) * quantity * 100,
                "position_theta": trade_data.get("greeks", {}).get("theta", 0) * quantity * 100,
                "position_vega": trade_data.get("greeks", {}).get("vega", 0) * quantity * 100,
                "status": "open",
                "notes": trade_data.get("notes", ""),
                "history": []
            }
            
            # Add position to portfolio
            self.portfolio["positions"][position_id] = position
            
            # Update cash balance
            self.portfolio["cash"] -= position_size
            
            # Add to cash history
            self.portfolio["cash_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": -position_size,
                "type": "position_open",
                "position_id": position_id
            })
            
            # Update account value (doesn't change on opening a position)
            self._update_account_value()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Save portfolio
            self._save_portfolio()
            
            self.logger.info(f"Added new position {position_id}: {position['option_symbol']} with ${position_size:.2f} investment")
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error adding position: {str(e)}")
            return None
    
    def update_position(self, position_id, update_data):
        """
        Update an existing position with new data.
        
        Args:
            position_id (str): Position identifier
            update_data (dict): Updated position data
            
        Returns:
            bool: Success flag
        """
        if position_id not in self.portfolio["positions"]:
            self.logger.warning(f"Position {position_id} not found for update")
            return False
        
        try:
            position = self.portfolio["positions"][position_id]
            
            # Update price and calculate P&L
            if "current_price" in update_data:
                current_price = update_data["current_price"]
                position["current_price"] = current_price
                
                # Calculate current value and unrealized P&L
                current_value = current_price * position["quantity"] * 100
                position["current_value"] = current_value
                
                position["unrealized_pnl"] = current_value - position["position_size"]
                if position["position_size"] > 0:
                    position["unrealized_pnl_percent"] = (position["unrealized_pnl"] / position["position_size"]) * 100
            
            # Update Greeks
            if "greeks" in update_data:
                position["greeks"] = update_data["greeks"]
                position["position_delta"] = update_data["greeks"].get("delta", 0) * position["quantity"] * 100
                position["position_gamma"] = update_data["greeks"].get("gamma", 0) * position["quantity"] * 100
                position["position_theta"] = update_data["greeks"].get("theta", 0) * position["quantity"] * 100
                position["position_vega"] = update_data["greeks"].get("vega", 0) * position["quantity"] * 100
            
            # Update days held
            if "entry_time" in position:
                try:
                    entry_time = datetime.strptime(position["entry_time"], "%Y-%m-%d %H:%M:%S")
                    days_held = (datetime.now() - entry_time).total_seconds() / 86400  # Convert to days
                    position["days_held"] = round(days_held, 1)
                except (ValueError, TypeError):
                    pass
            
            # Add to position history
            if "current_price" in update_data or "greeks" in update_data:
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": position["current_price"],
                    "value": position["current_value"],
                    "unrealized_pnl": position["unrealized_pnl"],
                    "unrealized_pnl_percent": position["unrealized_pnl_percent"]
                }
                
                # Add Greeks to history if available
                if "greeks" in update_data:
                    history_entry["greeks"] = update_data["greeks"]
                
                # Add to history
                position["history"].append(history_entry)
                
                # Limit history length
                if len(position["history"]) > 100:
                    position["history"] = position["history"][-100:]
            
            # Update other fields if provided
            for key, value in update_data.items():
                if key not in ["current_price", "greeks", "history"]:
                    position[key] = value
            
            # Update account value
            self._update_account_value()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Save portfolio if significant changes
            if "current_price" in update_data or "greeks" in update_data:
                self._save_portfolio()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating position {position_id}: {str(e)}")
            return False
    
    def close_position(self, position_id, exit_data):
        """
        Close an existing position.
        
        Args:
            position_id (str): Position identifier
            exit_data (dict): Exit data including price, time, etc.
            
        Returns:
            dict: Closed position data
        """
        if position_id not in self.portfolio["positions"]:
            self.logger.warning(f"Position {position_id} not found for closing")
            return None
        
        try:
            position = self.portfolio["positions"][position_id]
            
            # Calculate final P&L
            exit_price = exit_data.get("exit_price", position["current_price"])
            exit_value = exit_price * position["quantity"] * 100
            realized_pnl = exit_value - position["position_size"]
            
            if position["position_size"] > 0:
                realized_pnl_percent = (realized_pnl / position["position_size"]) * 100
            else:
                realized_pnl_percent = 0.0
            
            # Update position with exit information
            position["exit_price"] = exit_price
            position["exit_value"] = exit_value
            position["realized_pnl"] = realized_pnl
            position["realized_pnl_percent"] = realized_pnl_percent
            position["exit_time"] = exit_data.get("exit_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            position["exit_reason"] = exit_data.get("exit_reason", "unknown")
            position["status"] = "closed"
            
            # Calculate holding period
            try:
                entry_time = datetime.strptime(position["entry_time"], "%Y-%m-%d %H:%M:%S")
                exit_time = datetime.strptime(position["exit_time"], "%Y-%m-%d %H:%M:%S")
                holding_period = (exit_time - entry_time).total_seconds() / 3600  # Hours
                position["holding_period_hours"] = round(holding_period, 1)
            except (ValueError, TypeError):
                position["holding_period_hours"] = 0
            
            # Add final entry to position history
            position["history"].append({
                "timestamp": position["exit_time"],
                "price": exit_price,
                "value": exit_value,
                "realized_pnl": realized_pnl,
                "realized_pnl_percent": realized_pnl_percent,
                "exit": True
            })
            
            # Update cash balance
            self.portfolio["cash"] += exit_value
            
            # Add to cash history
            self.portfolio["cash_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": exit_value,
                "type": "position_close",
                "position_id": position_id,
                "pnl": realized_pnl
            })
            
            # Move position to closed positions
            self.portfolio["closed_positions"][position_id] = position
            del self.portfolio["positions"][position_id]
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            # Update account value
            self._update_account_value()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Save portfolio
            self._save_portfolio(force=True)
            
            self.logger.info(f"Closed position {position_id} with P&L: ${realized_pnl:.2f} ({realized_pnl_percent:.2f}%)")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {str(e)}")
            return None
    
    def _update_account_value(self):
        """Update the total account value based on current positions and cash."""
        try:
            # Calculate total value of open positions
            positions_value = sum(pos["current_value"] for pos in self.portfolio["positions"].values())
            
            # Account value is cash plus positions
            new_account_value = self.portfolio["cash"] + positions_value
            
            # Calculate unrealized P&L
            unrealized_pnl = sum(pos["unrealized_pnl"] for pos in self.portfolio["positions"].values())
            
            # Check for significant change (to log)
            prev_value = self.portfolio["account_value"]
            if abs(new_account_value - prev_value) > (prev_value * 0.005):  # 0.5% change
                self.logger.info(f"Account value changed: ${prev_value:.2f} -> ${new_account_value:.2f} (${new_account_value - prev_value:.2f})")
            
            # Update account value
            self.portfolio["account_value"] = new_account_value
            
            # Add to equity history
            self.portfolio["equity_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "account_value": new_account_value,
                "cash": self.portfolio["cash"],
                "positions_value": positions_value,
                "unrealized_pnl": unrealized_pnl
            })
            
            # Limit history length
            if len(self.portfolio["equity_history"]) > 500:
                self.portfolio["equity_history"] = self.portfolio["equity_history"][-500:]
            
            # Update performance metrics for current drawdown
            self._check_drawdown()
            
            return new_account_value
            
        except Exception as e:
            self.logger.error(f"Error updating account value: {str(e)}")
            return self.portfolio["account_value"]
    
    def _update_performance_metrics(self, closed_position=None):
        """
        Update portfolio performance metrics.
        
        Args:
            closed_position (dict, optional): Recently closed position
        """
        try:
            # Update trade statistics if position closed
            if closed_position:
                self.performance_metrics["total_trades"] += 1
                self.performance_metrics["realized_pnl"] += closed_position["realized_pnl"]
                
                if closed_position["realized_pnl"] > 0:
                    self.performance_metrics["winning_trades"] += 1
                    self.performance_metrics["average_win"] = ((self.performance_metrics["average_win"] * 
                                                              (self.performance_metrics["winning_trades"] - 1) + 
                                                              closed_position["realized_pnl"]) / 
                                                             self.performance_metrics["winning_trades"])
                else:
                    self.performance_metrics["losing_trades"] += 1
                    self.performance_metrics["average_loss"] = ((self.performance_metrics["average_loss"] * 
                                                               (self.performance_metrics["losing_trades"] - 1) + 
                                                               closed_position["realized_pnl"]) / 
                                                              self.performance_metrics["losing_trades"])
            
            # Calculate win rate
            if self.performance_metrics["total_trades"] > 0:
                self.performance_metrics["win_rate"] = (self.performance_metrics["winning_trades"] / 
                                                      self.performance_metrics["total_trades"] * 100)
            
            # Calculate profit factor
            total_wins = self.performance_metrics["winning_trades"] * self.performance_metrics["average_win"]
            total_losses = abs(self.performance_metrics["losing_trades"] * self.performance_metrics["average_loss"])
            
            if total_losses > 0:
                self.performance_metrics["profit_factor"] = total_wins / total_losses
            elif total_wins > 0:
                self.performance_metrics["profit_factor"] = float('inf')  # No losses but has wins
            else:
                self.performance_metrics["profit_factor"] = 0.0
            
            # Calculate unrealized P&L
            self.performance_metrics["unrealized_pnl"] = sum(pos["unrealized_pnl"] 
                                                          for pos in self.portfolio["positions"].values())
            
            # Update Sharpe ratio if we have enough daily returns
            if len(self.performance_metrics["daily_returns"]) > 30:
                returns = np.array(self.performance_metrics["daily_returns"][-90:])  # Use last 90 days
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    self.performance_metrics["sharpe_ratio"] = (avg_return / std_return) * np.sqrt(252)  # Annualized
            
            # Update last update time
            self.performance_metrics["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _update_risk_metrics(self):
        """Update portfolio risk metrics based on current positions."""
        try:
            positions = self.portfolio["positions"].values()
            
            # Reset values
            portfolio_delta = 0.0
            portfolio_gamma = 0.0
            portfolio_theta = 0.0
            portfolio_vega = 0.0
            sector_allocation = {}
            
            # Calculate portfolio Greeks
            for position in positions:
                portfolio_delta += position.get("position_delta", 0)
                portfolio_gamma += position.get("position_gamma", 0)
                portfolio_theta += position.get("position_theta", 0)
                portfolio_vega += position.get("position_vega", 0)
                
                # Update sector allocation
                sector = position.get("sector", "unknown")
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += position.get("current_value", 0)
            
            # Calculate sector percentages
            if self.portfolio["account_value"] > 0:
                for sector in sector_allocation:
                    sector_allocation[sector] = (sector_allocation[sector] / self.portfolio["account_value"]) * 100
            
            # Calculate position concentration (Herfindahl-Hirschman Index)
            if positions:
                position_values = [pos.get("current_value", 0) for pos in positions]
                total_value = sum(position_values)
                
                if total_value > 0:
                    position_weights = [val / total_value for val in position_values]
                    hhi = sum(w * w for w in position_weights) * 10000  # Scale to 0-10000
                    self.risk_metrics["position_concentration"] = hhi
            
            # Calculate option type concentration
            call_value = sum(pos.get("current_value", 0) for pos in positions 
                            if pos.get("option_type", "") == "call")
            put_value = sum(pos.get("current_value", 0) for pos in positions 
                           if pos.get("option_type", "") == "put")
            
            total_option_value = call_value + put_value
            if total_option_value > 0:
                call_pct = (call_value / total_option_value) * 100
                put_pct = (put_value / total_option_value) * 100
                
                # Store as dictionary
                self.risk_metrics["option_concentration"] = {
                    "call_percentage": call_pct,
                    "put_percentage": put_pct,
                    "call_value": call_value,
                    "put_value": put_value
                }
            
            # Update risk metrics
            self.risk_metrics["portfolio_delta"] = portfolio_delta
            self.risk_metrics["portfolio_gamma"] = portfolio_gamma
            self.risk_metrics["portfolio_theta"] = portfolio_theta
            self.risk_metrics["portfolio_vega"] = portfolio_vega
            self.risk_metrics["sector_allocation"] = sector_allocation
            
            # Add to margin history
            margin_used = total_option_value if 'total_option_value' in locals() else 0
            margin_available = self.portfolio["cash"]
            
            self.portfolio["margin_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "margin_used": margin_used,
                "margin_available": margin_available,
                "margin_utilization": (margin_used / (margin_used + margin_available)) * 100 if (margin_used + margin_available) > 0 else 0
            })
            
            # Limit history length
            if len(self.portfolio["margin_history"]) > 500:
                self.portfolio["margin_history"] = self.portfolio["margin_history"][-500:]
            
            # Add to allocation history
            self.portfolio["allocation_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "positions_count": len(positions),
                "total_allocation": (sum(pos.get("current_value", 0) for pos in positions) / 
                                   self.portfolio["account_value"]) * 100 if self.portfolio["account_value"] > 0 else 0,
                "sector_allocation": sector_allocation
            })
            
            # Limit history length
            if len(self.portfolio["allocation_history"]) > 500:
                self.portfolio["allocation_history"] = self.portfolio["allocation_history"][-500:]
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
    
    def _check_drawdown(self):
        """Calculate and track portfolio drawdown."""
        try:
            # Get current equity
            current_equity = self.portfolio["account_value"]
            
            # Get equity history if available
            if self.portfolio["equity_history"]:
                # Find peak equity
                equity_values = [entry["account_value"] for entry in self.portfolio["equity_history"]]
                peak_equity = max(equity_values)
                
                # Calculate current drawdown
                if peak_equity > 0:
                    current_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
                else:
                    current_drawdown_pct = 0
                
                # Update current drawdown
                self.performance_metrics["current_drawdown"] = current_drawdown_pct
                
                # Check if this is a new drawdown
                if current_drawdown_pct > 0:
                    if self.performance_metrics["drawdown_start_date"] is None:
                        self.performance_metrics["drawdown_start_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Check if this is a new max drawdown
                    if current_drawdown_pct > self.performance_metrics["max_drawdown"]:
                        self.performance_metrics["max_drawdown"] = current_drawdown_pct
                        
                        # Log significant drawdowns
                        if current_drawdown_pct > 5:
                            self.logger.warning(f"New maximum drawdown: {current_drawdown_pct:.2f}%")
                else:
                    # Reset drawdown tracking if back to peak
                    if self.performance_metrics["drawdown_start_date"] is not None:
                        # Record the drawdown
                        self.performance_metrics["drawdowns"].append({
                            "start_date": self.performance_metrics["drawdown_start_date"],
                            "end_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "max_drawdown": self.performance_metrics["max_drawdown"]
                        })
                        
                        # Limit drawdown history
                        if len(self.performance_metrics["drawdowns"]) > 20:
                            self.performance_metrics["drawdowns"] = self.performance_metrics["drawdowns"][-20:]
                        
                        # Reset tracking
                        self.performance_metrics["drawdown_start_date"] = None
                        self.performance_metrics["current_drawdown"] = 0
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {str(e)}")
    
    def update_daily_returns(self):
        """Calculate and store daily returns for performance metrics."""
        try:
            # Get current equity
            current_equity = self.portfolio["account_value"]
            
            # Get previous day's equity from history
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            previous_equity = None
            for entry in reversed(self.portfolio["equity_history"]):
                entry_date = entry["timestamp"].split()[0]  # Get just the date part
                if entry_date <= yesterday:
                    previous_equity = entry["account_value"]
                    break
            
            # If no previous value, use current as first entry
            if previous_equity is None:
                # Don't record a return if we don't have a previous value
                return
            
            # Calculate daily return
            if previous_equity > 0:
                daily_return = ((current_equity - previous_equity) / previous_equity) * 100
                
                # Add to daily returns history
                self.performance_metrics["daily_returns"].append(daily_return)
                
                # Limit history length
                if len(self.performance_metrics["daily_returns"]) > 252:  # One trading year
                    self.performance_metrics["daily_returns"] = self.performance_metrics["daily_returns"][-252:]
                
                self.logger.debug(f"Recorded daily return: {daily_return:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error updating daily returns: {str(e)}")
    
    def add_cash(self, amount, description="Deposit"):
        """
        Add cash to the portfolio.
        
        Args:
            amount (float): Amount to add
            description (str): Description of transaction
            
        Returns:
            bool: Success flag
        """
        try:
            # Validate amount
            if amount <= 0:
                self.logger.warning(f"Cash deposit must be positive, got {amount}")
                return False
            
            # Add to cash balance
            self.portfolio["cash"] += amount
            
            # Add to cash history
            self.portfolio["cash_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "type": "deposit",
                "description": description
            })
            
            # Update account value
            self._update_account_value()
            
            # Save portfolio
            self._save_portfolio()
            
            self.logger.info(f"Added ${amount:.2f} cash to portfolio: {description}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding cash: {str(e)}")
            return False
    
    def remove_cash(self, amount, description="Withdrawal"):
        """
        Remove cash from the portfolio.
        
        Args:
            amount (float): Amount to remove
            description (str): Description of transaction
            
        Returns:
            bool: Success flag
        """
        try:
            # Validate amount
            if amount <= 0:
                self.logger.warning(f"Cash withdrawal must be positive, got {amount}")
                return False
            
            # Check if we have enough cash
            if amount > self.portfolio["cash"]:
                self.logger.warning(f"Insufficient cash (${self.portfolio['cash']:.2f}) for withdrawal ${amount:.2f}")
                return False
            
            # Subtract from cash balance
            self.portfolio["cash"] -= amount
            
            # Add to cash history
            self.portfolio["cash_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": -amount,
                "type": "withdrawal",
                "description": description
            })
            
            # Update account value
            self._update_account_value()
            
            # Save portfolio
            self._save_portfolio()
            
            self.logger.info(f"Removed ${amount:.2f} cash from portfolio: {description}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing cash: {str(e)}")
            return False
    
    def get_position(self, position_id):
        """
        Get a specific position by ID.
        
        Args:
            position_id (str): Position identifier
            
        Returns:
            dict: Position data or None if not found
        """
        return self.portfolio["positions"].get(position_id)
    
    def get_closed_position(self, position_id):
        """
        Get a specific closed position by ID.
        
        Args:
            position_id (str): Position identifier
            
        Returns:
            dict: Position data or None if not found
        """
        return self.portfolio["closed_positions"].get(position_id)
    
    def get_all_positions(self, include_closed=False):
        """
        Get all positions in the portfolio.
        
        Args:
            include_closed (bool): If True, include closed positions
            
        Returns:
            dict: All positions
        """
        if include_closed:
            # Combine open and closed positions
            all_positions = {}
            all_positions.update(self.portfolio["positions"])
            all_positions.update(self.portfolio["closed_positions"])
            return all_positions
        else:
            return self.portfolio["positions"]
    
    def get_positions_by_symbol(self, symbol):
        """
        Get all positions for a specific symbol.
        
        Args:
            symbol (str): Symbol to filter by
            
        Returns:
            list: Positions for the symbol
        """
        return [pos for pos in self.portfolio["positions"].values() if pos["symbol"] == symbol]
    
    def get_positions_by_strategy(self, strategy):
        """
        Get all positions for a specific strategy.
        
        Args:
            strategy (str): Strategy to filter by
            
        Returns:
            list: Positions for the strategy
        """
        return [pos for pos in self.portfolio["positions"].values() if pos["strategy"] == strategy]
    
    def get_account_summary(self):
        """
        Get a summary of the account.
        
        Returns:
            dict: Account summary
        """
        # Calculate metrics
        positions_count = len(self.portfolio["positions"])
        positions_value = sum(pos["current_value"] for pos in self.portfolio["positions"].values())
        unrealized_pnl = sum(pos["unrealized_pnl"] for pos in self.portfolio["positions"].values())
        
        if positions_value > 0:
            unrealized_pnl_percent = (unrealized_pnl / positions_value) * 100
        else:
            unrealized_pnl_percent = 0.0
        
        # Calculate allocation percentage
        if self.portfolio["account_value"] > 0:
            allocation_percent = (positions_value / self.portfolio["account_value"]) * 100
        else:
            allocation_percent = 0.0
        
        # Calculate cash percentage
        if self.portfolio["account_value"] > 0:
            cash_percent = (self.portfolio["cash"] / self.portfolio["account_value"]) * 100
        else:
            cash_percent = 100.0
        
        # Get performance metrics
        performance = {
            "realized_pnl": self.performance_metrics["realized_pnl"],
            "win_rate": self.performance_metrics["win_rate"],
            "profit_factor": self.performance_metrics["profit_factor"],
            "max_drawdown": self.performance_metrics["max_drawdown"],
            "current_drawdown": self.performance_metrics["current_drawdown"],
            "sharpe_ratio": self.performance_metrics["sharpe_ratio"]
        }
        
        # Get risk metrics
        risk = {
            "portfolio_delta": self.risk_metrics["portfolio_delta"],
            "portfolio_gamma": self.risk_metrics["portfolio_gamma"],
            "portfolio_theta": self.risk_metrics["portfolio_theta"],
            "portfolio_vega": self.risk_metrics["portfolio_vega"],
            "position_concentration": self.risk_metrics["position_concentration"],
            "option_concentration": self.risk_metrics["option_concentration"],
            "top_sectors": dict(sorted(self.risk_metrics["sector_allocation"].items(), 
                                     key=lambda x: x[1], reverse=True)[:3])
        }
        
        # Create summary
        summary = {
            "account_value": self.portfolio["account_value"],
            "cash": self.portfolio["cash"],
            "cash_percent": cash_percent,
            "positions_value": positions_value,
            "positions_count": positions_count,
            "allocation_percent": allocation_percent,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": unrealized_pnl_percent,
            "performance_metrics": performance,
            "risk_metrics": risk,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary
    
    def get_performance_metrics(self):
        """
        Get portfolio performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return self.performance_metrics
    
    def get_risk_metrics(self):
        """
        Get portfolio risk metrics.
        
        Returns:
            dict: Risk metrics
        """
        return self.risk_metrics
    
    def get_portfolio_history(self, days=30):
        """
        Get portfolio history for a specific period.
        
        Args:
            days (int): Number of days of history
            
        Returns:
            dict: Portfolio history
        """
        # Calculate cutoff date
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Filter history by date
        equity_history = [entry for entry in self.portfolio["equity_history"] 
                          if entry["timestamp"].split()[0] >= cutoff]
        
        cash_history = [entry for entry in self.portfolio["cash_history"] 
                        if entry["timestamp"].split()[0] >= cutoff]
        
        margin_history = [entry for entry in self.portfolio["margin_history"] 
                          if entry["timestamp"].split()[0] >= cutoff]
        
        allocation_history = [entry for entry in self.portfolio["allocation_history"] 
                             if entry["timestamp"].split()[0] >= cutoff]
        
        return {
            "equity_history": equity_history,
            "cash_history": cash_history,
            "margin_history": margin_history,
            "allocation_history": allocation_history,
            "daily_returns": self.performance_metrics["daily_returns"][-days:] if len(self.performance_metrics["daily_returns"]) > 0 else []
        }
    
    def check_allocation_limits(self, new_position_size, symbol=None, sector=None):
        """
        Check if a new position would exceed allocation limits.
        
        Args:
            new_position_size (float): Size of the new position
            symbol (str, optional): Symbol of the new position
            sector (str, optional): Sector of the new position
            
        Returns:
            tuple: (within_limits, messages)
        """
        messages = []
        within_limits = True
        
        # Check if we have enough cash
        if new_position_size > self.portfolio["cash"]:
            messages.append(f"Insufficient cash (${self.portfolio['cash']:.2f}) for position size ${new_position_size:.2f}")
            within_limits = False
        
        # Check maximum position size limit
        position_limit = self.portfolio["account_value"] * self.max_allocation_per_position
        if new_position_size > position_limit:
            messages.append(f"Position size ${new_position_size:.2f} exceeds max position limit ${position_limit:.2f} ({self.max_allocation_per_position*100:.1f}%)")
            within_limits = False
        
        # Check total portfolio allocation limit
        current_allocation = sum(pos["current_value"] for pos in self.portfolio["positions"].values())
        new_total_allocation = current_allocation + new_position_size
        portfolio_limit = self.portfolio["account_value"] * self.max_portfolio_allocation
        
        if new_total_allocation > portfolio_limit:
            messages.append(f"New total allocation ${new_total_allocation:.2f} would exceed portfolio limit ${portfolio_limit:.2f} ({self.max_portfolio_allocation*100:.1f}%)")
            within_limits = False
        
        # Check symbol concentration if provided
        if symbol:
            symbol_positions = self.get_positions_by_symbol(symbol)
            symbol_allocation = sum(pos["current_value"] for pos in symbol_positions)
            new_symbol_allocation = symbol_allocation + new_position_size
            symbol_limit = self.portfolio["account_value"] * (self.max_allocation_per_position * 1.5)  # Allow slightly higher for same symbol
            
            if new_symbol_allocation > symbol_limit:
                messages.append(f"New allocation for {symbol} (${new_symbol_allocation:.2f}) would exceed symbol limit ${symbol_limit:.2f}")
                within_limits = False
        
        # Check sector concentration if provided
        if sector:
            sector_allocation = sum(pos["current_value"] for pos in self.portfolio["positions"].values() 
                                   if pos.get("sector") == sector)
            new_sector_allocation = sector_allocation + new_position_size
            sector_limit = self.portfolio["account_value"] * self.max_sector_allocation
            
            if new_sector_allocation > sector_limit:
                messages.append(f"New allocation for {sector} sector (${new_sector_allocation:.2f}) would exceed sector limit ${sector_limit:.2f} ({self.max_sector_allocation*100:.1f}%)")
                within_limits = False
        
        return within_limits, messages
    
    def reset_portfolio(self, initial_capital=None):
        """
        Reset the portfolio to initial state.
        
        Args:
            initial_capital (float, optional): New initial capital amount
            
        Returns:
            bool: Success flag
        """
        try:
            # Confirm reset, check if we have positions
            if self.portfolio["positions"]:
                self.logger.warning(f"Resetting portfolio with {len(self.portfolio['positions'])} open positions!")
            
            # Set initial capital
            if initial_capital is None:
                initial_capital = self.config.get("initial_capital", 100000.0)
            
            # Reset portfolio
            self.portfolio = {
                "account_value": initial_capital,
                "cash": initial_capital,
                "positions": {},
                "closed_positions": {},
                "cash_history": [{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": initial_capital,
                    "type": "initial_capital",
                    "description": "Portfolio reset"
                }],
                "equity_history": [{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "account_value": initial_capital,
                    "cash": initial_capital,
                    "positions_value": 0.0,
                    "unrealized_pnl": 0.0
                }],
                "margin_history": [],
                "allocation_history": []
            }
            
            # Reset performance metrics
            self.performance_metrics = {
                "daily_returns": [],
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "sharpe_ratio": 0.0,
                "drawdowns": [],
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "drawdown_start_date": None,
                "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Reset risk metrics
            self.risk_metrics = {
                "portfolio_delta": 0.0,
                "portfolio_gamma": 0.0,
                "portfolio_theta": 0.0,
                "portfolio_vega": 0.0,
                "portfolio_beta": 0.0,
                "value_at_risk": 0.0,
                "position_concentration": 0.0,
                "option_concentration": 0.0,
                "sector_allocation": {}
            }
            
            # Save reset portfolio
            self._save_portfolio(force=True)
            
            self.logger.info(f"Portfolio reset with ${initial_capital:.2f} initial capital")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting portfolio: {str(e)}")
            return False_metrics["daily_returns"] = self.performance"""
Portfolio Updater Module

This module tracks and manages the portfolio, updating positions, calculating
performance metrics, and handling trade accounting.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import uuid

class PortfolioUpdater:
    """
    Manages the portfolio by tracking positions, calculating performance metrics,
    and handling trade accounting operations.
    
    - Tracks open and closed positions
    - Updates position values and P&L
    - Calculates portfolio-level metrics
    - Handles cash management and margin requirements
    - Logs portfolio state for historical analysis
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the portfolio updater.
        
        Args:
            config (dict): Configuration dictionary
            drive_manager: Optional GoogleDriveManager for cloud backups
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("portfolio_updater", {})
        self.drive_manager = drive_manager
        
        # Initialize portfolio state
        self.portfolio = {
            "account_value": self.config.get("initial_capital", 100000.0),
            "cash": self.config.get("initial_capital", 100000.0),
            "positions": {},
            "closed_positions": {},
            "cash_history": [],
            "equity_history": [],
            "margin_history": [],
            "allocation_history": []
        }
        
        # Portfolio settings
        self.max_allocation_per_position = self.config.get("max_allocation_per_position", 0.05)
        self.max_portfolio_allocation = self.config.get("max_portfolio_allocation", 0.8)
        self.portfolio_hedge_target = self.config.get("portfolio_hedge_target", 0.0)  # Net delta target
        self.max_sector_allocation = self.config.get("max_sector_allocation", 0.3)
        
        # Performance metrics
        self.performance_metrics = {
            "daily_returns": [],
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "sharpe_ratio": 0.0,
            "drawdowns":
