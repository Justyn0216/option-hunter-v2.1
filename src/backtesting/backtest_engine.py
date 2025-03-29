def _run_backtest_simulation(self, backtest_state, options_data, symbols):
        """
        Run the day-by-day backtest simulation.
        
        Args:
            backtest_state (dict): Current state of the backtest
            options_data (dict): Historical options data
            symbols (list): List of symbols to test
        """
        self.logger.info("Running backtest simulation")
        
        # Set up progress tracking
        total_days = (backtest_state["end_date"] - backtest_state["current_date"]).days + 1
        
        # Run through each day
        for _ in tqdm(range(total_days), desc="Backtesting"):
            date_str = backtest_state["current_date"].strftime('%Y-%m-%d')
            
            # Skip weekends and holidays (no data)
            if date_str not in options_data or not options_data[date_str]["underlying"]:
                backtest_state["current_date"] += timedelta(days=1)
                continue
            
            # Set current data
            backtest_state["current_data"] = options_data[date_str]
            
            # 1. Update positions first (check for exits)
            self._update_positions(backtest_state)
            
            # 2. Look for new entries
            self._find_new_entries(backtest_state, symbols)
            
            # 3. Record daily equity
            self._record_daily_equity(backtest_state, date_str)
            
            # Move to next day
            backtest_state["current_date"] += timedelta(days=1)
    
    def _update_positions(self, backtest_state):
        """
        Update existing positions and check for exits.
        
        Args:
            backtest_state (dict): Current state of the backtest
        """
        current_date = backtest_state["current_date"]
        current_data = backtest_state["current_data"]
        positions = backtest_state["positions"]
        
        # Check each position
        for position_id in list(positions.keys()):
            position = positions[position_id]
            symbol = position["symbol"]
            option_symbol = position["option_symbol"]
            option_type = position["option_type"]
            
            # Check if we have data for this symbol
            if symbol not in current_data["underlying"]:
                continue
            
            # Find the option in current data
            option_data = None
            if symbol in current_data["options"]:
                option_chain = current_data["options"][symbol]
                for option in option_chain:
                    if option["symbol"] == option_symbol:
                        option_data = option
                        break
            
            # If option not found, might be due to expiration or data issue
            if option_data is None:
                # Check if the option has expired
                exp_date = datetime.strptime(position["expiration"], '%Y-%m-%d').date()
                if current_date.date() >= exp_date:
                    # Option expired, calculate settlement value
                    self._handle_option_expiration(backtest_state, position)
                continue
            
            # Update position with current market data
            position["current_price"] = (option_data["bid"] + option_data["ask"]) / 2
            position["current_underlying_price"] = current_data["underlying"][symbol]["last"]
            
            # Update Greeks if available
            if "greeks" in option_data:
                position["current_greeks"] = option_data["greeks"]
            
            # Calculate current position value and P&L
            position["current_value"] = position["current_price"] * position["quantity"] * 100
            position["unrealized_pnl"] = position["current_value"] - position["cost_basis"]
            position["unrealized_pnl_pct"] = (position["current_price"] / position["entry_price"] - 1) * 100
            
            # Check exit conditions
            should_exit, exit_reason = self._check_exit_conditions(position, option_data)
            
            if should_exit:
                self._exit_position(backtest_state, position_id, exit_reason)
    
    def _handle_option_expiration(self, backtest_state, position):
        """
        Handle expiration of an option position.
        
        Args:
            backtest_state (dict): Current state of the backtest
            position (dict): Position that has expired
        """
        # Get underlying price
        symbol = position["symbol"]
        if symbol not in backtest_state["current_data"]["underlying"]:
            # No underlying data, assume worthless expiration
            self._exit_position(backtest_state, position["position_id"], "Expired worthless")
            return
        
        underlying_price = backtest_state["current_data"]["underlying"][symbol]["last"]
        strike_price = position["strike"]
        option_type = position["option_type"]
        
        # Calculate intrinsic value at expiration
        if option_type == "call":
            intrinsic_value = max(0, underlying_price - strike_price)
        else:  # put
            intrinsic_value = max(0, strike_price - underlying_price)
        
        # Set exit price to intrinsic value
        position["exit_price"] = intrinsic_value
        
        # Exit the position
        self._exit_position(backtest_state, position["position_id"], "Expired")
    
    def _check_exit_conditions(self, position, option_data):
        """
        Check if any exit conditions are met for a position.
        
        Args:
            position (dict): Position to check
            option_data (dict): Current option data
            
        Returns:
            tuple: (should_exit, exit_reason)
        """
        # 1. Check stop loss
        if "stop_loss" in position and position["current_price"] <= position["stop_loss"]:
            return True, "Stop loss triggered"
        
        # 2. Check take profit
        if "take_profit" in position and position["current_price"] >= position["take_profit"]:
            return True, "Take profit triggered"
        
        # 3. Check max loss
        max_loss_pct = self.backtest_config.get("max_loss_percentage", 50)
        if position["unrealized_pnl_pct"] <= -max_loss_pct:
            return True, f"Max loss reached ({position['unrealized_pnl_pct']:.2f}%)"
        
        # 4. Check days to expiration
        current_date = datetime.now().date()
        exp_date = datetime.strptime(position["expiration"], '%Y-%m-%d').date()
        days_to_exp = (exp_date - current_date).days
        
        min_days = self.backtest_config.get("min_days_to_expiration", 2)
        if days_to_exp <= min_days:
            return True, f"Close to expiration ({days_to_exp} days left)"
        
        # 5. Check time-based exit (max hold time)
        if "entry_date" in position:
            entry_date = datetime.strptime(position["entry_date"], '%Y-%m-%d').date()
            days_held = (current_date - entry_date).days
            
            max_hold_days = self.backtest_config.get("max_hold_days", 15)
            if days_held >= max_hold_days:
                return True, f"Max hold time reached ({days_held} days)"
        
        # 6. Check theta decay
        if "current_greeks" in position and "theta" in position["current_greeks"]:
            theta = position["current_greeks"]["theta"]
            theta_threshold = self.backtest_config.get("theta_threshold", -0.03)
            
            if theta < theta_threshold:
                return True, f"Excessive theta decay ({theta:.4f})"
        
        return False, None
    
    def _exit_position(self, backtest_state, position_id, exit_reason):
        """
        Exit a position and record the trade.
        
        Args:
            backtest_state (dict): Current state of the backtest
            position_id (str): ID of the position to exit
            exit_reason (str): Reason for exiting
        """
        if position_id not in backtest_state["positions"]:
            return
        
        position = backtest_state["positions"][position_id]
        current_date = backtest_state["current_date"]
        
        # Set exit details
        position["exit_date"] = current_date.strftime('%Y-%m-%d')
        position["exit_reason"] = exit_reason
        
        # If exit price not already set (like in expiration), use current price
        if "exit_price" not in position:
            position["exit_price"] = position["current_price"]
        
        # Calculate P&L
        position["realized_pnl"] = (position["exit_price"] - position["entry_price"]) * position["quantity"] * 100
        position["realized_pnl_pct"] = (position["exit_price"] / position["entry_price"] - 1) * 100
        
        # Add to cash
        backtest_state["cash"] += position["exit_price"] * position["quantity"] * 100
        
        # Record trade
        backtest_state["closed_trades"].append(position)
        
        # Add to trade history
        trade_record = {
            "trade_id": position["position_id"],
            "symbol": position["symbol"],
            "option_symbol": position["option_symbol"],
            "option_type": position["option_type"],
            "strike": position["strike"],
            "entry_date": position["entry_date"],
            "exit_date": position["exit_date"],
            "entry_price": position["entry_price"],
            "exit_price": position["exit_price"],
            "quantity": position["quantity"],
            "pnl": position["realized_pnl"],
            "pnl_pct": position["realized_pnl_pct"],
            "exit_reason": exit_reason
        }
        
        backtest_state["trade_history"].append(trade_record)
        
        # Remove from active positions
        del backtest_state["positions"][position_id]
    
    def _find_new_entries(self, backtest_state, symbols):
        """
        Find new entry opportunities based on strategy.
        
        Args:
            backtest_state (dict): Current state of the backtest
            symbols (list): List of symbols to check
        """
        current_date = backtest_state["current_date"]
        current_data = backtest_state["current_data"]
        
        # Check available capital
        available_capital = backtest_state["cash"]
        min_capital_per_trade = self.backtest_config.get("min_capital_per_trade", 5000)
        
        if available_capital < min_capital_per_trade:
            return
        
        # Check max positions
        max_positions = self.backtest_config.get("max_positions", 5)
        if len(backtest_state["positions"]) >= max_positions:
            return
        
        # Check each symbol
        for symbol in symbols:
            # Skip if not in data
            if symbol not in current_data["underlying"] or symbol not in current_data["options"]:
                continue
            
            # Find opportunities
            opportunities = self._scan_for_opportunities(symbol, current_data)
            
            # Take top opportunities
            max_new_trades = max_positions - len(backtest_state["positions"])
            max_capital = min(available_capital, max_new_trades * min_capital_per_trade)
            
            for opportunity in opportunities:
                # Skip if would exceed max positions
                if len(backtest_state["positions"]) >= max_positions:
                    break
                
                # Skip if not enough capital
                position_size = opportunity["price"] * 100  # 100 shares per contract
                if position_size > max_capital:
                    continue
                
                # Enter the trade
                self._enter_position(backtest_state, opportunity)
                
                # Update available capital
                available_capital -= position_size
                max_capital -= position_size
                
                # Break if out of capital
                if max_capital < min_capital_per_trade:
                    break
    
    def _scan_for_opportunities(self, symbol, current_data):
        """
        Scan for trading opportunities based on strategy.
        
        Args:
            symbol (str): Symbol to scan
            current_data (dict): Current market data
            
        Returns:
            list: List of opportunity dictionaries
        """
        opportunities = []
        
        # Get underlying data
        underlying_data = current_data["underlying"][symbol]
        underlying_price = underlying_data["last"]
        
        # Get option chain
        option_chain = current_data["options"][symbol]
        
        # Use the option pricer to evaluate options
        for option in option_chain:
            # Extract option details
            option_type = option["option_type"]
            strike = option["strike"]
            expiration = option["expiration"]
            bid = option["bid"]
            ask = option["ask"]
            mid_price = (bid + ask) / 2
            
            # Skip options with very low liquidity
            if option.get("volume", 0) < 10 or option.get("open_interest", 0) < 100:
                continue
            
            # Calculate days to expiration
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            current_date = datetime.now()
            days_to_exp = (exp_date - current_date).days
            
            # Skip options expiring too soon or too far
            min_days = self.backtest_config.get("min_days_to_expiration", 7)
            max_days = self.backtest_config.get("max_days_to_expiration", 45)
            
            if days_to_exp < min_days or days_to_exp > max_days:
                continue
            
            # Get implied volatility
            if "greeks" in option and "mid_iv" in option["greeks"]:
                iv = option["greeks"]["mid_iv"]
            else:
                iv = 0.3  # Default if not available
            
            # Calculate theoretical price
            time_to_exp = days_to_exp / 365.0
            r = 0.02  # Risk-free rate
            
            option_enum = self.option_pricer.OptionType.CALL if option_type == "call" else self.option_pricer.OptionType.PUT
            model_price = self.option_pricer.black_scholes(underlying_price, strike, time_to_exp, r, iv, option_enum)
            
            # Check if option is undervalued
            classification, diff_percent = self.option_pricer.classify_option(mid_price, model_price)
            
            if classification == self.option_pricer.UNDERVALUED:
                # Calculate Greeks
                greeks = self.option_pricer.calculate_greeks(underlying_price, strike, time_to_exp, r, iv, option_enum)
                
                # Create opportunity
                opportunity = {
                    "symbol": symbol,
                    "option_symbol": option["symbol"],
                    "option_type": option_type,
                    "strike": strike,
                    "expiration": expiration,
                    "days_to_expiration": days_to_exp,
                    "underlying_price": underlying_price,
                    "bid": bid,
                    "ask": ask,
                    "price": mid_price,
                    "model_price": model_price,
                    "diff_percent": diff_percent,
                    "greeks": greeks
                }
                
                opportunities.append(opportunity)
        
        # Sort opportunities by undervaluation percentage
        opportunities.sort(key=lambda x: x["diff_percent"])
        
        return opportunities
    
    def _enter_position(self, backtest_state, opportunity):
        """
        Enter a new position.
        
        Args:
            backtest_state (dict): Current state of the backtest
            opportunity (dict): Trading opportunity
        """
        current_date = backtest_state["current_date"]
        
        # Calculate position size (1 contract by default)
        quantity = 1
        position_size = opportunity["price"] * quantity * 100  # 100 shares per contract
        
        # Calculate stop loss and take profit
        stop_loss_pct = self.backtest_config.get("stop_loss_percentage", 25)
        take_profit_pct = self.backtest_config.get("take_profit_percentage", 50)
        
        stop_loss = opportunity["price"] * (1 - stop_loss_pct / 100)
        take_profit = opportunity["price"] * (1 + take_profit_pct / 100)
        
        # Generate position ID
        position_id = f"{opportunity['symbol']}_{current_date.strftime('%Y%m%d')}_{len(backtest_state['positions'])}"
        
        # Create position
        position = {
            "position_id": position_id,
            "symbol": opportunity["symbol"],
            "option_symbol": opportunity["option_symbol"],
            "option_type": opportunity["option_type"],
            "strike": opportunity["strike"],
            "expiration": opportunity["expiration"],
            "days_to_expiration": opportunity["days_to_expiration"],
            "entry_date": current_date.strftime('%Y-%m-%d'),
            "entry_price": opportunity["price"],
            "quantity": quantity,
            "cost_basis": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "current_price": opportunity["price"],
            "current_underlying_price": opportunity["underlying_price"],
            "initial_greeks": opportunity["greeks"]
        }
        
        # Deduct from cash
        backtest_state["cash"] -= position_size
        
        # Add to positions
        backtest_state["positions"][position_id] = position
    
    def _record_daily_equity(self, backtest_state, date_str):
        """
        Record daily account equity.
        
        Args:
            backtest_state (dict): Current state of the backtest
            date_str (str): Current date string
        """
        # Calculate total equity
        cash = backtest_state["cash"]
        positions_value = 0
        
        for position in backtest_state["positions"].values():
            positions_value += position["current_price"] * position["quantity"] * 100
        
        total_equity = cash + positions_value
        
        # Record
        backtest_state["daily_equity"].append({
            "date": date_str,
            "cash": cash,
            "positions_value": positions_value,
            "total_equity": total_equity
        })
    
    def _calculate_performance_metrics(self, backtest_state):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_state (dict): Final state of the backtest
            
        Returns:
            dict: Performance metrics
        """
        self.logger.info("Calculating performance metrics")
        
        # Basic metrics
        initial_capital = backtest_state["initial_capital"]
        final_equity = backtest_state["daily_equity"][-1]["total_equity"] if backtest_state["daily_equity"] else initial_capital
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Convert daily equity to DataFrame for calculations
        equity_df = pd.DataFrame(backtest_state["daily_equity"])
        
        # Skip further calculations if no equity data
        if equity_df.empty:
            return {
                "success": True,
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return": total_return,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "trade_history": []
            }
        
        # Trade statistics
        total_trades = len(backtest_state["closed_trades"])
        winning_trades = len([t for t in backtest_state["closed_trades"] if t["realized_pnl"] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average profit and loss
        profits = [t["realized_pnl"] for t in backtest_state["closed_trades"] if t["realized_pnl"] > 0]
        losses = [t["realized_pnl"] for t in backtest_state["closed_trades"] if t["realized_pnl"] <= 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(profits)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_df['equity_peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] / equity_df['equity_peak'] - 1) * 100
        max_drawdown = abs(equity_df['drawdown'].min())
        
        # Daily returns
        equity_df['daily_return'] = equity_df['total_equity'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        avg_daily_return = equity_df['daily_return'].mean()
        std_daily_return = equity_df['daily_return'].std()
        sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Monthly returns
        if 'date' in equity_df.columns:
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df['month'] = equity_df['date'].dt.to_period('M')
            monthly_returns = equity_df.groupby('month')['total_equity'].last().pct_change()
            monthly_return_avg = monthly_returns.mean() * 100
            monthly_return_std = monthly_returns.std() * 100
        else:
            monthly_return_avg = 0
            monthly_return_std = 0
        
        # Results dictionary
        results = {
            "success": True,
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate * 100,  # Convert to percentage
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "monthly_return_avg": monthly_return_avg,
            "monthly_return_std": monthly_return_std,
            "trade_history": backtest_state["trade_history"],
            "equity_curve": [
                {"date": str(row["date"]), "equity": row["total_equity"]}
                for _, row in equity_df.iterrows()
            ]
        }
        
        return results
    
    def _save_backtest_results(self, results):
        """
        Save backtest results to file.
        
        Args:
            results (dict): Backtest results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"data/backtest_results/backtest_{timestamp}.json"
        
        try:
            # Create simplified results without full equity curve for storage
            storage_results = results.copy()
            
            # Simplify large data structures
            if "equity_curve" in storage_results:
                # Sample equity curve (every 5th point)
                storage_results["equity_curve"] = storage_results["equity_curve"][::5]
            
            if "trade_history" in storage_results and len(storage_results["trade_history"]) > 100:
                # Limit to 100 most recent trades
                storage_results["trade_history"] = storage_results["trade_history"][-100:]
            
            # Save to file
            with open(results_file, 'w') as f:
                json.dump(storage_results, f, indent=2)
            
            self.logger.info(f"Saved backtest results to {results_file}")
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    f"backtest_{timestamp}.json",
                    json.dumps(storage_results, indent=2),
                    folder="backtest_results",
                    mime_type="application/json"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
    
    def analyze_results(self, results):
        """
        Analyze and print backtest results.
        
        Args:
            results (dict): Backtest results
        """
        if not results.get("success", False):
            self.logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
            return
        
        # Print summary
        print("\n========== BACKTEST RESULTS ==========")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print("\n---------- Trade Statistics ----------")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.2f}%)")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Profit Factor: {results['profit_factor']:.4f}")
        print(f"Avg Profit: ${results['avg_profit']:.2f}")
        print(f"Avg Loss: ${results['avg_loss']:.2f}")
        print("======================================")
        
        # Plotting
        try:
            self._plot_equity_curve(results)
            self._plot_monthly_returns(results)
            self._plot_drawdowns(results)
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
    
    def _plot_equity_curve(self, results):
        """
        Plot equity curve.
        
        Args:
            results (dict): Backtest results
        """
        if "equity_curve" not in results or not results["equity_curve"]:
            return
        
        # Convert to DataFrame
        equity_data = pd.DataFrame(results["equity_curve"])
        equity_data["date"] = pd.to_datetime(equity_data["date"])
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_data["date"], equity_data["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig("data/backtest_results/equity_curve.png")
        plt.close()
    
    def _plot_monthly_returns(self, results):
        """
        Plot monthly returns.
        
        Args:
            results (dict): Backtest results
        """
        if "equity_curve" not in results or not results["equity_curve"]:
            return
        
        # Convert to DataFrame
        equity_data = pd.DataFrame(results["equity_curve"])
        equity_data["date"] = pd.to_datetime(equity_data["date"])
        equity_data["month"] = equity_data["date"].dt.to_period("M")
        
        # Calculate monthly returns
        monthly_equity = equity_data.groupby("month")["equity"].last().reset_index()
        monthly_equity["return"] = monthly_equity["equity"].pct_change() * 100
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_equity["month"].astype(str), monthly_equity["return"])
        plt.title("Monthly Returns")
        plt.xlabel("Month")
        plt.ylabel("Return (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("data/backtest_results/monthly_returns.png")
        plt.close()
    
    def _plot_drawdowns(self, results):
        """
        Plot drawdowns.
        
        Args:
            results (dict): Backtest results
        """
        if "equity_curve" not in results or not results["equity_curve"]:
            return
        
        # Convert to DataFrame
        equity_data = pd.DataFrame(results["equity_curve"])
        equity_data["date"] = pd.to_datetime(equity_data["date"])
        
        # Calculate drawdowns
        equity_data["peak"] = equity_data["equity"].cummax()
        equity_data["drawdown"] = (equity_data["equity"] / equity_data["peak"] - 1) * 100
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_data["date"], equity_data["drawdown"], 0, color='red', alpha=0.3)
        plt.plot(equity_data["date"], equity_data["drawdown"], color='red')
        plt.title("Portfolio Drawdowns")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("data/backtest_results/drawdowns.png")
        plt.close()
    
    def generate_report(self, results):
        """
        Generate a comprehensive backtest report.
        
        Args:
            results (dict): Backtest results
            
        Returns:
            str: Path to the generated report
        """
        try:
            report_file = f"data/backtest_results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # HTML report template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Option Hunter Backtest Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .metric {{ font-weight: bold; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .section {{ margin-bottom: 30px; }}
                    .chart-container {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>Option Hunter Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <table>
                        <tr>
                            <td class="metric">Initial Capital</td>
                            <td>${results['initial_capital']:.2f}</td>
                        </tr>
                        <tr>
                            <td class="metric">Final Equity</td>
                            <td>${results['final_equity']:.2f}</td>
                        </tr>
                        <tr>
                            <td class="metric">Total Return</td>
                            <td class="{('positive' if results['total_return'] >= 0 else 'negative')}">{results['total_return']:.2f}%</td>
                        </tr>
                        <tr>
                            <td class="metric">Max Drawdown</td>
                            <td class="negative">{results['max_drawdown']:.2f}%</td>
                        </tr>
                        <tr>
                            <td class="metric">Sharpe Ratio</td>
                            <td>{results['sharpe_ratio']:.4f}</td>
                        </tr>
                        <tr>
                            <td class="metric">Monthly Return (Avg)</td>
                            <td>{results.get('monthly_return_avg', 0):.2f}%</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Trade Statistics</h2>
                    <table>
                        <tr>
                            <td class="metric">Total Trades</td>
                            <td>{results['total_trades']}</td>
                        </tr>
                        <tr>
                            <td class="metric">Winning Trades</td>
                            <td>{results['winning_trades']} ({results['win_rate']:.2f}%)</td>
                        </tr>
                        <tr>
                            <td class="metric">Losing Trades</td>
                            <td>{results['losing_trades']}</td>
                        </tr>
                        <tr>
                            <td class="metric">Profit Factor</td>
                            <td>{results['profit_factor']:.4f}</td>
                        </tr>
                        <tr>
                            <td class="metric">Average Profit</td>
                            <td class="positive">${results['avg_profit']:.2f}</td>
                        </tr>
                        <tr>
                            <td class="metric">Average Loss</td>
                            <td class="negative">${abs(results['avg_loss']):.2f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Charts</h2>
                    <div class="chart-container">
                        <h3>Equity Curve</h3>
                        <img src="equity_curve.png" alt="Equity Curve" style="max-width: 100%;" />
                    </div>
                    <div class="chart-container">
                        <h3>Monthly Returns</h3>
                        <img src="monthly_returns.png" alt="Monthly Returns" style="max-width: 100%;" />
                    </div>
                    <div class="chart-container">
                        <h3>Drawdowns</h3>
                        <img src="drawdowns.png" alt="Drawdowns" style="max-width: 100%;" />
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recent Trades</h2>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Type</th>
                            <th>Entry Date</th>
                            <th>Exit Date</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Exit Reason</th>
                        </tr>
            """
            
            # Add recent trades (up to 20)
            recent_trades = results.get("trade_history", [])[-20:]
            for trade in reversed(recent_trades):
                pnl_class = "positive" if trade["pnl"] >= 0 else "negative"
                html_content += f"""
                        <tr>
                            <td>{trade["symbol"]}</td>
                            <td>{trade["option_type"]} {trade["strike"]}</td>
                            <td>{trade["entry_date"]}</td>
                            <td>{trade["exit_date"]}</td>
                            <td>${trade["entry_price"]:.4f}</td>
                            <td>${trade["exit_price"]:.4f}</td>
                            <td class="{pnl_class}">${trade["pnl"]:.2f}</td>
                            <td class="{pnl_class}">{trade["pnl_pct"]:.2f}%</td>
                            <td>{trade["exit_reason"]}</td>
                        </tr>
                """
            
            # Close the HTML
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Backtest Parameters</h2>
                    <p>This report contains detailed metrics and visualizations of the backtest results.</p>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(report_file, "w") as f:
                f.write(html_content)
            
            self.logger.info(f"Generated backtest report: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None
    
    def run_scenario_analysis(self, base_params, param_variations):
        """
        Run multiple backtest scenarios with parameter variations.
        
        Args:
            base_params (dict): Base parameters for backtest
            param_variations (dict): Parameter variations to test
            
        Returns:
            dict: Scenario analysis results
        """
        self.logger.info("Running scenario analysis")
        
        scenario_results = []
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_variations.keys())
        param_values = list(param_variations.values())
        
        # Run each scenario
        for params in itertools.product(*param_values):
            # Create scenario parameters
            scenario_params = base_params.copy()
            for i, name in enumerate(param_names):
                scenario_params[name] = params[i]
            
            # Run backtest with these parameters
            
            # TODO: Implement scenario-specific backtest
            
            # For now, just append placeholder results
            scenario_results.append({
                "params": {name: params[i] for i, name in enumerate(param_names)},
                "results": {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0
                }
            })
        
        # Analyze scenario results
        optimal_scenario = max(scenario_results, key=lambda x: x["results"]["sharpe_ratio"])
        
        return {
            "scenarios": scenario_results,
            "optimal_scenario": optimal_scenario
        }"""
Backtest Engine Module

This module implements a backtesting system for the Option Hunter strategy.
It allows testing of strategies against historical data to evaluate performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BacktestEngine:
    """
    Engine for backtesting options trading strategies using historical data.
    """
    
    def __init__(self, config, parameter_hub, option_pricer, drive_manager=None):
        """
        Initialize the BacktestEngine.
        
        Args:
            config (dict): Configuration settings
            parameter_hub: Instance of MasterParameterHub for parameters
            option_pricer: Instance of OptionPricer for option pricing
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.parameter_hub = parameter_hub
        self.option_pricer = option_pricer
        self.drive_manager = drive_manager
        
        # Extract backtest configuration
        self.backtest_config = config.get("backtesting", {})
        
        # Create directories
        os.makedirs("data/backtest_results", exist_ok=True)
        os.makedirs("data/collected_options_data", exist_ok=True)
        
        self.logger.info("BacktestEngine initialized")
    
    def run_backtest(self, start_date=None, end_date=None, symbols=None):
        """
        Run a backtest over the specified period and symbols.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            symbols (list, optional): List of symbols to backtest
            
        Returns:
            dict: Backtest results
        """
        # Use configuration defaults if not specified
        if start_date is None:
            start_date = self.backtest_config.get(
                "default_start_date", 
                (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            )
        
        if end_date is None:
            end_date = self.backtest_config.get(
                "default_end_date",
                datetime.now().strftime('%Y-%m-%d')
            )
        
        if symbols is None:
            symbols = self.backtest_config.get(
                "default_symbols",
                self.config.get("watchlist", {}).get("high_priority", ["SPY", "QQQ", "AAPL", "MSFT"])
            )
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date} with {len(symbols)} symbols")
        
        # Load historical options data
        options_data = self._load_historical_options_data(symbols, start_date, end_date)
        
        if not options_data:
            self.logger.error("No historical options data available for backtest")
            return {
                "success": False,
                "error": "No historical options data available"
            }
        
        # Initialize backtest state
        initial_capital = self.backtest_config.get("initial_capital", 100000)
        
        backtest_state = {
            "current_date": datetime.strptime(start_date, '%Y-%m-%d'),
            "end_date": datetime.strptime(end_date, '%Y-%m-%d'),
            "cash": initial_capital,
            "initial_capital": initial_capital,
            "positions": {},
            "closed_trades": [],
            "daily_equity": [],
            "trade_history": [],
            "current_data": None
        }
        
        # Run day-by-day simulation
        try:
            self._run_backtest_simulation(backtest_state, options_data, symbols)
        except Exception as e:
            self.logger.error(f"Error during backtest simulation: {str(e)}")
            return {
                "success": False,
                "error": f"Simulation error: {str(e)}"
            }
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(backtest_state)
        
        # Save results
        self._save_backtest_results(results)
        
        return results
    
    def _load_historical_options_data(self, symbols, start_date, end_date):
        """
        Load historical options data for backtest.
        
        Args:
            symbols (list): List of symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Dictionary mapping dates to available options data
        """
        self.logger.info("Loading historical options data")
        
        # This would typically load from a database or files
        # For now, we'll just create sample data for demonstration
        
        # In a real implementation, this would load actual historical options data
        # from a data provider or local storage
        
        # Check if we have real historical data
        data_dir = "data/collected_options_data"
        has_real_data = False
        
        for symbol in symbols:
            symbol_dir = f"{data_dir}/{symbol}"
            if os.path.exists(symbol_dir) and os.listdir(symbol_dir):
                has_real_data = True
                break
        
        if has_real_data:
            return self._load_real_historical_data(symbols, start_date, end_date)
        else:
            self.logger.warning("No real historical data found, using simulated data")
            return self._generate_simulated_data(symbols, start_date, end_date)
    
    def _load_real_historical_data(self, symbols, start_date, end_date):
        """
        Load real historical options data from files.
        
        Args:
            symbols (list): List of symbols
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Dictionary mapping dates to available options data
        """
        data_dir = "data/collected_options_data"
        data_by_date = {}
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Iterate through dates
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            data_by_date[date_str] = {"options": {}, "underlying": {}}
            
            # Load data for each symbol
            for symbol in symbols:
                symbol_dir = f"{data_dir}/{symbol}"
                date_file = f"{symbol_dir}/{date_str}.json"
                
                if os.path.exists(date_file):
                    try:
                        with open(date_file, 'r') as f:
                            symbol_data = json.load(f)
                        
                        # Add to collected data
                        if "options" in symbol_data:
                            data_by_date[date_str]["options"][symbol] = symbol_data["options"]
                        
                        if "underlying" in symbol_data:
                            data_by_date[date_str]["underlying"][symbol] = symbol_data["underlying"]
                            
                    except Exception as e:
                        self.logger.error(f"Error loading data file {date_file}: {str(e)}")
            
            current_date += timedelta(days=1)
        
        return data_by_date
    
    def _generate_simulated_data(self, symbols, start_date, end_date):
        """
        Generate simulated options data for testing.
        
        Args:
            symbols (list): List of symbols
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            dict: Dictionary mapping dates to simulated options data
        """
        self.logger.info("Generating simulated options data for backtest")
        
        data_by_date = {}
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Simulate price movements for each symbol
        symbol_prices = {}
        
        for symbol in symbols:
            # Start with a random price between $50 and $500
            base_price = np.random.uniform(50, 500)
            
            # Simulate a price series with random walk
            daily_returns = np.random.normal(0.0005, 0.015, (end - start).days + 1)
            prices = [base_price]
            
            for ret in daily_returns:
                next_price = prices[-1] * (1 + ret)
                prices.append(next_price)
            
            symbol_prices[symbol] = prices
        
        # Generate options data for each date
        current_date = start
        day_index = 0
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            data_by_date[date_str] = {"options": {}, "underlying": {}}
            
            for symbol in symbols:
                # Get current price
                current_price = symbol_prices[symbol][day_index]
                
                # Create underlying data
                data_by_date[date_str]["underlying"][symbol] = {
                    "symbol": symbol,
                    "last": current_price,
                    "change": symbol_prices[symbol][day_index] - symbol_prices[symbol][max(0, day_index-1)],
                    "volume": np.random.randint(100000, 10000000)
                }
                
                # Generate options for this symbol
                options = self._generate_simulated_options(symbol, current_price, current_date)
                data_by_date[date_str]["options"][symbol] = options
            
            current_date += timedelta(days=1)
            day_index += 1
        
        return data_by_date
    
    def _generate_simulated_options(self, symbol, underlying_price, current_date):
        """
        Generate simulated options for a symbol.
        
        Args:
            symbol (str): Symbol
            underlying_price (float): Current price of underlying
            current_date (datetime): Current date
            
        Returns:
            list: List of simulated option contracts
        """
        options = []
        
        # Generate expiration dates (1, 2, 3 months out)
        expirations = [
            (current_date + timedelta(days=30)).strftime('%Y-%m-%d'),
            (current_date + timedelta(days=60)).strftime('%Y-%m-%d'),
            (current_date + timedelta(days=90)).strftime('%Y-%m-%d')
        ]
        
        # Generate strikes around current price
        strikes = [
            round(underlying_price * 0.8),
            round(underlying_price * 0.9),
            round(underlying_price),
            round(underlying_price * 1.1),
            round(underlying_price * 1.2)
        ]
        
        # Generate implied volatility values
        base_iv = np.random.uniform(0.2, 0.4)  # Base IV between 20% and 40%
        
        # Create options for each expiration and strike
        for expiration in expirations:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            days_to_exp = (exp_date - current_date).days
            
            # Time to expiration in years
            T = days_to_exp / 365.0
            
            # Risk-free rate
            r = 0.02
            
            for strike in strikes:
                # Adjust IV based on strike (volatility smile)
                moneyness = strike / underlying_price
                iv_adjustment = 0.05 * abs(moneyness - 1) * 2  # Volatility smile
                
                # Higher IV for longer-dated options
                iv_exp_adjustment = 0.02 * (days_to_exp / 30)
                
                call_iv = base_iv + iv_adjustment + iv_exp_adjustment
                put_iv = base_iv + iv_adjustment + iv_exp_adjustment
                
                # Calculate option prices using Black-Scholes
                call_price = self.option_pricer.black_scholes(
                    underlying_price, strike, T, r, call_iv, self.option_pricer.OptionType.CALL
                )
                
                put_price = self.option_pricer.black_scholes(
                    underlying_price, strike, T, r, put_iv, self.option_pricer.OptionType.PUT
                )
                
                # Add bid-ask spread
                call_bid = call_price * 0.95
                call_ask = call_price * 1.05
                
                put_bid = put_price * 0.95
                put_ask = put_price * 1.05
                
                # Generate Greeks
                call_greeks = self.option_pricer.calculate_greeks(
                    underlying_price, strike, T, r, call_iv, self.option_pricer.OptionType.CALL
                )
                
                put_greeks = self.option_pricer.calculate_greeks(
                    underlying_price, strike, T, r, put_iv, self.option_pricer.OptionType.PUT
                )
                
                # Generate option symbols
                call_symbol = f"{symbol}{exp_date.strftime('%y%m%d')}C{str(int(strike*1000)).zfill(8)}"
                put_symbol = f"{symbol}{exp_date.strftime('%y%m%d')}P{str(int(strike*1000)).zfill(8)}"
                
                # Add call option
                call_option = {
                    "symbol": call_symbol,
                    "option_type": "call",
                    "strike": strike,
                    "expiration": expiration,
                    "bid": call_bid,
                    "ask": call_ask,
                    "last": (call_bid + call_ask) / 2,
                    "volume": np.random.randint(10, 1000),
                    "open_interest": np.random.randint(100, 5000),
                    "underlying_price": underlying_price,
                    "greeks": {
                        "delta": call_greeks["delta"],
                        "gamma": call_greeks["gamma"],
                        "theta": call_greeks["theta"],
                        "vega": call_greeks["vega"],
                        "mid_iv": call_iv
                    }
                }
                
                # Add put option
                put_option = {
                    "symbol": put_symbol,
                    "option_type": "put",
                    "strike": strike,
                    "expiration": expiration,
                    "bid": put_bid,
                    "ask": put_ask,
                    "last": (put_bid + put_ask) / 2,
                    "volume": np.random.randint(10, 1000),
                    "open_interest": np.random.randint(100, 5000),
                    "underlying_price": underlying_price,
                    "greeks": {
                        "delta": put_greeks["delta"],
                        "gamma": put_greeks["gamma"],
                        "theta": put_greeks["theta"],
                        "vega": put_greeks["vega"],
                        "mid_iv": put_iv
                    }
                }
                
                options.append(call_option)
                options.append(put_option)
        
        return options
