import argparse
import json
import time
from datetime import datetime
from config import Config
from data_fetcher import DataFetcher
from risk_manager import RiskManager
from execution import MockExecutor, AlpacaExecutor, CCXTExecutor
from strategy import RuleBasedStrategy, DQNStrategy, LLMStrategy, SelfLearningEngine


class TradingBotOrchestrator:
    """
    Central Orchestrator.
    Ties together Data Fetching, AI/Strategy Decisions, Risk Checking,
    Order Execution, Ledger Recording, and the Self-Learning Optimization Loop.
    """

    def __init__(self, mode="paper", strategy_name="technical"):
        self.mode = mode  # "backtest", "paper", or "live"
        self.strategy_name = strategy_name
        
        # 1. Initialize core system utilities
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager()
        self.learning_engine = SelfLearningEngine()
        
        # 2. Select strategy backend
        self._init_strategy()
        
        # 3. Select execution engine
        self._init_executor()
        
        # 4. Open trade memory buffer (logs open entry prices and sizing)
        self.active_trade_log = None
        self.load_active_trade()

    def _init_strategy(self):
        """Instantiates the chosen decision engine strategy."""
        if self.strategy_name == "rl":
            self.strategy = DQNStrategy()
            print("[Orchestrator] Neural Network Reinforcement Learning (DQN) Strategy active.")
        elif self.strategy_name == "llm":
            self.strategy = LLMStrategy()
            print("[Orchestrator] AI LLM Strategy (Google Gemini) active.")
        else:
            self.strategy = RuleBasedStrategy()
            print("[Orchestrator] Standard Quantitative Rule-Based Strategy active.")

    def _init_executor(self):
        """Instantiates the broker order submission framework."""
        # Check defaults or explicit sandbox configs
        if Config.EXECUTION_ENGINE == "alpaca" and Config.ALPACA_API_KEY:
            self.executor = AlpacaExecutor()
            print(f"[Orchestrator] Connected to Alpaca Trading API (Paper Mode: {Config.ALPACA_IS_PAPER})")
        elif Config.EXECUTION_ENGINE == "ccxt" and Config.CCXT_API_KEY:
            self.executor = CCXTExecutor(self.data_fetcher)
            print(f"[Orchestrator] Connected to CCXT Exchange: {Config.CCXT_EXCHANGE_ID.upper()} "
                  f"(Paper/Sandbox: {Config.PAPER_TRADING})")
        else:
            self.executor = MockExecutor()
            print("[Orchestrator] Configured with Local Mock Broker (No API keys required, balance persisted).")

    def load_active_trade(self):
        """Loads incomplete transaction states to track entry prices across bot reboots."""
        active_trade_file = Config.DATABASE_DIR / "active_trade.json"
        if active_trade_file.exists():
            try:
                with open(active_trade_file, "r") as f:
                    self.active_trade_log = json.load(f)
            except Exception as e:
                print(f"[Orchestrator] Error loading active trade log: {e}")

    def save_active_trade(self):
        """Saves active transaction configurations."""
        active_trade_file = Config.DATABASE_DIR / "active_trade.json"
        try:
            if self.active_trade_log:
                with open(active_trade_file, "w") as f:
                    json.dump(self.active_trade_log, f, indent=4)
            elif active_trade_file.exists():
                active_trade_file.unlink()
        except Exception as e:
            print(f"[Orchestrator] Error saving active trade state: {e}")

    def log_to_ledger(self, completed_trade: dict):
        """Appends a completed transaction to the historical learning ledger."""
        ledger = []
        if Config.TRADE_LEDGER_PATH.exists():
            try:
                with open(Config.TRADE_LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
            except Exception:
                pass
                
        ledger.append(completed_trade)
        
        try:
            with open(Config.TRADE_LEDGER_PATH, "w") as f:
                json.dump(ledger, f, indent=4)
            print(f"[Orchestrator] Trade appended to structured ledger: {completed_trade.get('action')} "
                  f"PnL: ${completed_trade.get('pnl', 0.0):.2f}")
        except Exception as e:
            print(f"[Orchestrator] Failed writing trade log to ledger: {e}")

    # =====================================================================
    # Live & Paper Trading Execution Cycle
    # =====================================================================
    def run_live_cycle(self):
        """
        Runs a live tick evaluation check.
        Multi-Coin Portfolio Autonomous Scanner:
        - Scans ALL market coins on every tick.
        - Evaluates BUY opportunities across all coins concurrently and holds multiple assets.
        - Monitors all active open positions for SELL signals, Stop-Loss, or Take-Profit thresholds.
        """
        print(f"\n--- Multi-Asset Market Scanner Tick Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # 1. Fetch current portfolio and balance state
        portfolio = self.executor.get_portfolio_state()
        equity = portfolio["equity"]
        cash = portfolio["cash"]
        
        # Check Daily Drawdown Circuit Breakers
        if not self.risk_manager.check_circuit_breakers(equity):
            if portfolio.get("positions_count", 0) > 0 or portfolio.get("has_position", False):
                print("[Orchestrator] Emergency Close triggered by Risk Manager drawdown limits!")
                closures = self.executor.close_all_positions()
                for receipt in closures:
                    self.process_trade_closure(receipt, {"close": receipt.get("execution_price", 0.0)})
            return

        watchlist = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "AVAX/USD", "LINK/USD"]
        original_symbol = self.data_fetcher.symbol

        # 2. Iterate through all market coins
        for sym in watchlist:
            self.data_fetcher.symbol = sym
            try:
                market_state = self.data_fetcher.get_latest_market_state()
                current_price = market_state["close"]
                atr = market_state.get("atr", 0.0)

                # Check if we currently hold a position in this coin
                pos_info = {}
                if hasattr(self.executor, "positions"):
                    pos_info = self.executor.positions.get(sym, {})
                elif Config.TRADING_SYMBOL == sym and portfolio.get("has_position"):
                    pos_info = {"size": portfolio["position_size"], "entry_price": portfolio["entry_price"]}

                has_coin_position = pos_info.get("size", 0.0) > 0.0

                # 2.1 Update Stop-Loss / Take-Profit for held position in Mock Executor
                if has_coin_position and hasattr(self.executor, "update_price_action"):
                    exit_receipts = self.executor.update_price_action(current_price, symbol=sym)
                    if exit_receipts:
                        for rec in exit_receipts:
                            self.process_trade_closure(rec, market_state)
                        continue

                # 2.2 Strategy Signal Evaluation
                strategy_portfolio = {
                    "equity": equity,
                    "has_position": has_coin_position,
                    "position_size": pos_info.get("size", 0.0),
                    "entry_price": pos_info.get("entry_price", 0.0)
                }

                signal = self.strategy.generate_signal(market_state, strategy_portfolio)

                # 2.3 Risk Manager Validation
                validated_order = self.risk_manager.validate_risk(signal, current_price, atr, equity)
                action = validated_order["action"]

                if action == "BUY" and not has_coin_position:
                    size = validated_order["size"]
                    cost = size * current_price
                    if cost <= cash:
                        print(f"[Opportunity Match!] Buying {sym} at ${current_price:,.2f} | Confidence: {signal['confidence']:.2f} | Reason: {signal['reason']}")
                        
                        if hasattr(self.executor, "execute_order"):
                            try:
                                order_receipt = self.executor.execute_order(
                                    action="BUY",
                                    size=size,
                                    stop_loss=validated_order["stop_loss"],
                                    take_profit=validated_order["take_profit"],
                                    current_price=current_price,
                                    symbol=sym
                                )
                            except TypeError:
                                order_receipt = self.executor.execute_order(
                                    action="BUY",
                                    size=size,
                                    stop_loss=validated_order["stop_loss"],
                                    take_profit=validated_order["take_profit"],
                                    current_price=current_price
                                )

                            if order_receipt.get("status") == "SUCCESS":
                                order_receipt["symbol"] = sym
                                order_receipt["strategy_reason"] = signal["reason"]
                                order_receipt["strategy_reflection"] = signal["reflection"]
                                self.active_trade_log = order_receipt
                                self.save_active_trade()
                                cash -= cost
                    else:
                        print(f"[Risk Manager Suppressed {sym}] Insufficient cash for multi-position allocation.")

                elif action == "SELL" and has_coin_position:
                    print(f"[Exit Trigger!] Selling {sym} position at ${current_price:,.2f} | Reason: {signal['reason']}")
                    if hasattr(self.executor, "execute_order"):
                        try:
                            order_receipt = self.executor.execute_order(
                                action="SELL",
                                size=pos_info["size"],
                                stop_loss=0.0,
                                take_profit=0.0,
                                current_price=current_price,
                                symbol=sym
                            )
                        except TypeError:
                            order_receipt = self.executor.execute_order(
                                action="SELL",
                                size=pos_info["size"],
                                stop_loss=0.0,
                                take_profit=0.0,
                                current_price=current_price
                            )

                        if order_receipt.get("status") == "SUCCESS":
                            order_receipt["symbol"] = sym
                            order_receipt["strategy_reason"] = signal["reason"]
                            order_receipt["strategy_reflection"] = signal["reflection"]
                            self.process_trade_closure(order_receipt, market_state)

            except Exception as e:
                print(f"[Scanner] Error processing {sym}: {e}")
            finally:
                self.data_fetcher.symbol = original_symbol



    def process_trade_closure(self, sell_receipt: dict, latest_market: dict):
        """Merges Buy entry logs and Sell exit logs to calculate returns and trigger optimization audits."""
        if not self.active_trade_log:
            # Reconstruct dummy buy log if not found in active memories
            buy_price = latest_market["close"]
            qty = sell_receipt.get("size", 1.0)
            pnl = sell_receipt.get("pnl", 0.0)
        else:
            buy_price = self.active_trade_log.get("execution_price", latest_market["close"])
            qty = sell_receipt.get("size", self.active_trade_log.get("size", 1.0))
            pnl = (sell_receipt.get("execution_price") - buy_price) * qty

        completed_trade = {
            "timestamp": sell_receipt.get("timestamp"),
            "symbol": Config.TRADING_SYMBOL,
            "action": "TRADE_COMPLETED",
            "buy_price": buy_price,
            "sell_price": sell_receipt.get("execution_price"),
            "size": qty,
            "pnl": pnl,
            "exit_trigger": sell_receipt.get("exit_trigger", "Strategy Signal"),
            "strategy_reason": sell_receipt.get("strategy_reason", self.active_trade_log.get("strategy_reason") if self.active_trade_log else "N/A"),
            "strategy_reflection": sell_receipt.get("strategy_reflection", self.active_trade_log.get("strategy_reflection") if self.active_trade_log else "N/A")
        }
        
        self.log_to_ledger(completed_trade)
        
        # Reset memory state
        self.active_trade_log = None
        self.save_active_trade()

        # Run Self-Learning Optimization Loop
        print("[Orchestrator] Triggering Self-Learning parameter audit...")
        self.learning_engine.audit_and_optimize()
        
        # Reload rule thresholds in Strategy if configured
        if isinstance(self.strategy, RuleBasedStrategy):
            self.strategy.load_learned_parameters()

    # =====================================================================
    # Historical Backtest Loop & Reinforcement Learning Training
    # =====================================================================
    def run_backtest(self):
        """Simulates historical trade execution ticks and trains RL algorithms."""
        print(f"\n[Backtest] Loading historical data for {Config.TRADING_SYMBOL}...")
        df_raw = self.data_fetcher.fetch_historical_data(limit=1000)
        df_indicators = self.data_fetcher.calculate_indicators(df_raw)
        
        print(f"[Backtest] Simulating over {len(df_indicators)} ticks...")
        
        # Setup local mock broker state
        backtest_broker = MockExecutor()
        backtest_broker.cash = 10000.0
        backtest_broker.position_size = 0.0
        backtest_broker.entry_price = 0.0
        
        # Track stats
        trade_count = 0
        winning_trades = 0
        total_pnl = 0.0
        
        # Cache for RL state transition mapping
        last_state_vec = None
        last_action_idx = 0
        
        for idx in range(30, len(df_indicators)):
            # Slice state historical profile up to index (avoiding future leak)
            current_slice = df_indicators.iloc[:idx+1]
            market_state = current_slice.iloc[-1].to_dict()
            current_price = market_state["close"]
            atr = market_state.get("atr", 0.0)
            
            # Check stops
            exit_receipt = backtest_broker.update_price_action(current_price)
            if exit_receipt:
                pnl = exit_receipt["pnl"]
                total_pnl += pnl
                trade_count += 1
                if pnl > 0:
                    winning_trades += 1
                
                # DQN Training step reward reinforcement
                if self.strategy_name == "rl" and last_state_vec is not None:
                    reward = pnl
                    # Get next state
                    next_state_vec = self.strategy._get_state_vector(market_state, {"has_position": False})
                    # Add to replay buffer
                    self.strategy.memory.append((last_state_vec, last_action_idx, reward, next_state_vec, True))
                    self.strategy.train_step()
                    
                last_state_vec = None
                continue

            portfolio_state = backtest_broker.get_portfolio_state()
            equity = backtest_broker.get_equity(current_price)
            
            # Strategy input portfolio format
            strategy_portfolio = {
                "equity": equity,
                "has_position": portfolio_state["has_position"],
                "position_size": portfolio_state["position_size"],
                "entry_price": portfolio_state["entry_price"]
            }
            
            # Signal calculation
            signal = self.strategy.generate_signal(market_state, strategy_portfolio)
            validated = self.risk_manager.validate_risk(signal, current_price, atr, equity)
            action = validated["action"]
            
            if action in ("BUY", "SELL"):
                # DQN transition mapping
                if self.strategy_name == "rl":
                    action_idx = 1 if action == "BUY" else 2
                    last_state_vec = self.strategy._get_state_vector(market_state, strategy_portfolio)
                    last_action_idx = action_idx
                    
                receipt = backtest_broker.execute_order(
                    action=action,
                    size=validated["size"],
                    stop_loss=validated["stop_loss"],
                    take_profit=validated["take_profit"],
                    current_price=current_price
                )
                
                if action == "SELL" and receipt.get("status") == "SUCCESS":
                    pnl = receipt["pnl"]
                    total_pnl += pnl
                    trade_count += 1
                    if pnl > 0:
                        winning_trades += 1
                        
                    if self.strategy_name == "rl" and last_state_vec is not None:
                        reward = pnl
                        next_state_vec = self.strategy._get_state_vector(market_state, {"has_position": False})
                        self.strategy.memory.append((last_state_vec, last_action_idx, reward, next_state_vec, True))
                        self.strategy.train_step()
            else:
                # Train RL agent to receive small holding penalty or reward
                if self.strategy_name == "rl" and portfolio_state["has_position"]:
                    # Small step reward/penalty for holding profitable/unprofitable position
                    pnl_unrealized = (current_price - portfolio_state["entry_price"]) * portfolio_state["position_size"]
                    state_vec = self.strategy._get_state_vector(market_state, strategy_portfolio)
                    next_state_vec = self.strategy._get_state_vector(market_state, strategy_portfolio)
                    self.strategy.memory.append((state_vec, 0, pnl_unrealized * 0.01, next_state_vec, False))
                    self.strategy.train_step()

        print("\n=== BACKTEST RESULTS SUMMARY ===")
        print(f"Strategy:       {self.strategy_name.upper()}")
        print(f"Total Trades:   {trade_count}")
        print(f"Win Rate:       {(winning_trades/trade_count)*100:.2f}%" if trade_count > 0 else "Win Rate: 0%")
        print(f"Final Net PnL:  ${total_pnl:.2f}")
        print(f"Ending Balance: ${backtest_broker.cash:.2f}")
        print("================================")
        
        # Save trained neural network weights
        if self.strategy_name == "rl":
            self.strategy.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Self-Learning AI Trading Bot Orchestrator")
    parser.add_argument("--mode", type=str, default="paper", choices=["backtest", "paper", "live"],
                        help="Execution mode (backtest, paper-trading simulator, live exchange execution)")
    parser.add_argument("--strategy", type=str, default="technical", choices=["technical", "rl", "llm"],
                        help="Strategy decision engine (technical, rl, llm)")
    
    args = parser.parse_args()
    
    # Check configurations overrides
    if args.strategy:
        Config.STRATEGY_TYPE = args.strategy
        
    print(Config.get_summary())
    
    bot = TradingBotOrchestrator(mode=args.mode, strategy_name=args.strategy)
    
    if args.mode == "backtest":
        bot.run_backtest()
    else:
        # Run paper or live trading loop
        print(f"[Loop] Initiating active trading loop in {args.mode.upper()} mode...")
        # Resolve tick sleep timer based on Config timeframe
        timeframe_secs = 60
        if Config.TIMEFRAME == "5m":
            timeframe_secs = 300
        elif Config.TIMEFRAME == "15m":
            timeframe_secs = 900
        elif Config.TIMEFRAME == "1h":
            timeframe_secs = 3600
            
        # For simulation safety and responsiveness, loop faster in paper mode
        sleep_timer = 10 if args.mode == "paper" else timeframe_secs
        
        try:
            while True:
                bot.run_live_cycle()
                time.sleep(sleep_timer)
        except KeyboardInterrupt:
            print("\n[Halt] Shutting down trading orchestrator loop. Closing files...")
