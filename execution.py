import json
from abc import ABC, abstractmethod
from datetime import datetime
from config import Config

# Dynamic import helpers for CCXT and Alpaca
try:
    import ccxt
except ImportError:
    ccxt = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    HAS_ALPACA_TRADING = True
except ImportError:
    HAS_ALPACA_TRADING = False


# =====================================================================
# Base Execution Engine Interface
# =====================================================================
class BaseExecutionEngine(ABC):
    @abstractmethod
    def get_portfolio_state(self) -> dict:
        """
        Queries broker for current account metrics and positions.
        Returns:
            dict: {
                "equity": float (total portfolio value),
                "cash": float (available cash),
                "position_size": float (amount of asset held),
                "entry_price": float (average purchase price),
                "has_position": bool
            }
        """
        pass

    @abstractmethod
    def execute_order(self, action: str, size: float, stop_loss: float, take_profit: float, current_price: float) -> dict:
        """
        Submits a trade order.
        Returns:
            dict: {
                "status": "SUCCESS" | "FAILED",
                "order_id": str,
                "execution_price": float,
                "size": float,
                "pnl": float (if closing a position, else 0.0),
                "timestamp": str
            }
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> list:
        """
        Force-closes all active trading positions immediately (used in emergency halts).
        Returns:
            list: List of closure records/receipts.
        """
        pass


# =====================================================================
# 1. Mock Executor (Persistent Local Simulator)
# =====================================================================
class MockExecutor(BaseExecutionEngine):
    """
    A persistent local paper-trading simulation engine supporting multi-asset portfolios.
    Tracks multiple open positions (BTC, ETH, SOL, etc.) simultaneously in local JSON storage.
    """
    def __init__(self):
        self.portfolio_path = Config.DATABASE_DIR / "mock_portfolio.json"
        self.symbol = Config.TRADING_SYMBOL
        self.cash = 10000.0
        self.positions = {}  # { "BTC/USD": {"size": 0.1, "entry_price": 60000, "stop_loss": 58000, "take_profit": 64000} }
        
        # Legacy compatibility properties
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        self.load_portfolio()

    def load_portfolio(self):
        if self.portfolio_path.exists():
            try:
                with open(self.portfolio_path, "r") as f:
                    state = json.load(f)
                    self.cash = state.get("cash", 10000.0)
                    self.positions = state.get("positions", {})
                    # Load legacy fallback if present
                    if not self.positions and state.get("position_size", 0) > 0:
                        sym = state.get("symbol", Config.TRADING_SYMBOL)
                        self.positions[sym] = {
                            "size": state.get("position_size", 0.0),
                            "entry_price": state.get("entry_price", 0.0),
                            "stop_loss": state.get("stop_loss", 0.0),
                            "take_profit": state.get("take_profit", 0.0)
                        }
            except Exception as e:
                print(f"[MockExecutor] Error loading mock portfolio: {e}")
        else:
            self.save_portfolio()

    def save_portfolio(self):
        try:
            active_sym = Config.TRADING_SYMBOL
            active_pos = self.positions.get(active_sym, {})
            state = {
                "cash": self.cash,
                "positions": self.positions,
                # Legacy compatibility fields
                "position_size": active_pos.get("size", 0.0),
                "entry_price": active_pos.get("entry_price", 0.0),
                "stop_loss": active_pos.get("stop_loss", 0.0),
                "take_profit": active_pos.get("take_profit", 0.0),
                "equity": self.get_equity()
            }
            with open(self.portfolio_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"[MockExecutor] Error saving mock portfolio: {e}")

    def get_equity(self, current_prices: dict = None) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            price = pos["entry_price"]
            if current_prices and sym in current_prices:
                price = current_prices[sym]
            equity += pos["size"] * price
        return round(equity, 2)

    def get_portfolio_state(self, symbol: str = None) -> dict:
        target_sym = symbol or Config.TRADING_SYMBOL
        pos = self.positions.get(target_sym, {})
        has_pos = pos.get("size", 0.0) > 0.0
        
        return {
            "equity": self.get_equity(),
            "cash": round(self.cash, 2),
            "position_size": pos.get("size", 0.0),
            "entry_price": pos.get("entry_price", 0.0),
            "has_position": has_pos,
            "stop_loss": pos.get("stop_loss", 0.0),
            "take_profit": pos.get("take_profit", 0.0),
            "positions_count": len(self.positions),
            "all_positions": self.positions
        }

    def update_price_action(self, current_price: float, symbol: str = None) -> list:
        target_sym = symbol or Config.TRADING_SYMBOL
        if target_sym not in self.positions:
            return []

        pos = self.positions[target_sym]
        hit_sl = current_price <= pos["stop_loss"] and pos["stop_loss"] > 0
        hit_tp = current_price >= pos["take_profit"] and pos["take_profit"] > 0

        if hit_sl or hit_tp:
            exit_reason = "STOP_LOSS" if hit_sl else "TAKE_PROFIT"
            exit_price = pos["stop_loss"] if hit_sl else pos["take_profit"]
            print(f"[MockExecutor] {exit_reason} hit for {target_sym} at price: ${exit_price:.2f} (Current: ${current_price:.2f})")
            
            res = self.execute_order(
                action="SELL",
                size=pos["size"],
                stop_loss=0.0,
                take_profit=0.0,
                current_price=exit_price,
                symbol=target_sym
            )
            res["exit_trigger"] = exit_reason
            return [res]
        return []

    def execute_order(self, action: str, size: float, stop_loss: float, take_profit: float, current_price: float, symbol: str = None) -> dict:
        action = action.upper()
        target_sym = symbol or Config.TRADING_SYMBOL
        timestamp = datetime.now().isoformat()
        
        if action == "BUY":
            cost = size * current_price
            if cost > self.cash:
                print(f"[MockExecutor] BUY rejected for {target_sym}: Insufficient cash. Cost: ${cost:.2f}, Balance: ${self.cash:.2f}")
                return {"status": "FAILED", "reason": "Insufficient cash"}
                
            self.cash -= cost
            self.positions[target_sym] = {
                "size": size,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            self.save_portfolio()
            
            print(f"[MockExecutor] BUY SUCCESS: Executed {size:.4f} {target_sym} at ${current_price:.2f}. "
                  f"SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            
            return {
                "status": "SUCCESS",
                "symbol": target_sym,
                "order_id": f"mock_buy_{int(datetime.now().timestamp())}",
                "execution_price": current_price,
                "size": size,
                "pnl": 0.0,
                "timestamp": timestamp
            }
            
        elif action == "SELL":
            if target_sym not in self.positions:
                return {"status": "FAILED", "reason": f"No positions in {target_sym} to sell"}
                
            pos = self.positions.pop(target_sym)
            qty = pos["size"]
            entry_p = pos["entry_price"]
            
            revenue = size * current_price
            pnl = (current_price - entry_p) * size
            
            self.cash += revenue
            self.save_portfolio()
            
            print(f"[MockExecutor] SELL SUCCESS: Executed {size:.4f} {target_sym} at ${current_price:.2f}. PnL: ${pnl:.2f}")
            
            return {
                "status": "SUCCESS",
                "symbol": target_sym,
                "order_id": f"mock_sell_{int(datetime.now().timestamp())}",
                "execution_price": current_price,
                "size": size,
                "pnl": pnl,
                "timestamp": timestamp
            }
            
        return {"status": "FAILED", "reason": f"Unknown action: {action}"}

    def close_all_positions(self) -> list:
        receipts = []
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            res = self.execute_order("SELL", pos["size"], 0.0, 0.0, pos["entry_price"], symbol=sym)
            receipts.append(res)
        return receipts



# =====================================================================
# 2. Alpaca Executor (Alpaca-py Client Integration)
# =====================================================================
class AlpacaExecutor(BaseExecutionEngine):
    """
    Broker execution engine connected to Alpaca API.
    Handles account audits and submits Bracket Orders (with dynamic SL & TP).
    """
    def __init__(self):
        if not HAS_ALPACA_TRADING:
            raise ImportError("alpaca-py is required to run AlpacaExecutor.")
        
        self.symbol = Config.TRADING_SYMBOL.replace("/", "")
        self.client = TradingClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
            paper=Config.ALPACA_IS_PAPER
        )

    def get_portfolio_state(self) -> dict:
        account = self.client.get_account()
        
        # Check active position for symbol
        position_size = 0.0
        entry_price = 0.0
        try:
            position = self.client.get_open_position(self.symbol)
            position_size = float(position.qty)
            entry_price = float(position.avg_entry_price)
        except Exception:
            # get_open_position throws error if there is no position
            pass
            
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "position_size": position_size,
            "entry_price": entry_price,
            "has_position": position_size > 0.0
        }

    def execute_order(self, action: str, size: float, stop_loss: float, take_profit: float, current_price: float) -> dict:
        action = action.upper()
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        
        # Setup order parameters
        # For safety and professional operation, we use bracket orders for BUY positions
        if action == "BUY" and stop_loss > 0 and take_profit > 0:
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=size,
                side=side,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss)
            )
        else:
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=size,
                side=side,
                time_in_force=TimeInForce.GTC
            )

        try:
            order = self.client.submit_order(order_data)
            print(f"[AlpacaExecutor] Order submitted successfully. ID: {order.id}")
            
            # Alpaca execution is asynchronous; in live/paper trading we mock/assume current fill price
            # In production standard we would poll for execution status or parse webhook payloads.
            return {
                "status": "SUCCESS",
                "order_id": str(order.id),
                "execution_price": current_price,
                "size": size,
                "pnl": 0.0,  # calculated upon subsequent closures
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[AlpacaExecutor] Order submission failed: {e}")
            return {"status": "FAILED", "reason": str(e)}

    def close_all_positions(self) -> list:
        try:
            print("[AlpacaExecutor] Emergency circuit breaker! Closing open position...")
            closed = self.client.close_position(self.symbol)
            return [closed]
        except Exception as e:
            print(f"[AlpacaExecutor] Position closure error: {e}")
            return []


# =====================================================================
# 3. CCXT Executor (Crypto Exchange Integration)
# =====================================================================
class CCXTExecutor(BaseExecutionEngine):
    """
    Exchange execution engine powered by CCXT.
    Supports spot trading with local order mapping.
    """
    def __init__(self, data_fetcher):
        if ccxt is None:
            raise ImportError("ccxt is required to run CCXTExecutor.")
        self.fetcher = data_fetcher
        self.client = data_fetcher.ccxt_client
        self.symbol = Config.TRADING_SYMBOL.replace("USDT", "/USDT").replace("USD", "/USD") if "/" not in Config.TRADING_SYMBOL else Config.TRADING_SYMBOL
        self.quote_asset = self.symbol.split("/")[-1]
        self.base_asset = self.symbol.split("/")[0]

    def get_portfolio_state(self) -> dict:
        balance = self.client.fetch_balance()
        
        # Get free cash (quote asset, e.g., USDT)
        cash = balance.get(self.quote_asset, {}).get("free", 0.0)
        
        # Get active position (base asset, e.g., BTC)
        position_size = balance.get(self.base_asset, {}).get("total", 0.0)
        
        ticker = self.client.fetch_ticker(self.symbol)
        current_price = ticker.get('last', 1.0)
        equity = cash + (position_size * current_price)
        
        # Calculate a mock average entry price based on recent order history
        # (Since spot balances do not store entry prices directly)
        entry_price = current_price
        try:
            trades = self.client.fetch_my_trades(self.symbol, limit=5)
            if trades:
                buys = [t for t in trades if t['side'] == 'buy']
                if buys:
                    entry_price = buys[-1]['price']
        except Exception:
            pass

        return {
            "equity": equity,
            "cash": cash,
            "position_size": position_size,
            "entry_price": entry_price,
            "has_position": position_size > 0.0001
        }

    def execute_order(self, action: str, size: float, stop_loss: float, take_profit: float, current_price: float) -> dict:
        action = action.upper()
        side = 'buy' if action == "BUY" else 'sell'
        
        try:
            # Standard Spot Order execution
            order = self.client.create_market_order(self.symbol, side, size)
            print(f"[CCXTExecutor] Spot Order placed: {order['id']}")
            
            pnl = 0.0
            if action == "SELL":
                # Fetch entry price from balance calculations or recent transactions to log PnL
                try:
                    trades = self.client.fetch_my_trades(self.symbol, limit=2)
                    if len(trades) >= 2:
                        # simple pnl estimate
                        pnl = (current_price - trades[0]['price']) * size
                except Exception:
                    pass

            return {
                "status": "SUCCESS",
                "order_id": order['id'],
                "execution_price": order.get('price', current_price),
                "size": size,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[CCXTExecutor] Order execution failed: {e}")
            return {"status": "FAILED", "reason": str(e)}

    def close_all_positions(self) -> list:
        state = self.get_portfolio_state()
        if state["has_position"]:
            res = self.execute_order("SELL", state["position_size"], 0.0, 0.0, state["entry_price"])
            return [res]
        return []
