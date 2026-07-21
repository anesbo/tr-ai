import json
from datetime import datetime, timedelta
from config import Config


class RiskManager:
    """
    Risk Management & Circuit Breakers Layer.
    Ensures non-negotiable safety rules override any AI decision:
      - Validates trading signals.
      - Calculates risk-adjusted position sizes (max 2% account equity risk).
      - Computes precise Stop-Loss (SL) and Take-Profit (TP) points.
      - Implements a hard 5% daily drawdown circuit breaker (halts trading for 24 hours).
    """

    def __init__(self):
        self.max_risk_pct = Config.MAX_EQUITY_RISK_PCT  # e.g., 0.02 (2%)
        self.drawdown_limit_pct = Config.DRAWDOWN_LIMIT_PCT  # e.g., 0.05 (5%)
        self.state_file_path = Config.DATABASE_DIR / "risk_state.json"
        
        # Load or initialize daily risk trackers
        self.daily_start_equity = 0.0
        self.daily_start_time = None
        self.circuit_breaker_active = False
        self.halt_until = None
        
        self.load_risk_state()

    def load_risk_state(self):
        """Loads persistent risk limits and circuit breaker statuses from local database."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r") as f:
                    state = json.load(f)
                    self.daily_start_equity = state.get("daily_start_equity", 0.0)
                    
                    start_time_str = state.get("daily_start_time")
                    if start_time_str:
                        self.daily_start_time = datetime.fromisoformat(start_time_str)
                        
                    self.circuit_breaker_active = state.get("circuit_breaker_active", False)
                    
                    halt_until_str = state.get("halt_until")
                    if halt_until_str:
                        self.halt_until = datetime.fromisoformat(halt_until_str)
            except Exception as e:
                print(f"[RiskManager] Error loading risk state: {e}")

    def save_risk_state(self):
        """Saves current daily tracking variables to avoid state losses on bot restart."""
        try:
            state = {
                "daily_start_equity": self.daily_start_equity,
                "daily_start_time": self.daily_start_time.isoformat() if self.daily_start_time else None,
                "circuit_breaker_active": self.circuit_breaker_active,
                "halt_until": self.halt_until.isoformat() if self.halt_until else None
            }
            with open(self.state_file_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"[RiskManager] Error saving risk state: {e}")

    def check_circuit_breakers(self, current_equity: float) -> bool:
        """
        Monitors account health.
        Triggers the circuit breaker if the daily drawdown exceeds limits.
        Returns:
            bool: True if trading is allowed, False if trading is halted.
        """
        now = datetime.now()

        # 1. Reset daily tracker if 24 hours have passed
        if not self.daily_start_time or (now - self.daily_start_time) >= timedelta(days=1):
            self.daily_start_equity = current_equity
            self.daily_start_time = now
            print(f"[RiskManager] Daily equity baseline reset to ${self.daily_start_equity:.2f}")
            self.save_risk_state()

        # 2. Check if currently halted due to circuit breaker
        if self.circuit_breaker_active:
            if self.halt_until and now >= self.halt_until:
                # Reset circuit breaker
                self.circuit_breaker_active = False
                self.halt_until = None
                self.daily_start_equity = current_equity
                self.daily_start_time = now
                print("[RiskManager] Circuit breaker cooldown expired. Re-enabling trading operations.")
                self.save_risk_state()
                return True
            else:
                remaining = self.halt_until - now if self.halt_until else timedelta(0)
                print(f"[RiskManager] WARNING: Circuit breaker is ACTIVE. Trading halted. Cooldown ends in: {remaining}")
                return False

        # 3. Calculate current daily drawdown
        drawdown_pct = 0.0
        if self.daily_start_equity > 0:
            drawdown_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity

        if drawdown_pct >= self.drawdown_limit_pct:
            self.circuit_breaker_active = True
            self.halt_until = now + timedelta(days=1)
            print(f"[RiskManager] CRITICAL: Daily drawdown threshold hit! Drawdown: {drawdown_pct*100:.2f}%. "
                  f"Halt trading until {self.halt_until.strftime('%Y-%m-%d %H:%M:%S')}")
            self.save_risk_state()
            return False

        return True

    def calculate_position_size(self, current_price: float, atr: float, account_equity: float) -> float:
        """
        Calculates position sizing according to portfolio risk parameters:
          - Max 2% risk of account equity based on Stop-Loss distance.
          - Size = (Equity * Risk%) / SL_Distance.
        """
        # Determine Stop-Loss distance in currency units
        # If ATR is available, use volatility-based stop-loss: 2 * ATR
        # Otherwise, fallback to a fixed percentage stop-loss
        if atr > 0:
            sl_distance = 2.0 * atr
        else:
            sl_distance = current_price * Config.STOP_LOSS_PCT

        if sl_distance <= 0:
            return 0.0

        # Account risk amount (e.g. $1000 * 2% = $20)
        risk_capital = account_equity * self.max_risk_pct
        
        # Calculate nominal size in asset base units
        # Example: Risk $20 / $50 SL distance = 0.4 units
        asset_size = risk_capital / sl_distance
        nominal_value = asset_size * current_price

        # Safeguard: Never allow nominal exposure to exceed 95% of equity (no leverage safety rule)
        max_nominal_allowed = account_equity * 0.95
        if nominal_value > max_nominal_allowed:
            asset_size = max_nominal_allowed / current_price
            
        return round(asset_size, 6)

    def calculate_stops(self, entry_price: float, atr: float, direction: str) -> tuple:
        """
        Calculates precise Stop-Loss and Take-Profit prices.
        Supports volatility-based (ATR) or fixed percentage boundaries.
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        if atr > 0:
            # Dynamic volatility-based stop/take profit
            # SL = 2 * ATR, TP = 4 * ATR (1:2 Risk-to-Reward Ratio)
            sl_dist = 2.0 * atr
            tp_dist = 4.0 * atr
        else:
            # Fixed percentage stop/take profit
            sl_dist = entry_price * Config.STOP_LOSS_PCT
            tp_dist = entry_price * Config.TAKE_PROFIT_PCT

        if direction == "BUY":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        elif direction == "SELL":
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist
        else:
            stop_loss = entry_price
            take_profit = entry_price

        return round(stop_loss, 4), round(take_profit, 4)

    def validate_risk(self, signal: dict, current_price: float, atr: float, account_equity: float) -> dict:
        """
        Validates the generated strategy signal against risk rules.
        Overrides signal action to HOLD if parameters violate safety checks.
        """
        action = signal.get("action", "HOLD")
        if action == "HOLD":
            return {"action": "HOLD", "size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "reason": "No entry action"}

        # Calculate position size
        size = self.calculate_position_size(current_price, atr, account_equity)
        
        if size <= 0:
            return {"action": "HOLD", "size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "reason": "Calculated position size is 0."}
            
        # Calculate stops
        stop_loss, take_profit = self.calculate_stops(current_price, atr, action)
        
        # Verify stop loss alignment
        if action == "BUY" and stop_loss >= current_price:
            return {"action": "HOLD", "size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "reason": "Invalid Buy Stop-Loss price setup."}
        if action == "SELL" and stop_loss <= current_price:
            return {"action": "HOLD", "size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "reason": "Invalid Sell Stop-Loss price setup."}

        return {
            "action": action,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": signal.get("reason", "Risk validation cleared.")
        }
