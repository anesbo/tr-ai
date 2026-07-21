import json
import os
import random
from collections import deque
import urllib.request
import urllib.error
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from config import Config

# Handle optional PyTorch import for DQN Strategy
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


# =====================================================================
# Base Strategy Class
# =====================================================================
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, market_state: dict, portfolio_state: dict) -> dict:
        """
        Analyzes the market and portfolio states.
        Returns:
            dict: {
                "action": "BUY" | "SELL" | "HOLD",
                "confidence": float (0.0 to 1.0),
                "reason": str,
                "reflection": str (post-analysis/lessons learned)
            }
        """
        pass


# =====================================================================
# 1. Technical Indicators Rule-Based Strategy
# =====================================================================
class RuleBasedStrategy(BaseStrategy):
    """
    Standard quantitative strategy utilizing RSI, MACD, and Bollinger Bands.
    Learns/updates thresholds dynamically from SelfLearningEngine via local storage.
    """
    def __init__(self):
        self.learned_params_path = Config.DATABASE_DIR / "learned_params.json"
        self.rsi_buy_threshold = 30.0
        self.rsi_sell_threshold = 70.0
        self.macd_trigger_active = True
        self.bb_trigger_active = True
        self.load_learned_parameters()

    def load_learned_parameters(self):
        """Loads self-learned parameters optimized by the SelfLearningEngine."""
        if self.learned_params_path.exists():
            try:
                with open(self.learned_params_path, "r") as f:
                    params = json.load(f)
                    self.rsi_buy_threshold = params.get("rsi_buy_threshold", self.rsi_buy_threshold)
                    self.rsi_sell_threshold = params.get("rsi_sell_threshold", self.rsi_sell_threshold)
                    self.macd_trigger_active = params.get("macd_trigger_active", self.macd_trigger_active)
                    self.bb_trigger_active = params.get("bb_trigger_active", self.bb_trigger_active)
                    print(f"[RuleBasedStrategy] Loaded self-learned parameters: RSI BUY: {self.rsi_buy_threshold:.1f}, RSI SELL: {self.rsi_sell_threshold:.1f}")
            except Exception as e:
                print(f"[RuleBasedStrategy] Error loading learned parameters: {e}")

    def generate_signal(self, market_state: dict, portfolio_state: dict) -> dict:
        rsi = market_state.get('rsi', 50.0)
        macd = market_state.get('macd', 0.0)
        macd_signal = market_state.get('macd_signal', 0.0)
        close = market_state.get('close', 0.0)
        bb_lower = market_state.get('bb_lower', 0.0)
        bb_upper = market_state.get('bb_upper', 0.0)
        
        has_position = portfolio_state.get('has_position', False)
        
        reasons = []
        buy_score = 0
        sell_score = 0
        
        # A. RSI triggers
        if rsi <= self.rsi_buy_threshold:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi:.2f} <= {self.rsi_buy_threshold})")
        elif rsi >= self.rsi_sell_threshold:
            sell_score += 2
            reasons.append(f"RSI overbought ({rsi:.2f} >= {self.rsi_sell_threshold})")
            
        # B. MACD Crossover triggers
        if self.macd_trigger_active:
            if macd > macd_signal:
                buy_score += 1
                reasons.append("MACD above signal (Bullish Crossover)")
            elif macd < macd_signal:
                sell_score += 1
                reasons.append("MACD below signal (Bearish Crossover)")
                
        # C. Bollinger Bands breakout triggers
        if self.bb_trigger_active:
            if close <= bb_lower:
                buy_score += 2
                reasons.append(f"Price ({close:.2f}) touched lower Bollinger Band ({bb_lower:.2f})")
            elif close >= bb_upper:
                sell_score += 2
                reasons.append(f"Price ({close:.2f}) touched upper Bollinger Band ({bb_upper:.2f})")

        action = "HOLD"
        confidence = 0.5
        
        if not has_position and buy_score >= 3:
            action = "BUY"
            confidence = min(0.95, 0.5 + (buy_score * 0.1))
        elif has_position and sell_score >= 3:
            action = "SELL"
            confidence = min(0.95, 0.5 + (sell_score * 0.1))
            
        reason_str = ", ".join(reasons) if reasons else "No clear signal triggers met."
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason_str,
            "reflection": "Executing baseline indicator triggers."
        }


# =====================================================================
# 2. PyTorch Deep Q-Network (DQN) Reinforcement Learning Strategy
# =====================================================================
if HAS_PYTORCH:
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(QNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            
        def forward(self, x):
            return self.network(x)
else:
    class QNetwork:
        pass


class DQNStrategy(BaseStrategy):
    """
    Deep Q-Network Reinforcement Learning agent.
    Learns trade policy mapping (Market Indicators + Position Status) -> Actions.
    """
    def __init__(self, state_dim=6, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 0 = HOLD, 1 = BUY, 2 = SELL
        self.model_path = Config.MODELS_DIR / "dqn_weights.pt"
        self.epsilon = 0.15  # Exploration probability
        self.gamma = 0.99
        
        if not HAS_PYTORCH:
            print("[DQNStrategy] WARNING: PyTorch not found. DQN strategy falls back to HOLD signals.")
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        
        self.load_model()

    def load_model(self):
        if self.model_path.exists():
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"[DQNStrategy] Loaded neural network weights from {self.model_path.name}")
            except Exception as e:
                print(f"[DQNStrategy] Error loading model: {e}")

    def save_model(self):
        if HAS_PYTORCH:
            try:
                torch.save(self.policy_net.state_dict(), self.model_path)
                print(f"[DQNStrategy] Neural network weights saved to {self.model_path}")
            except Exception as e:
                print(f"[DQNStrategy] Failed to save neural network weights: {e}")

    def _get_state_vector(self, market_state: dict, portfolio_state: dict) -> np.ndarray:
        """Converts market indicators and holding state to a normalized numpy state vector."""
        close = market_state.get('close', 1.0)
        rsi = market_state.get('rsi', 50.0) / 100.0  # Normalize 0-1
        
        # MACD normalized by price
        macd = market_state.get('macd', 0.0) / close
        
        # Distance to upper/lower BB
        bb_middle = market_state.get('bb_middle', close)
        bb_upper = market_state.get('bb_upper', close)
        bb_lower = market_state.get('bb_lower', close)
        bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-9)
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
        
        # ATR volatility ratio
        atr_ratio = market_state.get('atr', 0.0) / close
        
        # Holding state
        has_position = 1.0 if portfolio_state.get('has_position', False) else 0.0
        
        return np.array([rsi, macd, bb_width, bb_pos, atr_ratio, has_position], dtype=np.float32)

    def generate_signal(self, market_state: dict, portfolio_state: dict) -> dict:
        if not HAS_PYTORCH:
            return {"action": "HOLD", "confidence": 1.0, "reason": "PyTorch missing", "reflection": "N/A"}
            
        state = self._get_state_vector(market_state, portfolio_state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
            reason = "Exploration action (Epsilon-greedy selection)"
        else:
            self.policy_net.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                action_idx = q_values.argmax(dim=1).item()
                reason = f"Model execution (Max Q-value prediction, Q-Values: {q_values.cpu().numpy()[0]})"
                
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action = action_map[action_idx]
        
        # Guard action context: cannot BUY if holding, cannot SELL if empty
        has_position = portfolio_state.get('has_position', False)
        if action == "BUY" and has_position:
            action = "HOLD"
            reason += " | Suppressed BUY (already holding position)"
        elif action == "SELL" and not has_position:
            action = "HOLD"
            reason += " | Suppressed SELL (no position to close)"
            
        return {
            "action": action,
            "confidence": 0.85,
            "reason": reason,
            "reflection": "Executing Deep Q-Network state decision agent."
        }

    def train_step(self, batch_size=32):
        """Standard DQN training step from memory replay buffer."""
        if not HAS_PYTORCH or len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        self.policy_net.train()
        # Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)
        
        # Target Q = r + gamma * max_a Q_target(s', a)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network weights
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * local_param.data)


# =====================================================================
# 3. Gemini LLM Market Analysis & Self-Reflective Strategy
# =====================================================================
class LLMStrategy(BaseStrategy):
    """
    AI decision engine powered by Google Gemini.
    Generates strategic signals using technical metrics and historical performance contexts.
    """
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if not self.api_key:
            print("[LLMStrategy] WARNING: GEMINI_API_KEY environment variable is missing. Strategy outputs HOLD.")

    def generate_signal(self, market_state: dict, portfolio_state: dict) -> dict:
        if not self.api_key:
            return {
                "action": "HOLD",
                "confidence": 1.0,
                "reason": "Gemini API key is not configured.",
                "reflection": "Awaiting API configuration."
            }

        # Gather last trades context to support historical reflection
        past_trades_summary = self._get_past_trades_summary()

        prompt = f"""You are an expert Quantitative Trader. Analyze the market indicators and make an autonomous, structured trading decision.

### MARKET DATA ({market_state.get('symbol', 'Asset')} - {market_state.get('timeframe', '1h')} timeframe):
- Current Price: {market_state.get('close', 0.0)}
- Relative Strength Index (RSI): {market_state.get('rsi', 50.0):.2f}
- MACD Line: {market_state.get('macd', 0.0):.4f} (Signal: {market_state.get('macd_signal', 0.0):.4f})
- Bollinger Bands: Middle: {market_state.get('bb_middle', 0.0):.2f}, Upper: {market_state.get('bb_upper', 0.0):.2f}, Lower: {market_state.get('bb_lower', 0.0):.2f}
- ATR Volatility (Average True Range): {market_state.get('atr', 0.0):.4f}

### PORTFOLIO STATE:
- Balance: ${portfolio_state.get('equity', 1000.0):.2f}
- Holds Position: {portfolio_state.get('has_position', False)}
- Position Size: {portfolio_state.get('position_size', 0.0)}
- Average Entry Price: {portfolio_state.get('entry_price', 0.0)}

### RECENT TRADE PERFORMANCE SUMMARY:
{past_trades_summary}

### DECISION PROTOCOLS:
1. If holds position is FALSE: Decide if we should BUY or HOLD.
2. If holds position is TRUE: Decide if we should SELL (close position) or HOLD.
3. Incorporate recent lessons from the Trade Summary reflection to avoid repeat mistakes.

You MUST respond strictly with a raw JSON object matching the JSON schema below. DO NOT enclose inside markdown ```json blocks. No extra text.

JSON Schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence detailing indicator reasoning>",
  "reflection": "<short note evaluating strategy adjustments or lessons from past performance>"
}}"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
            text_response = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            decision = json.loads(text_response)
            
            # Action structural validation
            action = decision.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
                
            return {
                "action": action,
                "confidence": float(decision.get("confidence", 0.5)),
                "reason": decision.get("reason", "Parsed LLM signal."),
                "reflection": decision.get("reflection", "N/A")
            }
        except Exception as e:
            print(f"[LLMStrategy] Gemini execution API call error: {e}")
            return {
                "action": "HOLD",
                "confidence": 1.0,
                "reason": f"Execution error: {e}",
                "reflection": "N/A"
            }

    def _get_past_trades_summary(self) -> str:
        """Reads local ledger file and aggregates statistics for prompt injection."""
        if not Config.TRADE_LEDGER_PATH.exists():
            return "No previous trades recorded."
            
        try:
            with open(Config.TRADE_LEDGER_PATH, "r") as f:
                trades = json.load(f)
            if not trades:
                return "No previous trades recorded."
                
            recent = trades[-5:]  # Look at last 5 trades
            summary = []
            for i, t in enumerate(recent):
                action = t.get("action", "N/A")
                price = t.get("execution_price", 0.0)
                pnl = t.get("pnl", 0.0)
                refl = t.get("strategy_reflection", "None")
                summary.append(f"- Trade {i+1}: {action} at ${price:.2f} | PnL: ${pnl:.2f} | Reflection: {refl}")
            return "\n".join(summary)
        except Exception as e:
            return f"Error reading historical logs: {e}"


# =====================================================================
# 4. Self-Learning Engine (Adaptive Performance Optimizer)
# =====================================================================
class SelfLearningEngine:
    """
    Self-Learning Module.
    Reviews `trade_ledger.json` historical outcomes to dynamically adapt
    underlying strategy configurations (RSI thresholds, indicators, risk sizes).
    """
    def __init__(self):
        self.ledger_path = Config.TRADE_LEDGER_PATH
        self.learned_params_path = Config.DATABASE_DIR / "learned_params.json"

    def audit_and_optimize(self):
        """Analyzes historical trades and saves optimized parameters if profitable adaptations are discovered."""
        if not self.ledger_path.exists():
            return
            
        try:
            with open(self.ledger_path, "r") as f:
                trades = json.load(f)
                
            if len(trades) < 5:
                # Need a minimum baseline profile of trades before optimization
                return
                
            df = pd.DataFrame(trades)
            # Filter trades with finalized returns/PnL
            completed_trades = df[df['pnl'].notna() & (df['pnl'] != 0.0)]
            if len(completed_trades) < 3:
                return
                
            total_trades = len(completed_trades)
            winning_trades = completed_trades[completed_trades['pnl'] > 0]
            win_rate = len(winning_trades) / total_trades
            
            # Simple adaptive thresholds heuristic based on recent metrics:
            # If win rate is low (< 45%), let's make indicators more conservative:
            # - Tighten RSI buy bounds (buy lower)
            # - Tighten RSI sell bounds (sell higher)
            # If win rate is high (> 60%), we can expand trading bounds slightly.
            
            current_rsi_buy = 30.0
            current_rsi_sell = 70.0
            
            if self.learned_params_path.exists():
                with open(self.learned_params_path, "r") as f:
                    curr_params = json.load(f)
                    current_rsi_buy = curr_params.get("rsi_buy_threshold", 30.0)
                    current_rsi_sell = curr_params.get("rsi_sell_threshold", 70.0)

            new_rsi_buy = current_rsi_buy
            new_rsi_sell = current_rsi_sell
            
            if win_rate < 0.45:
                # Shift thresholds to require more extreme/oversold prices before entering buy, and sell faster
                new_rsi_buy = max(20.0, current_rsi_buy - 1.0)
                new_rsi_sell = max(65.0, current_rsi_sell - 1.0)
                action_taken = "lowered RSI Buy/Sell levels to require more conservative entries (Defensive Mode)"
            elif win_rate > 0.60:
                # Loosen slightly to capture more volume as current indicators are working
                new_rsi_buy = min(35.0, current_rsi_buy + 0.5)
                new_rsi_sell = min(75.0, current_rsi_sell + 0.5)
                action_taken = "expanded RSI parameters slightly to catch more trades (Growth Mode)"
            else:
                action_taken = "retained existing parameter balances"
                
            optimized_params = {
                "rsi_buy_threshold": new_rsi_buy,
                "rsi_sell_threshold": new_rsi_sell,
                "macd_trigger_active": True,
                "bb_trigger_active": True,
                "win_rate_last_audit": win_rate,
                "total_audited_trades": total_trades,
                "last_audit_time": str(pd.Timestamp.now())
            }
            
            with open(self.learned_params_path, "w") as f:
                json.dump(optimized_params, f, indent=4)
                
            print(f"[SelfLearningEngine] Audit completed. Win Rate: {win_rate*100:.1f}%. Strategy {action_taken}.")
        except Exception as e:
            print(f"[SelfLearningEngine] Error during audit loop: {e}")
