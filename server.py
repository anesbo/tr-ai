import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from data_fetcher import DataFetcher
from risk_manager import RiskManager
from execution import MockExecutor, AlpacaExecutor, CCXTExecutor
from strategy import RuleBasedStrategy, DQNStrategy, LLMStrategy, SelfLearningEngine
from main import TradingBotOrchestrator


import threading
import time

# Instantiate central orchestrator instance
bot = TradingBotOrchestrator(mode="paper", strategy_name=Config.STRATEGY_TYPE)


class AutoTraderManager:
    """
    Background Autonomous Trading Worker.
    Continuously executes live tick evaluations, risk checks, and auto orders.
    """
    def __init__(self, orchestrator, interval_seconds=5):
        self.orchestrator = orchestrator
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print("[AutoTrader] Autonomous Auto-Pilot Trading Started!")

    def stop(self):
        if self.is_running:
            self.is_running = False
            print("[AutoTrader] Autonomous Auto-Pilot Trading Paused.")

    def _run_loop(self):
        while self.is_running:
            try:
                self.orchestrator.run_live_cycle()
            except Exception as e:
                print(f"[AutoTrader] Error in live tick loop: {e}")
            time.sleep(self.interval_seconds)


autotrader = AutoTraderManager(bot, interval_seconds=5)


class DashboardAPIHandler(SimpleHTTPRequestHandler):
    """
    Custom HTTP Request Handler serving both the static Dashboard frontend 
    and REST API endpoints for real-time monitoring and control.
    """

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.serve_dashboard_html()
        elif self.path == "/api/status":
            self.serve_api_status()
        elif self.path == "/api/chart":
            self.serve_api_chart()
        elif self.path == "/api/trades":
            self.serve_api_trades()
        elif self.path == "/api/analysis":
            self.serve_api_analysis()
        elif self.path == "/api/autotrader/status":
            self.send_json_response({"autotrader_active": autotrader.is_running})
        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            req_data = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            req_data = {}

        if self.path == "/api/autotrader/start":
            try:
                autotrader.start()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "Auto-Pilot Autonomous Trading Enabled! The AI is now buying and selling automatically.",
                    "autotrader_active": True
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif self.path == "/api/autotrader/stop":
            try:
                autotrader.stop()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "Auto-Pilot Autonomous Trading Paused.",
                    "autotrader_active": False
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif self.path == "/api/trigger_tick":
            try:
                bot.run_live_cycle()
                self.send_json_response({"status": "SUCCESS", "message": "Live tick cycle executed."})
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif self.path == "/api/trigger_audit":
            try:
                bot.learning_engine.audit_and_optimize()
                self.send_json_response({"status": "SUCCESS", "message": "Self-learning audit executed."})
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif self.path == "/api/config/wallet":
            try:
                Config.update_wallet_config(req_data)
                # Re-initialize bot executor and data fetcher clients
                bot._init_executor()
                bot.data_fetcher._init_clients()
                
                # If mock balance adjustment requested
                if "mock_cash" in req_data and isinstance(bot.executor, MockExecutor):
                    try:
                        bot.executor.cash = float(req_data["mock_cash"])
                        bot.executor.save_portfolio()
                    except Exception:
                        pass

                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "Wallet and Exchange connection settings updated successfully!",
                    "engine": Config.EXECUTION_ENGINE,
                    "symbol": Config.TRADING_SYMBOL,
                    "web3_address": Config.WEB3_WALLET_ADDRESS,
                    "web3_network": Config.WEB3_NETWORK
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif self.path == "/api/config/symbol":
            try:
                new_symbol = req_data.get("symbol", "BTC/USDT").strip().upper()
                # Format to standard slash pair if needed e.g. BTCUSDT -> BTC/USDT
                if "/" not in new_symbol and new_symbol.endswith("USDT"):
                    new_symbol = new_symbol[:-4] + "/USDT"
                elif "/" not in new_symbol and new_symbol.endswith("USD"):
                    new_symbol = new_symbol[:-3] + "/USD"

                Config.update_symbol(new_symbol)
                bot.data_fetcher.symbol = Config.TRADING_SYMBOL
                if hasattr(bot.executor, "symbol"):
                    bot.executor.symbol = Config.TRADING_SYMBOL
                
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"Trading pair changed to {Config.TRADING_SYMBOL}",
                    "symbol": Config.TRADING_SYMBOL
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        else:
            self.send_json_response({"error": "Endpoint not found"}, status=404)

    def serve_dashboard_html(self):
        html_file = Config.BASE_DIR / "dashboard" / "index.html"
        if not html_file.exists():
            self.send_error(404, "Dashboard HTML file not found.")
            return
        
        with open(html_file, "rb") as f:
            content = f.read()
            
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_api_status(self):
        try:
            # 1. Fetch latest market data state
            latest_market = bot.data_fetcher.get_latest_market_state()
            current_price = latest_market.get("close", 0.0)
            
            # 2. Get active portfolio state
            if isinstance(bot.executor, MockExecutor):
                portfolio = bot.executor.get_portfolio_state()
                portfolio["equity"] = bot.executor.get_equity(current_price)
            else:
                portfolio = bot.executor.get_portfolio_state()
                
            # 3. Read learned parameters
            learned_path = Config.DATABASE_DIR / "learned_params.json"
            learned_params = {}
            if learned_path.exists():
                try:
                    with open(learned_path, "r") as f:
                        learned_params = json.load(f)
                except Exception:
                    pass

            # 4. Generate strategy signal for current state
            strategy_portfolio = {
                "equity": portfolio.get("equity", 10000.0),
                "has_position": portfolio.get("has_position", False),
                "position_size": portfolio.get("position_size", 0.0),
                "entry_price": portfolio.get("entry_price", 0.0)
            }
            signal = bot.strategy.generate_signal(latest_market, strategy_portfolio)

            status_payload = {
                "config": {
                    "paper_trading": Config.PAPER_TRADING,
                    "engine": Config.EXECUTION_ENGINE,
                    "strategy": Config.STRATEGY_TYPE,
                    "symbol": Config.TRADING_SYMBOL,
                    "timeframe": Config.TIMEFRAME,
                    "wallet_address": Config.WEB3_WALLET_ADDRESS,
                    "wallet_network": Config.WEB3_NETWORK,
                    "wallet_type": Config.WALLET_TYPE,
                    "ccxt_exchange": Config.CCXT_EXCHANGE_ID,
                    "autotrader_active": autotrader.is_running
                },
                "portfolio": portfolio,
                "latest_price": current_price,
                "risk": {
                    "circuit_breaker_active": bot.risk_manager.circuit_breaker_active,
                    "daily_start_equity": bot.risk_manager.daily_start_equity,
                    "drawdown_limit_pct": bot.risk_manager.drawdown_limit_pct
                },
                "learned": {
                    "rsi_buy": learned_params.get("rsi_buy_threshold", 30.0),
                    "rsi_sell": learned_params.get("rsi_sell_threshold", 70.0),
                    "last_win_rate": learned_params.get("win_rate_last_audit", 0.0)
                },
                "signal": signal,
                "autotrader_active": autotrader.is_running
            }
            
            self.send_json_response(status_payload)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def serve_api_chart(self):
        try:
            df = bot.data_fetcher.fetch_historical_data(limit=50)
            df_ind = bot.data_fetcher.calculate_indicators(df)
            
            payload = {
                "timestamps": [str(t)[11:16] for t in df_ind['timestamp']],
                "close": df_ind['close'].tolist(),
                "bb_upper": df_ind['bb_upper'].tolist(),
                "bb_lower": df_ind['bb_lower'].tolist(),
                "rsi": df_ind['rsi'].tolist(),
                "macd": df_ind['macd'].tolist()
            }
            self.send_json_response(payload)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def serve_api_trades(self):
        trades = []
        if Config.TRADE_LEDGER_PATH.exists():
            try:
                with open(Config.TRADE_LEDGER_PATH, "r") as f:
                    trades = json.load(f)
            except Exception:
                pass
        self.send_json_response(trades)

    def serve_api_analysis(self):
        try:
            latest_market = bot.data_fetcher.get_latest_market_state()
            current_price = latest_market.get("close", 0.0)
            rsi = latest_market.get("rsi", 50.0)
            macd = latest_market.get("macd", 0.0)
            macd_signal = latest_market.get("macd_signal", 0.0)
            bb_upper = latest_market.get("bb_upper", 0.0)
            bb_lower = latest_market.get("bb_lower", 0.0)
            atr = latest_market.get("atr", 0.0)

            # Portfolio state
            if isinstance(bot.executor, MockExecutor):
                portfolio = bot.executor.get_portfolio_state()
                equity = bot.executor.get_equity(current_price)
            else:
                portfolio = bot.executor.get_portfolio_state()
                equity = portfolio.get("equity", 10000.0)

            strategy_portfolio = {
                "equity": equity,
                "has_position": portfolio.get("has_position", False),
                "position_size": portfolio.get("position_size", 0.0),
                "entry_price": portfolio.get("entry_price", 0.0)
            }

            signal = bot.strategy.generate_signal(latest_market, strategy_portfolio)

            # Calculated risk bounds
            sl_price, tp_price = bot.risk_manager.calculate_stops(current_price, atr, "BUY")

            # Technical stance calculation
            stance = "NEUTRAL"
            if rsi < 40 and macd > macd_signal:
                stance = "BULLISH ACCUMULATION"
            elif rsi > 60 and macd < macd_signal:
                stance = "BEARISH DISTRIBUTION"
            elif signal.get("action") == "BUY":
                stance = "STRONG BULLISH BREAKOUT"
            elif signal.get("action") == "SELL":
                stance = "STRONG BEARISH EXIT"

            # Formulate structured AI Analyst breakdown
            commentary = (
                f"Coin {Config.TRADING_SYMBOL} is trading at ${current_price:,.2f}. "
                f"RSI is currently {rsi:.1f} ({'Oversold' if rsi <= 30 else 'Overbought' if rsi >= 70 else 'Neutral range'}). "
                f"MACD line ({macd:.2f}) vs Signal ({macd_signal:.2f}) indicates "
                f"{'Bullish Crossover' if macd > macd_signal else 'Bearish Crossover'}. "
                f"Price is positioned between Lower BB (${bb_lower:,.2f}) and Upper BB (${bb_upper:,.2f}). "
                f"Entry Rationale: {signal.get('reason', 'Awaiting target threshold setup')}. "
                f"Calculated SL Target: ${sl_price:,.2f} | TP Target: ${tp_price:,.2f}."
            )

            payload = {
                "symbol": Config.TRADING_SYMBOL,
                "price": current_price,
                "timeframe": Config.TIMEFRAME,
                "stance": stance,
                "signal": signal,
                "metrics": {
                    "rsi": round(rsi, 2),
                    "macd": round(macd, 4),
                    "macd_signal": round(macd_signal, 4),
                    "bb_upper": round(bb_upper, 2),
                    "bb_lower": round(bb_lower, 2),
                    "atr": round(atr, 2)
                },
                "risk_plan": {
                    "stop_loss": round(sl_price, 2),
                    "take_profit": round(tp_price, 2),
                    "max_risk_pct": bot.risk_manager.max_risk_pct * 100
                },
                "commentary": commentary
            }
            self.send_json_response(payload)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server(port=5000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardAPIHandler)
    print(f"\n=======================================================")
    print(f"[+] AI Trading Bot Web Dashboard is running!")
    print(f"[+] Access Interface at: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dashboard Web Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    run_server(port)
