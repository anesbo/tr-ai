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


# Instantiate central orchestrator instance
bot = TradingBotOrchestrator(mode="paper", strategy_name=Config.STRATEGY_TYPE)


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
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/trigger_tick":
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
                    "timeframe": Config.TIMEFRAME
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
                "signal": signal
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
    print(f"🚀 AI Trading Bot Web Dashboard is running!")
    print(f"🌐 Access Interface at: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dashboard Web Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    run_server(port)
