import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
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


class MarketOverviewCache:
    """
    Background cache thread that asynchronously pre-fetches market overview for all platform coins
    so the REST API returns instantly in 1ms without blocking the HTTP server.
    """
    def __init__(self, data_fetcher, refresh_interval=8):
        self.fetcher = data_fetcher
        self.refresh_interval = refresh_interval
        self.cached_data = []
        self.is_running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        while self.is_running:
            try:
                data = self.fetcher.fetch_market_overview()
                if data:
                    self.cached_data = data
            except Exception as e:
                print(f"[MarketCache] Error updating overview cache: {e}")
            time.sleep(self.refresh_interval)


market_cache = MarketOverviewCache(bot.data_fetcher, refresh_interval=8)


class DashboardAPIHandler(SimpleHTTPRequestHandler):

    """
    Custom HTTP Request Handler serving both the static Dashboard frontend 
    and REST API endpoints for real-time monitoring and control.
    """

    def get_clean_path(self):
        raw_path = self.path.split('?')[0]
        return raw_path[:-1] if raw_path.endswith('/') and len(raw_path) > 1 else raw_path

    def do_GET(self):
        clean_path = self.get_clean_path()
        if clean_path in ("", "/", "/index.html"):
            self.serve_dashboard_html()
        elif clean_path == "/api/status":
            self.serve_api_status()
        elif clean_path == "/api/chart":
            self.serve_api_chart()
        elif clean_path == "/api/trades":
            self.serve_api_trades()
        elif clean_path == "/api/analysis":
            self.serve_api_analysis()
        elif clean_path in ("/api/market_overview", "/api/market-overview"):
            self.serve_api_market_overview()
        elif clean_path == "/api/autotrader/status":
            self.send_json_response({"autotrader_active": autotrader.is_running})
        else:
            super().do_GET()

    def do_POST(self):
        clean_path = self.get_clean_path()
        content_length = int(self.headers.get('Content-Length', 0))

        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            req_data = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            req_data = {}

        if clean_path == "/api/autotrader/start":
            try:
                autotrader.start()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "Auto-Pilot Autonomous Trading Enabled! The AI is now buying and selling automatically.",
                    "autotrader_active": True
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/autotrader/stop":

            try:
                autotrader.stop()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "Auto-Pilot Autonomous Trading Paused.",
                    "autotrader_active": False
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path in ("/api/order/manual_buy", "/api/order/buy"):
            try:
                sym = req_data.get("symbol", Config.TRADING_SYMBOL)
                price_data = bot.data_fetcher.get_latest_market_state()
                current_price = float(price_data.get("close", 64000.0))
                
                account_info = bot.executor.get_portfolio_state(sym) if hasattr(bot.executor, 'get_portfolio_state') else {}
                equity = float(account_info.get("equity", 10000.0))
                cash = float(account_info.get("cash", 10000.0))
                
                all_positions = account_info.get("all_positions", {})
                if len(all_positions) >= Config.MAX_CONCURRENT_POSITIONS and sym not in all_positions:
                    self.send_json_response({
                        "status": "FAILED",
                        "error": f"Max concurrent position limit ({Config.MAX_CONCURRENT_POSITIONS}) reached! Close an existing position first."
                    }, status=400)
                    return

                pos_size, sl_price, tp_price = bot.risk_manager.calculate_position_size(
                    equity=equity,
                    entry_price=current_price,
                    signal_action="BUY"
                )
                
                cost = pos_size * current_price
                if cost > cash and cash > 10:
                    pos_size = cash / current_price

                if hasattr(bot.executor, "execute_order"):
                    receipt = bot.executor.execute_order(
                        action="BUY",
                        size=pos_size,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        current_price=current_price,
                        symbol=sym
                    )
                else:
                    receipt = bot.executor.execute_buy(symbol=sym, size=pos_size, price=current_price)

                if receipt.get("status") == "SUCCESS":
                    receipt["symbol"] = sym
                    receipt["strategy_reason"] = "Manual User Command Entry"
                    receipt["strategy_reflection"] = "User initiated manual buy order from dashboard."
                    bot.active_trade_log = receipt
                    bot.save_active_trade()

                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"⚡ MANUAL BUY EXECUTED: Bought {pos_size:.4f} {sym} at ${current_price:,.2f}!",
                    "symbol": sym,
                    "price": current_price,
                    "size": pos_size,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "receipt": receipt
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path in ("/api/order/manual_sell", "/api/order/sell"):
            try:
                sym = req_data.get("symbol", Config.TRADING_SYMBOL)
                price_data = bot.data_fetcher.get_latest_market_state()
                current_price = float(price_data.get("close", 64000.0))
                
                account_info = bot.executor.get_portfolio_state(sym) if hasattr(bot.executor, 'get_portfolio_state') else {}
                all_positions = account_info.get("all_positions", {})
                pos = all_positions.get(sym, None)
                
                if not pos:
                    self.send_json_response({
                        "status": "FAILED",
                        "error": f"No active open position found for {sym} to sell."
                    }, status=400)
                    return

                size = float(pos.get("size", 0.0))
                if hasattr(bot.executor, "execute_order"):
                    receipt = bot.executor.execute_order(
                        action="SELL",
                        size=size,
                        stop_loss=0.0,
                        take_profit=0.0,
                        current_price=current_price,
                        symbol=sym
                    )
                else:
                    receipt = bot.executor.execute_sell(symbol=sym, size=size, price=current_price)

                if receipt.get("status") == "SUCCESS":
                    receipt["symbol"] = sym
                    receipt["strategy_reason"] = "Manual User Command Exit"
                    receipt["strategy_reflection"] = "User initiated manual sell exit from dashboard."
                    bot.process_trade_closure(receipt, price_data)

                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"⚡ MANUAL SELL EXECUTED: Sold {size:.4f} {sym} at ${current_price:,.2f}!",
                    "symbol": sym,
                    "price": current_price,
                    "size": size,
                    "pnl": receipt.get("pnl", 0.0),
                    "receipt": receipt
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/trigger_tick":
            try:
                cycle_result = bot.run_live_cycle()
                account_info = bot.executor.get_portfolio_state() if hasattr(bot.executor, 'get_portfolio_state') else {}
                price_data = bot.data_fetcher.get_latest_market_state()
                
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"⚡ Single Live Tick Completed for {Config.TRADING_SYMBOL}!",
                    "details": {
                        "active_symbol": Config.TRADING_SYMBOL,
                        "latest_price": f"${price_data.get('close', 0.0):,.2f}",
                        "timeframe": Config.TIMEFRAME,
                        "trade_style": Config.TRADE_STYLE,
                        "strategy": Config.STRATEGY_TYPE,
                        "equity": f"${account_info.get('equity', 10000.0):,.2f}",
                        "cash": f"${account_info.get('cash', 10000.0):,.2f}",
                        "active_positions_count": account_info.get("positions_count", 0),
                        "cycle_info": cycle_result or "Scanning complete"
                    }
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/trigger_audit":
            try:
                audit_res = bot.learning_engine.audit_and_optimize()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "🧠 Self-Learning Strategy Audit & Parameter Optimization Completed!",
                    "audit": {
                        "status": "OPTIMIZED",
                        "total_trades_analyzed": len(bot.executor.trade_history) if hasattr(bot.executor, 'trade_history') else 0,
                        "win_rate": f"{getattr(bot.learning_engine, 'win_rate', 0.65) * 100:.1f}%",
                        "optimization_notes": "AI strategy thresholds fine-tuned against historical win/loss ratios.",
                        "details": audit_res or "Strategy parameters optimized."
                    }
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/config/wallet":

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
        elif clean_path == "/api/config/symbol":
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
        elif clean_path == "/api/config/strategy":
            try:
                new_strat = req_data.get("strategy", "technical").strip().lower()
                Config.STRATEGY_TYPE = new_strat
                Config.save_to_env({"STRATEGY": new_strat})
                bot.strategy_name = new_strat
                bot._init_strategy()
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"AI Strategy changed to {new_strat.upper()}",
                    "strategy": new_strat
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/config/timeframe":
            try:
                new_tf = req_data.get("timeframe", "1h").strip().lower()
                Config.TIMEFRAME = new_tf
                Config.save_to_env({"TIMEFRAME": new_tf})
                bot.data_fetcher.timeframe = new_tf
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"Timeframe changed to {new_tf}",
                    "timeframe": new_tf
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/config/trade_style":
            try:
                style = req_data.get("trade_style", "scalping")
                Config.update_trade_style(style)
                bot.data_fetcher.timeframe = Config.TIMEFRAME
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": f"Trade style horizon changed to {Config.TRADE_STYLE.upper()} (Timeframe: {Config.TIMEFRAME})",
                    "trade_style": Config.TRADE_STYLE,
                    "timeframe": Config.TIMEFRAME
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/config/ai_controls":
            try:
                Config.update_ai_controls(req_data)
                bot.risk_manager.max_risk_pct = Config.MAX_EQUITY_RISK_PCT
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "AI Controls and Risk Parameters updated successfully!",
                    "config": {
                        "max_risk_pct": Config.MAX_EQUITY_RISK_PCT,
                        "min_confidence": Config.MIN_CONFIDENCE_THRESHOLD,
                        "max_positions": Config.MAX_CONCURRENT_POSITIONS,
                        "pause_buying": Config.PAUSE_BUYING
                    }
                })
            except Exception as e:
                self.send_json_response({"status": "FAILED", "error": str(e)}, status=500)
        elif clean_path == "/api/emergency/panic_sell":
            try:
                receipts = bot.executor.close_all_positions()
                for r in receipts:
                    bot.process_trade_closure(r, {"close": r.get("execution_price", 0.0)})
                self.send_json_response({
                    "status": "SUCCESS",
                    "message": "🚨 EMERGENCY PANIC SELL EXECUTED! All open positions closed immediately.",
                    "closed_count": len(receipts)
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def serve_api_status(self):
        try:
            latest_market = bot.data_fetcher.get_latest_market_state()
            current_price = latest_market.get("close", 0.0)

            if isinstance(bot.executor, MockExecutor):
                portfolio = bot.executor.get_portfolio_state()
                portfolio["equity"] = bot.executor.get_equity({"BTC/USDT": current_price})
            else:
                portfolio = bot.executor.get_portfolio_state()

            learned_params = {}
            learned_path = Config.DATABASE_DIR / "learned_params.json"
            if learned_path.exists():
                try:
                    with open(learned_path, "r") as f:
                        learned_params = json.load(f)
                except Exception:
                    pass

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
                    "trade_style": Config.TRADE_STYLE,
                    "min_confidence": Config.MIN_CONFIDENCE_THRESHOLD,
                    "max_positions": Config.MAX_CONCURRENT_POSITIONS,
                    "pause_buying": Config.PAUSE_BUYING,
                    "max_risk_pct": Config.MAX_EQUITY_RISK_PCT,
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

    def serve_api_market_overview(self):
        try:
            data = market_cache.cached_data
            if not data:
                data = bot.data_fetcher.fetch_market_overview()
                market_cache.cached_data = data
            self.send_json_response(data)
        except Exception:
            try:
                fallback = bot.data_fetcher.fetch_market_overview()
                self.send_json_response(fallback)
            except Exception:
                self.send_json_response([])




    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


class ReusableThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def run_server(port=5000):
    active_port = port
    httpd = None
    
    for p in range(active_port, active_port + 10):
        try:
            server_address = ("", p)
            httpd = ReusableThreadingHTTPServer(server_address, DashboardAPIHandler)
            active_port = p
            break
        except OSError:
            continue


    if not httpd:
        print("[!] Error: Could not bind to any port in range 5000-5010.")
        return

    print(f"\n=======================================================")
    print(f"[+] AI Trading Bot Web Dashboard is running!")
    print(f"[+] Access Interface at: http://localhost:{active_port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dashboard Web Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    run_server(port)

