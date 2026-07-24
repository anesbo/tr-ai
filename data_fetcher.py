import numpy as np
import pandas as pd
import datetime
import time
from config import Config

# Dynamic import helpers for CCXT and Alpaca to avoid import errors if not installed
try:
    import ccxt
except ImportError:
    ccxt = None

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False


class DataFetcher:
    """
    Data Ingestion Layer.
    Responsible for:
      - Fetching historical OHLCV data.
      - Ingesting real-time market data ticks.
      - Calculating custom technical indicators in pure Pandas/Numpy.
    """
    
    def __init__(self):
        self.symbol = Config.TRADING_SYMBOL
        self.timeframe = Config.TIMEFRAME
        self.ccxt_client = None
        self.alpaca_client = None
        self._init_clients()
        
    def _init_clients(self):
        """Initializes the exchange/broker clients based on the configuration."""
        # 1. CCXT Client Setup
        if ccxt is not None:
            if Config.EXECUTION_ENGINE == "ccxt" and Config.CCXT_API_KEY:
                exchange_class = getattr(ccxt, Config.CCXT_EXCHANGE_ID)
                self.ccxt_client = exchange_class({
                    'apiKey': Config.CCXT_API_KEY,
                    'secret': Config.CCXT_SECRET_KEY,
                    'password': Config.CCXT_PASSWORD,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future' if 'perpetual' in Config.TRADING_SYMBOL.lower() else 'spot'}
                })
                if Config.PAPER_TRADING:
                    if hasattr(self.ccxt_client, 'set_sandbox_mode'):
                        self.ccxt_client.set_sandbox_mode(True)
            else:
                # Public read-only CCXT binance client for dry runs / mock engines
                self.ccxt_client = ccxt.binance({'enableRateLimit': True})
        
        # 2. Alpaca Client Setup
        if HAS_ALPACA and Config.EXECUTION_ENGINE == "alpaca":
            # For historical data, Alpaca has separate clients for stock and crypto
            api_key = Config.ALPACA_API_KEY
            secret_key = Config.ALPACA_SECRET_KEY
            
            # Simple heuristic to determine if symbol is crypto
            is_crypto = "/" in self.symbol or self.symbol.endswith("USD") and self.symbol not in ("SPY", "QQQ", "AAPL", "MSFT", "TSLA")
            if is_crypto:
                self.alpaca_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
                self._alpaca_type = "crypto"
            else:
                self.alpaca_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
                self._alpaca_type = "stock"

    def fetch_historical_data(self, limit=100) -> pd.DataFrame:
        """
        Fetches historical OHLCV data.
        Returns a Pandas DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        if Config.EXECUTION_ENGINE == "alpaca" and HAS_ALPACA and Config.ALPACA_API_KEY:
            return self._fetch_alpaca_historical(limit)
        else:
            # Default fallback/mock engine uses CCXT public endpoints
            return self._fetch_ccxt_historical(limit)

    def _fetch_ccxt_historical(self, limit=100) -> pd.DataFrame:
        """Helper to fetch from CCXT with automatic symbol format fallback."""
        if not self.ccxt_client:
            raise ValueError("CCXT is not installed or initialized.")
        
        sym = self.symbol
        if "/" not in sym:
            if sym.endswith("USDT"):
                sym = sym[:-4] + "/USDT"
            elif sym.endswith("USD"):
                sym = sym[:-3] + "/USD"

        # Try primary symbol and fallback (e.g. XRP/USD -> XRP/USDT)
        symbols_to_try = [sym]
        if sym.endswith("/USD"):
            symbols_to_try.append(sym.replace("/USD", "/USDT"))
        elif sym.endswith("/USDT"):
            symbols_to_try.append(sym.replace("/USDT", "/USD"))

        for s in symbols_to_try:
            try:
                ohlcv = self.ccxt_client.fetch_ohlcv(s, timeframe=self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception:
                continue

        # Generate synthetic data if all network attempts fail
        return self._generate_synthetic_data(limit)


    def _fetch_alpaca_historical(self, limit=100) -> pd.DataFrame:
        """Helper to fetch from Alpaca."""
        # Map simple timeframes to Alpaca timeframes
        tf_map = {
            "1m": AlpacaTimeFrame.Minute,
            "5m": AlpacaTimeFrame.Minute * 5,
            "15m": AlpacaTimeFrame.Minute * 15,
            "1h": AlpacaTimeFrame.Hour,
            "1d": AlpacaTimeFrame.Day
        }
        alpaca_tf = tf_map.get(self.timeframe, AlpacaTimeFrame.Hour)
        
        # Estimate start time based on limit and timeframe
        now = datetime.datetime.now(datetime.timezone.utc)
        if self.timeframe == "1m":
            start = now - datetime.timedelta(minutes=limit * 2)
        elif self.timeframe == "5m":
            start = now - datetime.timedelta(minutes=limit * 10)
        elif self.timeframe == "15m":
            start = now - datetime.timedelta(minutes=limit * 30)
        elif self.timeframe == "1h":
            start = now - datetime.timedelta(hours=limit * 2)
        else:
            start = now - datetime.timedelta(days=limit * 2)
            
        try:
            # Alpaca API expects a list of symbols
            clean_symbol = self.symbol.replace("/", "")
            if self._alpaca_type == "crypto":
                req = CryptoBarsRequest(
                    symbol_or_symbols=[clean_symbol],
                    timeframe=alpaca_tf,
                    start=start,
                    limit=limit
                )
                bars = self.alpaca_client.get_crypto_bars(req)
            else:
                req = StockBarsRequest(
                    symbol_or_symbols=[clean_symbol],
                    timeframe=alpaca_tf,
                    start=start,
                    limit=limit
                )
                bars = self.alpaca_client.get_stock_bars(req)
                
            df = bars.df.reset_index()
            # Rename columns to standardized names
            df = df.rename(columns={'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.tail(limit)
        except Exception as e:
            print(f"[DataFetcher] Alpaca fetch error ({e}), falling back to CCXT...")
            return self._fetch_ccxt_historical(limit)

    def _generate_synthetic_data(self, limit=100) -> pd.DataFrame:
        """Generates synthetic price data for simulation testing when APIs are unavailable."""
        timestamps = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(limit)]
        timestamps.reverse()
        
        # Simple random walk starting at $60,000 for crypto or $150 for stocks
        start_price = 60000.0 if "BTC" in self.symbol else 150.0
        prices = [start_price]
        
        np.random.seed(42)  # Deterministic mock data
        for _ in range(1, limit):
            change = np.random.normal(0, start_price * 0.005)
            prices.append(max(start_price * 0.1, prices[-1] + change))
            
        df_data = []
        for i, price in enumerate(prices):
            noise = np.random.uniform(-price * 0.002, price * 0.002)
            o = price + noise
            c = price - noise
            h = max(o, c) + np.random.uniform(0, price * 0.004)
            l = min(o, c) - np.random.uniform(0, price * 0.004)
            v = np.random.uniform(10, 500)
            df_data.append([timestamps[i], o, h, l, c, v])
            
        return pd.DataFrame(df_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # =====================================================================
    # Technical Indicator Calculation Engine (Pure Python / Pandas / Numpy)
    # =====================================================================
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates RSI, MACD, Bollinger Bands, and ATR on the given DataFrame."""
        df = df.copy()
        
        # 1. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Wilder's smoothing/EMA
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (Moving Average Convergence Divergence)
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 3. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        
        # 4. ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.ewm(span=14, adjust=False).mean()
        
        # Fill leading NaN values from calculations
        df = df.bfill().ffill()
        return df

    def get_latest_market_state(self) -> dict:
        """Fetches historical data, computes indicators, and returns the last state row as a dict."""
        df = self.fetch_historical_data(limit=100)
        df_indicators = self.calculate_indicators(df)
        latest_row = df_indicators.iloc[-1].to_dict()
        # Add symbol metadata
        latest_row['symbol'] = self.symbol
        latest_row['timeframe'] = self.timeframe
        return latest_row

    def fetch_market_overview(self, symbol_list: list = None) -> list:
        """
        Scans a list of market coins and returns price, 24h change %, RSI, and signal recommendation for each.
        """
        if not symbol_list:
            symbol_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT"]


        results = []
        original_symbol = self.symbol

        for sym in symbol_list:
            self.symbol = sym
            try:
                df = self.fetch_historical_data(limit=30)
                df_ind = self.calculate_indicators(df)
                latest = df_ind.iloc[-1]
                prev_close = df_ind.iloc[-2]['close'] if len(df_ind) > 1 else latest['close']
                
                price = float(latest['close'])
                prev = float(prev_close)
                change_24h = ((price - prev) / prev * 100.0) if prev > 0 else 0.0
                
                rsi = float(latest.get('rsi', 50.0))
                macd = float(latest.get('macd', 0.0))
                macd_signal = float(latest.get('macd_signal', 0.0))

                signal = "HOLD"
                if rsi <= 35 or (macd > macd_signal and rsi < 50):
                    signal = "BUY"
                elif rsi >= 65 or (macd < macd_signal and rsi > 50):
                    signal = "SELL"

                results.append({
                    "symbol": sym,
                    "price": round(price, 4 if price < 10 else 2),
                    "change_24h": round(change_24h, 2),
                    "rsi": round(rsi, 1),
                    "signal": signal,
                    "active": sym.upper() == original_symbol.upper()
                })
            except Exception as e:
                results.append({
                    "symbol": sym,
                    "price": 0.0,
                    "change_24h": 0.0,
                    "rsi": 50.0,
                    "signal": "N/A",
                    "active": sym.upper() == original_symbol.upper()
                })
            finally:
                self.symbol = original_symbol

        return results



if __name__ == "__main__":
    fetcher = DataFetcher()
    print(f"Fetching data for {fetcher.symbol} ({fetcher.timeframe})...")
    try:
        latest = fetcher.get_latest_market_state()
        print("\n--- LATEST MARKET STATE ---")
        for k, v in latest.items():
            if isinstance(v, float):
                print(f"{k:12}: {v:.4f}")
            else:
                print(f"{k:12}: {v}")
    except Exception as e:
        print(f"Error testing DataFetcher: {e}")
