import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Workspace Directory Structure
    BASE_DIR = Path(__file__).resolve().parent
    DATABASE_DIR = BASE_DIR / "database"
    MODELS_DIR = BASE_DIR / "models"
    
    # Ensure standard directories exist
    DATABASE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Database Paths
    TRADE_LEDGER_PATH = DATABASE_DIR / "trade_ledger.json"
    
    # Global Bot Settings
    PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")
    EXECUTION_ENGINE = os.getenv("EXECUTION_ENGINE", "mock").lower()
    STRATEGY_TYPE = os.getenv("STRATEGY", "technical").lower()
    TRADING_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC/USDT").upper()
    TIMEFRAME = os.getenv("TIMEFRAME", "1h").lower()
    
    # Trade Style & Horizon Settings
    TRADE_STYLE = os.getenv("TRADE_STYLE", "scalping").lower()  # "scalping" (short-term), "day_trading" (medium), "swing" (long-term)
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.60"))
    MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
    PAUSE_BUYING = os.getenv("PAUSE_BUYING", "False").lower() in ("true", "1", "yes")

    # Risk Management & Limits
    MAX_EQUITY_RISK_PCT = float(os.getenv("MAX_EQUITY_RISK_PCT", "0.02"))  # default 2%
    DRAWDOWN_LIMIT_PCT = float(os.getenv("DRAWDOWN_LIMIT_PCT", "0.05"))    # default 5%
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))            # default 1.5%
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))          # default 3%

    
    # Alpaca Credentials
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_IS_PAPER = os.getenv("ALPACA_IS_PAPER", "True").lower() in ("true", "1", "yes")
    
    # CCXT Credentials
    CCXT_EXCHANGE_ID = os.getenv("CCXT_EXCHANGE_ID", "binance").lower()
    CCXT_API_KEY = os.getenv("CCXT_API_KEY", "")
    CCXT_SECRET_KEY = os.getenv("CCXT_SECRET_KEY", "")
    CCXT_PASSWORD = os.getenv("CCXT_PASSWORD", "")
    
    # Web3 / Wallet Config
    WEB3_WALLET_ADDRESS = os.getenv("WEB3_WALLET_ADDRESS", "")
    WEB3_NETWORK = os.getenv("WEB3_NETWORK", "bsc_testnet")
    WALLET_TYPE = os.getenv("WALLET_TYPE", "mock")
    
    # LLM API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    @classmethod
    def save_to_env(cls, env_updates: dict):
        """Helper to write config updates to .env file."""
        env_file = cls.BASE_DIR / ".env"
        existing = {}
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        existing[k.strip()] = v.strip()
        
        for k, v in env_updates.items():
            existing[k] = str(v)
            os.environ[k] = str(v)
            
        with open(env_file, "w") as f:
            for k, v in existing.items():
                f.write(f"{k}={v}\n")

    @classmethod
    def update_symbol(cls, symbol: str):
        """Dynamically updates the trading symbol."""
        cls.TRADING_SYMBOL = symbol.upper()
        cls.save_to_env({"TRADING_SYMBOL": cls.TRADING_SYMBOL})

    @classmethod
    def update_wallet_config(cls, wallet_data: dict):
        """Dynamically updates execution and wallet settings."""
        updates = {}
        if "execution_engine" in wallet_data:
            cls.EXECUTION_ENGINE = wallet_data["execution_engine"].lower()
            updates["EXECUTION_ENGINE"] = cls.EXECUTION_ENGINE
        if "paper_trading" in wallet_data:
            cls.PAPER_TRADING = bool(wallet_data["paper_trading"])
            updates["PAPER_TRADING"] = "True" if cls.PAPER_TRADING else "False"
        if "web3_wallet_address" in wallet_data:
            cls.WEB3_WALLET_ADDRESS = wallet_data["web3_wallet_address"]
            updates["WEB3_WALLET_ADDRESS"] = cls.WEB3_WALLET_ADDRESS
        if "web3_network" in wallet_data:
            cls.WEB3_NETWORK = wallet_data["web3_network"]
            updates["WEB3_NETWORK"] = cls.WEB3_NETWORK
        if "wallet_type" in wallet_data:
            cls.WALLET_TYPE = wallet_data["wallet_type"]
            updates["WALLET_TYPE"] = cls.WALLET_TYPE
        if "ccxt_exchange_id" in wallet_data:
            cls.CCXT_EXCHANGE_ID = wallet_data["ccxt_exchange_id"].lower()
            updates["CCXT_EXCHANGE_ID"] = cls.CCXT_EXCHANGE_ID
        if "ccxt_api_key" in wallet_data and wallet_data["ccxt_api_key"]:
            cls.CCXT_API_KEY = wallet_data["ccxt_api_key"]
            updates["CCXT_API_KEY"] = cls.CCXT_API_KEY
        if "ccxt_secret_key" in wallet_data and wallet_data["ccxt_secret_key"]:
            cls.CCXT_SECRET_KEY = wallet_data["ccxt_secret_key"]
            updates["CCXT_SECRET_KEY"] = cls.CCXT_SECRET_KEY
        if "alpaca_api_key" in wallet_data and wallet_data["alpaca_api_key"]:
            cls.ALPACA_API_KEY = wallet_data["alpaca_api_key"]
            updates["ALPACA_API_KEY"] = cls.ALPACA_API_KEY
        if "alpaca_secret_key" in wallet_data and wallet_data["alpaca_secret_key"]:
            cls.ALPACA_SECRET_KEY = wallet_data["alpaca_secret_key"]
            updates["ALPACA_SECRET_KEY"] = cls.ALPACA_SECRET_KEY

        cls.save_to_env(updates)

    @classmethod
    def update_trade_style(cls, style: str):
        """Updates trade style horizon (scalping, day_trading, swing) and sets matching timeframe and stop parameters."""
        style = style.lower()
        cls.TRADE_STYLE = style
        updates = {"TRADE_STYLE": style}
        
        if style == "scalping":
            cls.TIMEFRAME = "5m"
            cls.STOP_LOSS_PCT = 0.008  # 0.8%
            cls.TAKE_PROFIT_PCT = 0.015  # 1.5%
        elif style == "day_trading":
            cls.TIMEFRAME = "1h"
            cls.STOP_LOSS_PCT = 0.015  # 1.5%
            cls.TAKE_PROFIT_PCT = 0.03  # 3.0%
        elif style == "swing":
            cls.TIMEFRAME = "1d"
            cls.STOP_LOSS_PCT = 0.03   # 3.0%
            cls.TAKE_PROFIT_PCT = 0.07   # 7.0%

        updates["TIMEFRAME"] = cls.TIMEFRAME
        updates["STOP_LOSS_PCT"] = str(cls.STOP_LOSS_PCT)
        updates["TAKE_PROFIT_PCT"] = str(cls.TAKE_PROFIT_PCT)
        cls.save_to_env(updates)

    @classmethod
    def update_ai_controls(cls, controls: dict):
        """Updates granular AI parameters (risk pct, confidence threshold, max positions, pause buying)."""
        updates = {}
        if "max_risk_pct" in controls:
            cls.MAX_EQUITY_RISK_PCT = float(controls["max_risk_pct"])
            updates["MAX_EQUITY_RISK_PCT"] = str(cls.MAX_EQUITY_RISK_PCT)
        if "min_confidence" in controls:
            cls.MIN_CONFIDENCE_THRESHOLD = float(controls["min_confidence"])
            updates["MIN_CONFIDENCE_THRESHOLD"] = str(cls.MIN_CONFIDENCE_THRESHOLD)
        if "max_positions" in controls:
            cls.MAX_CONCURRENT_POSITIONS = int(controls["max_positions"])
            updates["MAX_CONCURRENT_POSITIONS"] = str(cls.MAX_CONCURRENT_POSITIONS)
        if "pause_buying" in controls:
            cls.PAUSE_BUYING = bool(controls["pause_buying"])
            updates["PAUSE_BUYING"] = "True" if cls.PAUSE_BUYING else "False"
            
        cls.save_to_env(updates)


    @classmethod
    def get_summary(cls):
        """Returns a string summary of the configuration, hiding sensitive credentials."""
        return (
            f"=== TRADING BOT CONFIGURATION ===\n"
            f"Paper Trading:       {cls.PAPER_TRADING}\n"
            f"Execution Engine:    {cls.EXECUTION_ENGINE}\n"
            f"Strategy Type:       {cls.STRATEGY_TYPE}\n"
            f"Trading Symbol:      {cls.TRADING_SYMBOL}\n"
            f"Timeframe:           {cls.TIMEFRAME}\n"
            f"Web3 Wallet:         {cls.WEB3_WALLET_ADDRESS or 'Not Connected'}\n"
            f"Web3 Network:        {cls.WEB3_NETWORK}\n"
            f"Alpaca API Configured: {bool(cls.ALPACA_API_KEY)}\n"
            f"CCXT API Configured:   {bool(cls.CCXT_API_KEY)} (Exchange: {cls.CCXT_EXCHANGE_ID})\n"
            f"================================="
        )

if __name__ == "__main__":
    print(Config.get_summary())

