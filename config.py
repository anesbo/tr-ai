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
    TRADING_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC/USD").upper()
    TIMEFRAME = os.getenv("TIMEFRAME", "1h").lower()
    
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
    
    # LLM API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
            f"Max Risk Per Trade:  {cls.MAX_EQUITY_RISK_PCT * 100}%\n"
            f"Daily Drawdown Limit:{cls.DRAWDOWN_LIMIT_PCT * 100}%\n"
            f"Default Stop-Loss:   {cls.STOP_LOSS_PCT * 100}%\n"
            f"Default Take-Profit: {cls.TAKE_PROFIT_PCT * 100}%\n"
            f"Alpaca API Configured: {bool(cls.ALPACA_API_KEY)}\n"
            f"CCXT API Configured:   {bool(cls.CCXT_API_KEY)} (Exchange: {cls.CCXT_EXCHANGE_ID})\n"
            f"Gemini API Configured: {bool(cls.GEMINI_API_KEY)}\n"
            f"OpenAI API Configured: {bool(cls.OPENAI_API_KEY)}\n"
            f"================================="
        )

if __name__ == "__main__":
    print(Config.get_summary())
