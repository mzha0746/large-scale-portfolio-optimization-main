"""
Download S&P 500 constituent prices via yfinance.
Falls back to synthetic correlated returns if network is unavailable.

Outputs
-------
data/returns.csv      : daily log-returns (T × N)
data/meta.json        : ticker list + date range
"""
import warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import DATA_DIR, SEED

# ── S&P 500 ticker universe ───────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","JNJ","MA","PG","HD","MRK","AVGO","CVX",
    "PEP","KO","ABBV","COST","MCD","WMT","CSCO","ABT","ACN","TMO",
    "CRM","BAC","NEE","NFLX","LIN","DHR","ADBE","TXN","CMCSA","WFC",
    "PM","RTX","ORCL","AMGN","AMD","INTU","HON","QCOM","LOW","IBM",
    "GS","CAT","SPGI","MS","AXP","BLK","ISRG","MDLZ","ELV","DE",
    "ADI","BKNG","GILD","C","CI","VRTX","PLD","ZTS","REGN","CB",
    "SBUX","SO","SCHW","MO","TJX","AMAT","MMC","AON","CME","DUK",
    "SYK","BMY","EOG","ETN","PNC","GE","APD","CL","NOC","ITW",
    "FCX","USB","PSA","NSC","FDX","AFL","EW","MCO","CSX","PCAR",
]

START = "2019-01-01"
END   = "2024-12-31"


def download_real(tickers, start, end):
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          progress=False, auto_adjust=True)["Close"]
        raw = raw.dropna(axis=1, thresh=int(0.95 * len(raw)))  # drop thin data
        returns = np.log(raw / raw.shift(1)).dropna()
        return returns
    except Exception as e:
        print(f"  yfinance failed ({e}), using synthetic data.")
        return None


def make_synthetic(n: int = 100, t: int = 1260) -> pd.DataFrame:
    """Correlated synthetic daily log-returns via Cholesky."""
    rng = np.random.default_rng(SEED)
    # random correlation structure
    A = rng.standard_normal((n, 5))          # 5 latent factors
    cov_latent = A @ A.T / 5 + np.diag(rng.uniform(0.5, 1.5, n))
    cov_latent = cov_latent / np.outer(np.sqrt(np.diag(cov_latent)),
                                        np.sqrt(np.diag(cov_latent)))
    vol = rng.uniform(0.12, 0.40, n) / np.sqrt(252)
    cov = np.outer(vol, vol) * cov_latent
    mu  = rng.uniform(0.05, 0.25, n) / 252
    L   = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    Z   = rng.standard_normal((t, n))
    rets = mu + Z @ L.T
    tickers = [f"S{i:03d}" for i in range(n)]
    dates   = pd.bdate_range("2019-01-02", periods=t)
    return pd.DataFrame(rets, index=dates, columns=tickers)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Fetching data …")

    returns = download_real(SP500_TICKERS, START, END)
    source  = "yfinance"

    if returns is None or len(returns.columns) < 50:
        print("  Using synthetic data (100 assets, 5-year daily).")
        returns = make_synthetic(n=100, t=1260)
        source  = "synthetic"

    # clip extreme outliers (>5σ)
    z = (returns - returns.mean()) / returns.std()
    returns = returns[np.abs(z) < 5].dropna()

    returns.to_csv(DATA_DIR / "returns.csv")
    meta = {
        "source":  source,
        "tickers": list(returns.columns),
        "n_assets": len(returns.columns),
        "n_days":   len(returns),
        "start":    str(returns.index[0].date()),
        "end":      str(returns.index[-1].date()),
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {meta['n_assets']} assets × {meta['n_days']} days  [{source}]")
    print(f"  Saved → data/returns.csv, data/meta.json")


if __name__ == "__main__":
    main()
