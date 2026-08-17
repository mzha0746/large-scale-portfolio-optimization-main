"""Shared utilities: data I/O, plotting, performance metrics."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_DIR  = Path(__file__).parent / "outputs"
SEED        = 42
RF_ANNUAL   = 0.04


# ── performance metrics ────────────────────────────────────────────────────────

def sharpe(returns: np.ndarray, rf: float = 0.0, ann: int = 252) -> float:
    mu = np.mean(returns) * ann
    sigma = np.std(returns, ddof=1) * np.sqrt(ann)
    return (mu - rf) / sigma if sigma > 1e-12 else 0.0


def max_drawdown(cum_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_returns)
    dd   = (cum_returns - peak) / peak
    return float(dd.min())


def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Daily portfolio returns given weight vector and daily asset returns matrix."""
    return returns @ weights


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.35, "font.size": 10,
    })


def save_fig(name: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved outputs/{name}.png")


# ── covariance estimation ──────────────────────────────────────────────────────

def ledoit_wolf(ret: pd.DataFrame):
    """
    Ledoit-Wolf analytical shrinkage toward scaled identity (sklearn).
    Reduces ill-conditioning without in-sample overfitting.
    Returns (Sigma_lw_annualised, shrinkage_intensity_alpha).
    """
    from sklearn.covariance import LedoitWolf as _LW
    lw = _LW(assume_centered=False).fit(ret.values)
    return lw.covariance_ * 252, float(lw.shrinkage_)


def risk_parity(Sigma: np.ndarray) -> np.ndarray:
    """
    Equal Risk Contribution (ERC) portfolio via Roncalli (2013) log-barrier:
        min_{x > 0}  x'Σx − Σ_i log(x_i)
    The solution satisfies  x_i·(Σx)_i = constant ∀i (equal marginal risk).
    Falls back to inverse-volatility weights if solver fails.
    """
    import cvxpy as cp
    n = Sigma.shape[0]
    x = cp.Variable(n)
    obj = cp.quad_form(x, Sigma + 1e-8 * np.eye(n)) - cp.sum(cp.log(x))
    prob = cp.Problem(cp.Minimize(obj), [x >= 1e-6])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        w = 1.0 / np.sqrt(np.diag(Sigma))
        return w / w.sum()
    w = np.clip(x.value, 0, None)
    return w / w.sum()


def erc_quality(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Std-dev of risk contributions (0 = perfect ERC)."""
    RC = w * (Sigma @ w) / float(w @ Sigma @ w)
    return float(RC.std())
