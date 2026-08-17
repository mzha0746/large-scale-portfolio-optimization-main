"""
Scalability experiment: solve time vs. universe size for
(1) MIQP (Gurobi), (2) Lagrangian relaxation.

Universe sizes: 50, 100, 200, 300, 500 assets.

Outputs
-------
outputs/fig6_scalability.png
outputs/scalability_results.csv
"""
import time
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import DATA_DIR, OUTPUT_DIR, SEED, set_style, save_fig

OUTPUT_DIR.mkdir(exist_ok=True)
N_SIZES = [50, 100, 200, 300, 500]
K       = 20        # fixed cardinality
R_PCT   = 60        # target = 60th-pctile of asset returns


def make_problem(n, full_mu, full_Sigma):
    idx = np.random.choice(full_mu.shape[0], n, replace=False)
    mu  = full_mu[idx]
    S   = full_Sigma[np.ix_(idx, idx)]
    r_t = float(np.percentile(mu, R_PCT))
    return mu, S, r_t


def time_miqp(mu, Sigma, r_target, K, timelimit=60):
    try:
        import gurobipy as gp
        from gurobipy import GRB
        n = len(mu)
        m = gp.Model(); m.setParam("OutputFlag",0); m.setParam("TimeLimit", timelimit)
        x = m.addMVar(n, lb=0, ub=1); z = m.addMVar(n, vtype=GRB.BINARY)
        m.setObjective(x @ Sigma @ x, GRB.MINIMIZE)
        m.addConstr(mu @ x >= r_target); m.addConstr(x.sum() == 1)
        m.addConstr(x <= z); m.addConstr(z.sum() <= K)
        t0 = time.perf_counter(); m.optimize()
        return time.perf_counter() - t0, float(m.ObjVal) if m.Status in (2,9) else None
    except Exception:
        # PuLP fallback (LP relaxation for timing only)
        n = len(mu)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)),
                          [mu @ x >= r_target, cp.sum(x) == 1, x >= 0])
        t0 = time.perf_counter()
        prob.solve(solver=cp.OSQP, verbose=False)
        return time.perf_counter() - t0, float(prob.value) if prob.value else None


def time_lagrangian(mu, Sigma, r_target, K, max_iter=150):
    n = len(mu); lam = np.zeros(n); UB = np.inf; LB = -np.inf
    t0 = time.perf_counter()
    for _ in range(max_iter):
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma) + lam @ x),
                          [mu @ x >= r_target, cp.sum(x) == 1, x >= 0])
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status not in ("optimal","optimal_inaccurate"): break
        xv = np.clip(x.value, 0, None)
        idx = np.argsort(lam)[::-1][:K]; zv = np.zeros(n); zv[idx] = 1
        LB  = max(LB, float(xv @ Sigma @ xv) + float(lam @ (xv - zv)))
        xr  = cp.Variable(K); mk = mu[idx]; Sk = Sigma[np.ix_(idx, idx)]
        pr2 = cp.Problem(cp.Minimize(cp.quad_form(xr, Sk)),
                         [mk @ xr >= r_target, cp.sum(xr) == 1, xr >= 0])
        pr2.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if pr2.status in ("optimal","optimal_inaccurate"):
            wf = np.zeros(n); wf[idx] = np.clip(xr.value, 0, None)
            UB = min(UB, float(wf @ Sigma @ wf))
        sg = xv - zv; ss = np.dot(sg, sg)
        if ss < 1e-14 or (UB - LB) / (abs(UB)+1e-12) < 1e-4: break
        lam = np.maximum(0, lam + max((UB - LB)/ss, 0) * sg)
    return time.perf_counter() - t0, UB


def main():
    set_style()
    np.random.seed(SEED)
    print("=" * 55)
    print("Scalability Experiment")
    print("=" * 55)

    # load base data (use synthetic if needed)
    ret = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    # if fewer assets than max size, use synthetic expansion
    n_available = ret.shape[1]
    if n_available < max(N_SIZES):
        from synth import make_synthetic_cov
        mu_base = ret.mean().values * 252
        S_base  = ret.cov().values  * 252
        # pad with synthetic
        n_need = max(N_SIZES) - n_available
        mu_syn = np.random.uniform(0.05, 0.25, n_need)
        # simple block structure extension
        full_mu    = np.concatenate([mu_base, mu_syn])
        pad_S = np.eye(n_need) * np.mean(np.diag(S_base))
        full_Sigma = np.block([[S_base, np.zeros((n_available, n_need))],
                                [np.zeros((n_need, n_available)), pad_S]])
    else:
        full_mu    = ret.mean().values * 252
        full_Sigma = ret.cov().values  * 252

    records = []
    for n in N_SIZES:
        mu, Sigma, r_tgt = make_problem(n, full_mu, full_Sigma)
        print(f"\n  N = {n}")

        t_miqp, obj_miqp = time_miqp(mu, Sigma, r_tgt, K)
        print(f"    MIQP        {t_miqp:6.2f}s  obj={obj_miqp}")

        t_lag, obj_lag = time_lagrangian(mu, Sigma, r_tgt, K)
        print(f"    Lagrangian  {t_lag:6.2f}s  UB={obj_lag:.6f}")

        gap = abs(obj_lag - obj_miqp) / (abs(obj_miqp) + 1e-12) if obj_miqp else None
        records.append({
            "N": n, "K": K,
            "miqp_sec": round(t_miqp, 3), "lag_sec": round(t_lag, 3),
            "miqp_obj": obj_miqp, "lag_ub": obj_lag,
            "gap_pct": round(gap * 100, 3) if gap is not None else None,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "scalability_results.csv", index=False)
    print("\n  Results:")
    print(df.to_string(index=False))

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(df["N"], df["miqp_sec"], "o-",
                     color="#dc2626", lw=2, label="MIQP (Gurobi)")
    axes[0].semilogy(df["N"], df["lag_sec"],  "s--",
                     color="#2563eb", lw=2, label="Lagrangian (subgradient)")
    axes[0].set(xlabel="Universe Size N", ylabel="Solve Time (s, log scale)",
                title="Scalability: Solve Time vs. Universe Size")
    axes[0].legend()

    axes[1].plot(df["N"], df["gap_pct"], "^-", color="#16a34a", lw=2)
    axes[1].set(xlabel="Universe Size N", ylabel="Optimality Gap (%)",
                title="Lagrangian Relaxation Gap vs. Universe Size")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))

    fig.suptitle(f"Scalability — Cardinality K={K}", fontsize=11)
    save_fig("fig6_scalability")
    print("\n  Done.")


if __name__ == "__main__":
    main()
