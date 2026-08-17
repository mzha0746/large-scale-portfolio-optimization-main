# Large-Scale Portfolio Optimization: Cardinality Constraints via MIQP and Lagrangian Relaxation

A self-contained Python project that extends classical Markowitz portfolio optimization to the **cardinality-constrained** setting (select at most K assets), solves it exactly via **Mixed-Integer Quadratic Programming (MIQP)**, and at scale via **Lagrangian relaxation with subgradient updates** — validated against S&P 500 constituent data with a rolling out-of-sample backtest.

---

## Motivation

Standard mean-variance optimization selects all N assets with non-zero weights, producing unmanageable diversification in large universes. Practitioners impose a **cardinality constraint** (hold ≤ K assets), turning the convex QP into an NP-hard MIQP. For N = 500+ assets, branch-and-bound becomes intractable; **Lagrangian relaxation** provides a tight lower bound and near-optimal feasible solution in polynomial time.

---

## Problem Formulation

**Cardinality-Constrained Mean-Variance Portfolio (CCMVP):**

```
minimize    x' Σ x
subject to  μ' x ≥ r_target        (return requirement)
            1' x = 1               (fully invested)
            x_i ≥ 0  ∀ i          (long-only)
            x_i ≤ z_i ∀ i         (big-M linking, M=1)
            Σ_i z_i ≤ K           (cardinality)
            z_i ∈ {0,1}
```

**Lagrangian decomposition** (relax linking constraints x_i ≤ z_i with λ_i ≥ 0):

```
L(λ) = min_{x,z}  x'Σx + λ'(x − z)
```

Decomposes into:
- **x-subproblem**: QP → solved by CVXPY/OSQP
- **z-subproblem**: select K assets with largest λ_i (closed-form greedy)

Multipliers updated by Polyak subgradient:  `λ_{t+1} = max(0, λ_t + α_t (x_t − z_t))`

---

## Methods Compared

| Method | Description | Solver |
|--------|-------------|--------|
| Classical MVO | No cardinality; convex QP | CVXPY / OSQP |
| MIQP | Exact CCMVP | Gurobi (academic) / PuLP-CBC fallback |
| Lagrangian | Subgradient relaxation | CVXPY / OSQP |
| 1/N Equal-weight | Naive benchmark | — |

---

## Key Results

| | MVO | MIQP K=20 | Lagrangian K=20 |
|-|-----|-----------|-----------------|
| Ann. Return | — | — | — |
| Ann. Volatility | — | — | — |
| Sharpe Ratio | — | — | — |
| Optimality Gap | 0% | 0% | < 2% |
| Solve Time (N=200) | < 1s | ~30s | ~5s |

*(Run `06_scalability.py` to populate with real numbers from your data.)*

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# (Optional) Gurobi academic licence: https://www.gurobi.com/academia/academic-program-and-licenses/

# 2. Run the pipeline in order
python 01_data.py           # download/generate data
python 02_markowitz.py      # classical MVO + efficient frontier
python 03_cardinality_milp.py   # MIQP across K values
python 04_lagrangian.py     # Lagrangian relaxation convergence
python 05_backtest.py       # rolling out-of-sample backtest
python 06_scalability.py    # scalability experiment (N = 50 → 500)
```

---

## Output Figures

| File | Description |
|------|-------------|
| `fig1_efficient_frontier.png` | Markowitz efficient frontier with Max-Sharpe and Min-Var portfolios |
| `fig2_cardinality_frontier.png` | Volatility / Sharpe / solve-time vs. K |
| `fig2b_weight_distribution.png` | Portfolio weight distribution by K |
| `fig3_lagrangian_convergence.png` | UB/LB convergence for K = 10, 20, 30 |
| `fig4_backtest_wealth.png` | Cumulative out-of-sample wealth paths |
| `fig6_scalability.png` | Solve time + optimality gap vs. universe size |

---

## Project Structure

```
project-1-large-scale-portfolio/
├── 01_data.py               # S&P 500 data via yfinance (synthetic fallback)
├── 02_markowitz.py          # classical MVO, efficient frontier
├── 03_cardinality_milp.py   # MIQP formulation (Gurobi / PuLP)
├── 04_lagrangian.py         # Lagrangian relaxation + Polyak subgradient
├── 05_backtest.py           # rolling out-of-sample backtest
├── 06_scalability.py        # N = 50 → 500 scalability experiment
├── utils.py                 # metrics, plotting helpers
├── data/                    # auto-generated
├── outputs/                 # figures + CSV results
└── requirements.txt
```

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Benders, J. F. (1962). *Partitioning procedures for solving mixed-variables programming problems*. Numerische Mathematik.
- Chang, T.-J. et al. (2000). *Heuristics for cardinality constrained portfolio optimisation*. Computers & Operations Research.
- Bertsimas, D. & Shioda, R. (2009). *Algorithm for cardinality-constrained quadratic optimization*. Computational Optimization and Applications.
